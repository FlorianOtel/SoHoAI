"""
Smart model routing — decides which LLM handles each request.

Routing strategy (post 2026-04-22 flip):
  1. If user explicitly requests a model → use it
  2. If force_cloud flag → use cloud (kept for API compat; cloud is now default anyway)
  3. If message context is very long → use cloud (better context handling)
  4. Default → external (Sonnet 4.6, cloud); LiteLLM falls back to internal
     (local Qwen3.5 via llama-server) if the external call fails.

LiteLLM handles the actual fallback retries; this module decides
the *starting point* and any pre-routing logic.

Anthropic prompt caching is applied when the selected model is served by
Anthropic (`_uses_anthropic()` checks the litellm_params.model prefix).
This is provider-agnostic: swapping external to Gemini/OpenAI in SoHoAI-config.yaml
automatically disables the Anthropic-specific cache_control injection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import litellm
import yaml
from litellm import Router

if TYPE_CHECKING:
    from chat_store import ChatStore

from usage_tracker import UsageTracker

logger = logging.getLogger(__name__)


class SmartRouter:
    """Wraps LiteLLM Router with SoHoAI-specific routing logic."""

    def __init__(self, config_path: str = "SoHoAI-config.yaml", store: ChatStore | None = None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Build dynamic URLs from server IPs
        server2_ip = self.config.get("server2_ip", "192.168.1.95")
        for model in self.config["model_list"]:
            if "api_base" in model.get("litellm_params", {}):
                # Replace placeholder with actual IP
                api_base = model["litellm_params"]["api_base"]
                if "${server2_ip}" in api_base:
                    model["litellm_params"]["api_base"] = api_base.replace("${server2_ip}", server2_ip)

        router_settings = self.config.get("router_settings", {})
        litellm_settings = self.config.get("litellm_settings", {})
        context_window_fallbacks = litellm_settings.get("context_window_fallbacks")

        # Instantiate usage tracker if store is provided
        self._usage_tracker = UsageTracker(store) if store is not None else None

        self.litellm_router = Router(
            model_list=self.config["model_list"],
            routing_strategy="simple-shuffle",
            # Fallback: if external (cloud) fails, go to internal (local)
            fallbacks=[
                {
                    "anthropic/claude-sonnet-4-6": ["internal/qwen3-4b"],
                }
            ],
            # Retry config
            num_retries=2,
            timeout=60,
            # From SoHoAI-config.yaml router_settings / litellm_settings
            enable_pre_call_checks=router_settings.get("enable_pre_call_checks", False),
            # Usage tracking via custom logger
            **({"context_window_fallbacks": context_window_fallbacks} if context_window_fallbacks else {}),
        )

        # Register tracker with global litellm callbacks list
        # Guard prevents double-registration on hot-reload cycles
        if self._usage_tracker is not None:
            import litellm as _litellm
            if self._usage_tracker not in _litellm.callbacks:
                _litellm.callbacks.append(self._usage_tracker)

        self.routing_config = self.config.get("routing", {})
        self.default_model = self.routing_config.get("default_model", "internal/qwen3-4b")
        self.cloud_model = self.routing_config.get("cloud_model", "anthropic/claude-sonnet-4-6")
        self.complexity_threshold = self.routing_config.get(
            "complexity_threshold_tokens", 2000
        )

    def select_model(
        self,
        messages: list[dict],
        requested_model: Optional[str] = None,
        force_cloud: bool = False,
    ) -> str:
        """
        Decide which model to target.

        Returns the model alias to pass to LiteLLM.
        """
        # Explicit request
        if requested_model:
            logger.info(f"Explicit model request: {requested_model}")
            return requested_model

        # Force cloud for critical tasks
        if force_cloud:
            logger.info("Cloud forced by request flag")
            return self.cloud_model

        # Estimate complexity by total token count (rough: 1 token ≈ 4 chars)
        total_chars = sum(len(m.get("content", "")) for m in messages)
        est_tokens = total_chars // 4
        if est_tokens > self.complexity_threshold:
            logger.info(
                f"Context size ({est_tokens} est. tokens) exceeds threshold "
                f"({self.complexity_threshold}), routing to cloud"
            )
            return self.cloud_model

        return self.default_model

    async def complete(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        force_cloud: bool = False,
        stream: bool = False,
        **kwargs,
    ) -> dict:
        """
        Run a chat completion with smart routing + fallback.

        Returns the LiteLLM response dict (OpenAI-compatible format).
        """
        target = self.select_model(messages, model, force_cloud)
        logger.info(f"Routing to model: {target}")

        # Anthropic prompt caching: inject ephemeral cache_control breakpoints
        # only when the target is served via Anthropic API. Skipped for all
        # other providers (Gemini, OpenAI, internal llama-server, etc.).
        messages_to_send = (
            self._apply_cache_control(messages) if self._uses_anthropic(target) else messages
        )

        if target.startswith("ollama-cloud/"):
            # 3-step geometric backoff: each successive attempt allows more time.
            # Timeouts: 60s → 90s → 120s. Worst case: 270s total (within CC's 300s httpx limit).
            # Starting at 60s covers both fast (<10s) and normal-slow (30-60s) responses in one attempt;
            # the old 30s floor caused every non-trivial kimi-k2.6 request to fail attempt 1 needlessly.
            # Only litellm.Timeout triggers a retry; other exceptions (auth, 4xx) propagate immediately.
            _backoff_timeouts = [60, 90, 120]
            last_exc: Exception | None = None
            for attempt, timeout in enumerate(_backoff_timeouts, start=1):
                if attempt > 1:
                    logger.warning(
                        "ollama-cloud %s timed out on attempt %d/3, retrying (timeout=%ds)...",
                        target, attempt - 1, timeout,
                    )
                kw = {**kwargs, "request_timeout": timeout}
                try:
                    if stream:
                        return await self.litellm_router.acompletion(
                            model=target, messages=messages_to_send, stream=True, **kw
                        )
                    else:
                        return await self.litellm_router.acompletion(
                            model=target, messages=messages_to_send, stream=False, **kw
                        )
                except litellm.Timeout as exc:
                    last_exc = exc
                    continue
                except Exception:
                    raise
            logger.error("ollama-cloud %s: all 3 attempts timed out", target)
            raise last_exc  # type: ignore[misc]
        else:
            try:
                if stream:
                    return await self.litellm_router.acompletion(
                        model=target,
                        messages=messages_to_send,
                        stream=True,
                        **kwargs,
                    )
                else:
                    return await self.litellm_router.acompletion(
                        model=target,
                        messages=messages_to_send,
                        stream=False,
                        **kwargs,
                    )
            except Exception as e:
                logger.error(f"All models failed: {e}")
                raise

    def _uses_anthropic(self, target: str) -> bool:
        """Return True if the target model alias is served via Anthropic API.

        Reads litellm_params.model from SoHoAI-config.yaml for the given alias.
        Used to gate Anthropic-specific cache_control injection so that
        swapping external to Gemini/OpenAI in SoHoAI-config.yaml requires no code change.
        """
        model_cfg = next(
            (m for m in self.config["model_list"] if m["model_name"] == target), {}
        )
        model_id = model_cfg.get("litellm_params", {}).get("model", "")
        return model_id.startswith("anthropic/")

    @staticmethod
    def _apply_cache_control(messages: list[dict]) -> list[dict]:
        """
        Inject Anthropic ephemeral cache_control markers for multi-turn prefix reuse.

        Breakpoints placed:
          1. System message (index 0 if role=='system') — longest-lived anchor.
          2. Rolling prefix anchor at the previous assistant turn (messages[-2]) —
             extends forward each turn; Anthropic matches the longest cached prefix.

        Returns a new list; original messages are not mutated. Only applied when
        content is a plain string — already-block content is left alone.

        Requires a minimum cacheable prefix of ~1024 tokens server-side; short
        conversations pay normal input cost until that threshold is reached.
        """
        if not messages:
            return messages

        new_messages = list(messages)

        # Breakpoint 1: system message
        first = new_messages[0]
        if first.get("role") == "system" and isinstance(first.get("content"), str):
            new_messages[0] = {
                **first,
                "content": [{
                    "type": "text",
                    "text": first["content"],
                    "cache_control": {"type": "ephemeral"},
                }],
            }

        # Breakpoint 2: rolling prefix at the previous complete turn (messages[-2]).
        # messages[-1] is the incoming user turn; messages[-2] is the last stable
        # message (typically the prior assistant response). Needs >=3 items so we
        # have at least [system?, prev_turn, current_user].
        if len(new_messages) >= 3:
            idx = len(new_messages) - 2
            anchor = new_messages[idx]
            if isinstance(anchor.get("content"), str):
                new_messages[idx] = {
                    **anchor,
                    "content": [{
                        "type": "text",
                        "text": anchor["content"],
                        "cache_control": {"type": "ephemeral"},
                    }],
                }

        return new_messages

    @property
    def available_models(self) -> list[str]:
        """List configured model aliases."""
        return [m["model_name"] for m in self.config["model_list"]]

    async def health_check(self) -> dict[str, bool]:
        """Quick health check for each model endpoint."""
        results = {}
        for model_cfg in self.config["model_list"]:
            name = model_cfg["model_name"]
            try:
                resp = await self.litellm_router.acompletion(
                    model=name,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                )
                results[name] = True
            except Exception:
                results[name] = False
        return results
