"""
Smart model routing — decides which LLM handles each request.

Routing strategy:
  1. If user explicitly requests a model → use it
  2. If force_cloud flag → use cloud (for complex/critical tasks)
  3. If message context is very long → use cloud (better context handling)
  4. Default → try specialist (GPU), fall back through chain

LiteLLM handles the actual fallback retries; this module decides
the *starting point* and any pre-routing logic.
"""

from __future__ import annotations

import logging
from typing import Optional

import yaml
from litellm import Router

logger = logging.getLogger(__name__)


class SmartRouter:
    """Wraps LiteLLM Router with HomeAI-Lab-specific routing logic."""

    def __init__(self, config_path: str = "config.yaml"):
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

        self.litellm_router = Router(
            model_list=self.config["model_list"],
            routing_strategy="simple-shuffle",
            # Fallback: if specialist fails, go straight to cloud
            fallbacks=[
                {
                    "specialist": ["external"],
                }
            ],
            # Retry config
            num_retries=2,
            timeout=60,
        )

        self.routing_config = self.config.get("routing", {})
        self.default_model = self.routing_config.get("default_model", "specialist")
        self.cloud_model = self.routing_config.get("cloud_model", "external")
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

        try:
            if stream:
                # Return async generator for SSE streaming
                response = await self.litellm_router.acompletion(
                    model=target,
                    messages=messages,
                    stream=True,
                    **kwargs,
                )
                return response
            else:
                response = await self.litellm_router.acompletion(
                    model=target,
                    messages=messages,
                    stream=False,
                    **kwargs,
                )
                return response
        except Exception as e:
            logger.error(f"All models failed: {e}")
            raise

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
