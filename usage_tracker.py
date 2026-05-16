"""
Usage tracking for LiteLLM — logs completion events to SQLite for cost analysis.

Integrates with litellm.integrations.custom_logger.CustomLogger to capture:
  - Token counts (input, output, cache_creation, cache_read)
  - Cost via litellm.completion_cost()
  - Metadata context (source, user_id, chat_id, orchestra_session_id)
  - Request ID and timestamps

All events are persisted to ChatStore.usage_events for analytics and billing.
"""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import TYPE_CHECKING

from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from chat_store import ChatStore

logger = logging.getLogger(__name__)


class UsageTracker(CustomLogger):
    """
    LiteLLM custom logger that persists usage events to SQLite.

    Receives completion events from litellm.Router and records:
      - Token counts (prompt, completion, cache tokens)
      - Cost via litellm.completion_cost()
      - Request metadata (user_id, chat_id, etc.)
      - Provider and model information
    """

    def __init__(self, store: ChatStore):
        """
        Initialize UsageTracker with a ChatStore reference.

        Args:
            store: ChatStore instance for recording usage events.
        """
        super().__init__()
        self._store = store

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Log a successful completion event.

        Called by litellm.Router after a successful acompletion() call.

        Args:
            kwargs: Original completion call kwargs (contains model, metadata, litellm_call_id, etc.)
            response_obj: litellm.ModelResponse object with usage and provider info.
            start_time: Start timestamp (seconds).
            end_time: End timestamp (seconds).
        """
        try:
            # Extract model name (use from kwargs for the original requested model)
            model = kwargs.get("model", "unknown")

            # Extract usage from response_obj.usage
            input_tokens = getattr(response_obj.usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(response_obj.usage, "completion_tokens", 0) or 0

            # Extract Anthropic cache tokens if available
            cache_creation_tokens = getattr(response_obj.usage, "cache_creation_input_tokens", 0) or 0
            cache_read_tokens = getattr(response_obj.usage, "cache_read_input_tokens", 0) or 0

            # Extract metadata context
            metadata = kwargs.get("metadata") or {}
            source = metadata.get("source", "unknown")
            user_id = metadata.get("user_id")
            chat_id = metadata.get("chat_id")
            orchestra_session_id = metadata.get("orchestra_session_id")

            # Extract request ID (litellm_call_id or generate new one)
            request_id = kwargs.get("litellm_call_id") or str(uuid.uuid4())

            # Derive provider from response or model name
            # Try custom_llm_provider from hidden_params first, then kwargs, then parse model string
            provider = None
            if hasattr(response_obj, "_hidden_params") and response_obj._hidden_params:
                provider = response_obj._hidden_params.get("custom_llm_provider")
            if not provider:
                provider = kwargs.get("custom_llm_provider")
            if not provider:
                # Fallback: parse model string prefix (e.g., "anthropic/claude-sonnet-4-6" → "anthropic")
                if "/" in model:
                    provider = model.split("/")[0]
                elif any(alias in model for alias in ["local", "gemma"]):
                    provider = "local"
                else:
                    provider = "unknown"

            # Calculate cost
            try:
                import litellm
                cost = litellm.completion_cost(completion_response=response_obj)
            except Exception as e:
                logger.warning(f"Failed to calculate cost for {model}: {e}")
                cost = 0.0

            # Emit warning if cost is 0.0 for non-local models (pricing gap)
            if cost == 0.0 and provider not in ("local", "unknown"):
                logger.warning(
                    f"completion_cost returned 0 for model={model} (provider={provider}) — pricing gap"
                )

            # Create ISO 8601 timestamp with Z suffix for UTC
            created_at = datetime.datetime.utcnow().isoformat() + "Z"

            # Record the usage event
            self._store.record_usage_event(
                request_id=request_id,
                created_at=created_at,
                source=source,
                user_id=user_id,
                chat_id=chat_id,
                orchestra_session_id=orchestra_session_id,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_tokens=cache_creation_tokens,
                cache_read_tokens=cache_read_tokens,
                cost_usd=cost,
                provider=provider,
            )

        except Exception as e:
            # Log error but do not raise — we don't want usage tracking to break the request
            logger.error(f"Error tracking usage event: {e}", exc_info=True)
