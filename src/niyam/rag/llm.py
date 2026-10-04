"""Chat completions through LiteLLM, so the provider is a config value.

Set NIYAM_LLM_MODEL (e.g. "groq/openai/gpt-oss-120b" or "gemini/gemini-2.5-flash") and
the provider's key (GROQ_API_KEY / GEMINI_API_KEY) in the environment or in .env.
NIYAM_LLM_FALLBACKS lists models to try, in order, when one fails (free tiers rate-limit);
models whose provider has no key are skipped.
"""

import logging
import os
from typing import Protocol

from dotenv import load_dotenv

from niyam.config import get_settings

log = logging.getLogger(__name__)

PROVIDER_KEYS = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def has_key(model: str) -> bool:
    """False when the model's provider needs a key that isn't set (unknown providers pass)."""
    env = PROVIDER_KEYS.get(model.split("/", 1)[0])
    return env is None or bool(os.environ.get(env))


class LLM(Protocol):
    def complete(self, messages: list[dict]) -> str: ...


class LiteLLM:
    def __init__(self, model: str | None = None, fallbacks: list[str] | None = None):
        load_dotenv(override=False)  # provider keys may live in .env
        s = get_settings()
        configured = [
            model or s.llm_model,
            *(fallbacks if fallbacks is not None else s.llm_fallbacks),
        ]
        self.models = [m for m in configured if has_key(m)] or configured
        self.temperature = s.llm_temperature
        self.timeout = s.llm_timeout_seconds

    def complete(self, messages: list[dict]) -> str:
        import litellm  # heavy import, only when actually calling a model

        last: Exception | None = None
        for model in self.models:
            try:
                resp = litellm.completion(
                    model=model,
                    messages=messages,
                    temperature=self.temperature,
                    timeout=self.timeout,
                    response_format={"type": "json_object"},
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # provider errors vary; try the next model
                log.warning("LLM %s failed: %s", model, exc)
                last = exc
        raise RuntimeError(f"all LLMs failed: {last}") from last
