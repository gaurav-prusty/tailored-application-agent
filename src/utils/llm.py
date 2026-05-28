"""
Thin wrapper around the Google GenAI SDK (Gemini).

MOST Important file.
We can invoke models directly from every sub-agent. We don't, for these reasons:

  1. ONE place to configure model IDs. SSoT - swapping models later will require minimal change.
  2. ONE place for retry/error handling. Rate limits and transient 5xx (mostly 504 - Gateway Timeout) errors get handled here so sub-agents stay focused on their own logic.
  3. ONE place for cost/quota logging.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# Load API key from .env file.
load_dotenv()

# Every sub-agent runs on Flash. Pro is paid-only on Gemini's free tier as of
# 2026-05. Pass thinking_budget=0 for extraction/classification; leave it at
# default (None) for writing/judgment tasks that need reasoning.
MODEL_FLASH = "gemini-2.5-flash"


@dataclass
class LLMResponse:
    """What our wrapper returns. Keeps sub-agents decoupled from the SDK's
    response shape — if Google changes their response object, only this
    file breaks."""
    text: str
    input_tokens: int
    output_tokens: int
    model: str


def _client() -> genai.Client:
    """Lazy client construction. We read GEMINI_API_KEY from env explicitly
    (the SDK also auto-detects GOOGLE_API_KEY, but we standardize on
    GEMINI_API_KEY in .env for clarity)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key.startswith("your-gemini-key"):
        raise RuntimeError(
            "GEMINI_API_KEY not set. Set it in your .env file (https://aistudio.google.com/apikey)."
        )
    return genai.Client(api_key=api_key)


def call_llm(
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 1024,
    max_retries: int = 3,
    response_schema: type | None = None,
    thinking_budget: int | None = None,
) -> LLMResponse:
    """Single-turn LLM call with retries on transient errors.

    Pass `response_schema` (a Pydantic model class) to force JSON output
    constrained to that schema. The returned `text` will be valid JSON.

    Pass `thinking_budget=0` to disable Gemini 2.5's internal reasoning,
    which otherwise consumes tokens from `max_tokens` before any visible
    output is emitted.
    """
    client = _client()
    last_err: Exception | None = None

    config_kwargs: dict[str, object] = {
        "system_instruction": system,
        "max_output_tokens": max_tokens,
    }
    if response_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget
        )

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            # `resp.text` pulls the text out of
            # resp.candidates[0].content.parts[0].text for us.
            text = resp.text or ""
            usage = resp.usage_metadata
            return LLMResponse(
                text=text,
                input_tokens=usage.prompt_token_count or 0,
                output_tokens=usage.candidates_token_count or 0,
                model=model,
            )
        except genai_errors.APIError as e:
            last_err = e
            # 429 = rate limit / quota; 5xx = server's fault. Both transient — retry.
            # Other 4xx = our fault (bad request, bad key) — fail fast.
            code = getattr(e, "code", None)
            if code == 429 or (code is not None and code >= 500):
                # Exponential backoff: 1s, 2s, 4s. Don't hammer a service
                # that's already telling you to slow down.
                time.sleep(2**attempt)
            else:
                raise

    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")
