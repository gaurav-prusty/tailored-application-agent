"""
Thin wrapper around the Google GenAI SDK (Gemini).

WHY this file exists:
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

# Cost-tier reference:
#   - FLASH -> classification, extraction, structured JSON. Cheap and fast.
#              Generous free-tier quota (~hundreds of requests/day).
#   - PRO   -> writing-quality work: resume tailoring, cover letters, auditing.
#              Tighter free-tier quota — use deliberately, not by default.
MODEL_FLASH = "gemini-2.5-flash"
MODEL_PRO = "gemini-2.5-pro"


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
) -> LLMResponse:
    """
    Single-turn LLM call with basic retry on rate limits / transient errors.

    Why retries: Google occasionally returns 429 (rate limit / quota exhausted)
    or 5xx (server side). These are transient. Without retry, your pipeline
    dies on a hiccup. With exponential backoff, it heals itself. This is
    defensive coding 101 for any code that talks to a network service.

    Why keyword-only args (the `*,`): with this many parameters, positional
    args become unreadable. Forcing keywords makes call sites self-documenting.

    Note on the API shape: in the Google GenAI SDK, the system prompt is
    NOT a separate "role" like in Anthropic/OpenAI — it goes into
    `config.system_instruction`. The user message is passed as `contents`.
    """
    client = _client()
    last_err: Exception | None = None

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                ),
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
