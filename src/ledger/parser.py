"""Resume parser: turns a raw resume into a list of atomic claims and writes them to the ledger.

CLI:
    python -m src.ledger.parser <resume_path>          # parse + insert
    python -m src.ledger.parser <resume_path> --reset  # clear ledger first (added the flag because the parser is destructive and it's easy to accidentally re-run it on the same resume, which would create duplicate claims)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from src.ledger.db import (
    Claim,
    count_claims,
    delete_all_claims,
    get_claims,
    init_db,
    insert_claims,
)
from src.utils.llm import MODEL_FLASH, LLMResponse, call_llm


class ParsedClaim(BaseModel):
    """One atomic claim emitted by the LLM. IDs (claim_id) are assigned downstream in Python."""

    category: Literal["experience", "skill", "education", "metric"]
    text: str = Field(description="The polished claim, one fact only.")
    source_role: str | None = Field(
        default=None, description="Job title this claim came from, or null."
    )
    source_company: str | None = Field(
        default=None, description="Company this claim came from, or null."
    )
    has_metric: bool = Field(
        description="True iff `text` contains a number/quantity."
    )
    raw_original: str = Field(
        description="Verbatim substring of the input resume."
    )


PARSE_PROMPT = """\
You are a precise resume parser. Your only job is to extract atomic, traceable
claims from the resume text the user will send you, and return them as JSON.

OUTPUT FORMAT
Return a JSON array. Each element is an object with exactly these fields:

  {
    "category":       "experience" | "skill" | "education" | "metric",
    "text":           string,   // the polished claim, one fact only
    "source_role":    string | null,  // job title this claim came from
    "source_company": string | null,
    "has_metric":     boolean,  // true iff `text` contains a number/quantity
    "raw_original":   string    // VERBATIM substring from the input resume
  }

Do not include any other fields. Do not wrap the array in another object.
Do not emit ids — they will be assigned downstream.

ATOMICITY RULE
A claim is ONE fact. If a single bullet contains multiple facts (e.g.,
"Built X and improved Y by 30%"), split it into multiple claim objects, each
with its own `text` and its own `raw_original` (the same source line is
allowed to appear in `raw_original` for each split claim).

CATEGORIES
- "experience" — a duty, achievement, or contribution at a specific role
- "skill"      — a tool, language, framework, or technique listed in a Skills
                 section. One claim per distinct skill.
- "education"  — a degree, certification, or course
- "metric"     — use ONLY for standalone quantitative achievements that span
                 multiple roles or aren't tied to one job. Per-role metrics
                 stay in "experience".

ANTI-INVENTION RULES (CRITICAL)
- Never add facts that are not in the source text. No estimated metrics, no
  assumed team sizes, no inferred dates, no implied technologies.
- If the source says "improved performance," the claim says "improved
  performance" — not "improved performance by 30%."
- If a field cannot be determined from the source, use null (for source_role
  and source_company) or omit details from `text`. Do not guess.
- `raw_original` MUST be a verbatim substring of the input. Copy it exactly,
  including original punctuation. Do not paraphrase it.

WHAT TO IGNORE
- Section headers (EXPERIENCE, EDUCATION, SKILLS) — not claims.
- Contact info (name, email, phone, links) — not claims.
- Pure formatting (blank lines, dividers).

EXAMPLE
Input line:
  "Designed a payments service in Go, cutting transaction failures 40%."

Correct output (two atomic claims from one line):
[
  {
    "category": "experience",
    "text": "Designed a payments service in Go",
    "source_role": "Senior Engineer",
    "source_company": "Acme",
    "has_metric": false,
    "raw_original": "Designed a payments service in Go, cutting transaction failures 40%."
  },
  {
    "category": "experience",
    "text": "Cut transaction failures 40%",
    "source_role": "Senior Engineer",
    "source_company": "Acme",
    "has_metric": true,
    "raw_original": "Designed a payments service in Go, cutting transaction failures 40%."
  }
]

Return ONLY the JSON array. No prose, no markdown fences, no commentary.
"""


# Generous ceiling: a 5KB resume parses to ~3-4K output tokens; 8K leaves
# headroom for longer resumes without paying for unused budget.
_PARSE_MAX_TOKENS = 8192


def parse_resume(resume_text: str) -> tuple[int, LLMResponse]:
    """Parse resume text into atomic claims and bulk-insert them. Returns (count_inserted, llm_response)."""
    if not resume_text.strip():
        raise ValueError("resume_text is empty")

    response = call_llm(
        model=MODEL_FLASH,
        system=PARSE_PROMPT,
        user=resume_text,
        max_tokens=_PARSE_MAX_TOKENS,
        response_schema=list[ParsedClaim],
        thinking_budget=0,
    )

    if not response.text.strip():
        raise RuntimeError("LLM returned empty response")

    try:
        raw_items = json.loads(response.text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"LLM output was not valid JSON ({e}). "
            f"Output length: {len(response.text)} chars, "
            f"output tokens: {response.output_tokens}/{_PARSE_MAX_TOKENS}. "
            "If output_tokens is at the cap, the response was truncated — "
            "raise _PARSE_MAX_TOKENS or check thinking_budget."
        ) from e
    if not isinstance(raw_items, list):
        raise RuntimeError(f"Expected a JSON array, got {type(raw_items).__name__}")

    try:
        parsed = [ParsedClaim.model_validate(item) for item in raw_items]
    except ValidationError as e:
        raise RuntimeError(f"LLM output failed schema validation: {e}") from e

    claims = [
        Claim(
            id=f"claim_{i:03d}",
            category=p.category,
            text=p.text,
            source_role=p.source_role,
            source_company=p.source_company,
            has_metric=p.has_metric,
            raw_original=p.raw_original,
        )
        for i, p in enumerate(parsed, start=1)
    ]
    inserted = insert_claims(claims)
    return inserted, response


def _print_summary(claims_inserted: int, response: LLMResponse) -> None:
    """Print parse stats to stdout."""
    all_claims = get_claims()
    by_category = Counter(c.category for c in all_claims)
    metric_count = sum(1 for c in all_claims if c.has_metric)

    print(f"\n[OK] Parsed and inserted {claims_inserted} claims.")
    print("\n--- Breakdown ---")
    for category in ("experience", "skill", "education", "metric"):
        print(f"  {category:<11} {by_category.get(category, 0)}")
    print(f"  with metrics: {metric_count}")
    print("\n--- Token usage ---")
    print(f"  input:  {response.input_tokens}")
    print(f"  output: {response.output_tokens}")
    print("  cost:   $0.00 (Gemini free tier)")


def _cli() -> None:
    """`python -m src.ledger.parser <resume_path> [--reset]`."""
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help"}:
        print("usage: python -m src.ledger.parser <resume_path> [--reset]", file=sys.stderr)
        sys.exit(2)

    resume_path = Path(args[0])
    reset = "--reset" in args[1:]

    if not resume_path.is_file():
        print(f"error: file not found: {resume_path}", file=sys.stderr)
        sys.exit(1)

    init_db()

    existing = count_claims()
    if existing > 0 and not reset:
        print(
            f"error: ledger already contains {existing} claims. "
            "Re-run with --reset to clear and re-parse.",
            file=sys.stderr,
        )
        sys.exit(1)
    if reset and existing > 0:
        deleted = delete_all_claims()
        print(f"Cleared {deleted} existing claims.")

    resume_text = resume_path.read_text(encoding="utf-8")
    inserted, response = parse_resume(resume_text)
    _print_summary(inserted, response)


if __name__ == "__main__":
    _cli()
