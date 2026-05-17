# JD Analyzer sub-agent.

# Smoke Tests

# 1. JD fetcher 
# python -c "from src.agents.jd_analyzer import fetch_jd; print(fetch_jd('https://hex.tech/careers/software-engineer-fullstack/')[:1500])"
# 2. Prompt wiring
#  python -c "from src.agents.jd_analyzer import analyze_jd; a, r = analyze_jd('Senior Python Engineer at Acme. Required: 5+ years Python, AWS. Kubernetes a plus. Remote-friendly.'); print(a.model_dump_json(indent=2)); print(f'\ntokens: {r.input_tokens} in, {r.output_tokens} out')"


from __future__ import annotations

import json
from typing import Literal

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, ValidationError

from src.utils.llm import MODEL_FLASH, LLMResponse, call_llm


class JDAnalysis(BaseModel):
    """Structured analysis of a job description"""

    role_title: str | None = Field(
        default=None,
        description="The job title as stated in the posting, e.g. 'Senior Backend Engineer'.",
    )
    company: str | None = Field(
        default=None,
        description="The hiring company name, if stated in the posting.",
    )
    seniority_level: Literal["intern", "junior", "mid", "senior", "lead", "principal"] = Field(
        description="The seniority level inferred from the posting."
    )
    required_skills: list[str] = Field(
        default_factory=list,
        description="Skills the JD presents as non-negotiable (must-have).",
    )
    preferred_skills: list[str] = Field(
        default_factory=list,
        description="Skills the JD presents as bonus/nice-to-have.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Flat list of high-priority terms an ATS would search for.",
    )
    culture_cues: list[str] = Field(
        default_factory=list,
        description="Phrases hinting at company culture: 'remote-first', 'fast-paced', 'ownership culture', etc.",
    )
    red_flags: list[str] = Field(
        default_factory=list,
        description="Concerning phrases: 'unpaid trial', 'rockstar', vague comp, 'wear many hats', etc.",
    )

# Real-browser header signature. UA alone isn't enough — Cloudflare/etc.
# also check Accept, Accept-Language, Accept-Encoding to catch scripts that
# only spoof the User-Agent.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}
_TIMEOUT = httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0)
_MIN_USEFUL_LENGTH = 200


class JDFetchError(RuntimeError):
    """Raised when a job posting URL can't be fetched or yields no usable text."""


def fetch_jd(url: str) -> str:
    """Fetch a job posting URL and return cleaned plain text. Raises JDFetchError on failure."""
    try:
        with httpx.Client(
            headers=_BROWSER_HEADERS,
            timeout=_TIMEOUT,
            follow_redirects=True,
        ) as client:
            resp = client.get(url)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise JDFetchError(f"HTTP error fetching {url}: {e}") from e

    text = _html_to_text(resp.text)
    if len(text) < _MIN_USEFUL_LENGTH:
        raise JDFetchError(
            f"Fetched {url} but extracted only {len(text)} chars of usable text. "
            "Page is likely JS-rendered or behind a bot wall. "
            "Save the JD as a .txt file and pass that path instead."
        )
    return text


def _html_to_text(html: str) -> str:
    """Strip non-content tags, isolate main content, and return plain text."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(
        ["script", "style", "nav", "footer", "header", "aside",
         "noscript", "form", "button", "svg", "iframe"]
    ):
        tag.decompose()

    # Selector cascade: prefer semantic main-content tags; fall back to body.
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.body
        or soup
    )

    return _normalize_whitespace(main.get_text(separator="\n"))


def _normalize_whitespace(text: str) -> str:
    """Trim each line and collapse runs of blank lines into single blanks."""
    out: list[str] = []
    prev_blank = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            if not prev_blank:
                out.append("")
            prev_blank = True
        else:
            out.append(line)
            prev_blank = False
    return "\n".join(out).strip()


JD_ANALYZER_PROMPT = """\
Analyze the job description (JD) the user sends and return a single JSON object
matching the schema you've been constrained to. Your job is to fill each field
correctly — the schema guarantees the shape, this prompt governs the content.

GENERAL RULE
Extract only from what the JD actually says. Do not infer skills, technologies,
or culture traits that aren't in the text. If a field cannot be determined from
the JD, use null (for role_title / company) or an empty list (for the others).
Do not guess.

REQUIRED VS PREFERRED
A skill is "required" only when the JD signals it as non-negotiable:
  - explicit words: "required", "must have", "minimum", "you have", "we need"
  - listed in a section titled "Requirements", "Must-haves", "Qualifications"
A skill is "preferred" when the JD signals it as a bonus:
  - explicit words: "preferred", "nice to have", "a plus", "bonus", "ideally"
  - listed in "Nice to have", "Bonus", "Preferred Qualifications"
If a skill is mentioned but its bucket is unclear, default to "required" if
it's in the first half of the requirements section, "preferred" otherwise.

SENIORITY
Use this priority order (use the first signal that's clearly present):
  1. Explicit years of experience ("5+ years" → senior; "8+" → lead/principal)
  2. Scope-of-impact language (manages teams, sets technical direction → senior/lead)
  3. Job title (only if 1 and 2 are silent)
Map years roughly: 0 → intern; 0–2 → junior; 2–5 → mid; 5–8 → senior;
8–12 → lead; 12+ → principal. Adjust if scope language overrides.

KEYWORDS
Distinct from skills. A keyword is any high-signal term an ATS would search
for that isn't already in required_skills or preferred_skills:
  - company stage / business model: "B2B SaaS", "Series B", "fintech"
  - domain terms: "data warehousing", "regulated industry", "FedRAMP"
  - methodology terms: "agile", "TDD", "platform engineering"
Avoid duplicating items from required_skills / preferred_skills.
Cap at ~10 — only the highest-signal terms.

CULTURE CUES
Concrete, specific phrases — not generic adjectives. Include things like:
  "remote-first", "async-first", "fast-paced startup", "ownership culture",
  "in-office 5 days", "individual contributor track"
Exclude empty filler: "great team", "exciting opportunity", "collaborative".

RED FLAGS
Real warning signs, not boilerplate. Include things like:
  - unpaid work: "unpaid trial", "equity-only compensation"
  - vague comp: "competitive salary" with no range
  - scope creep: "wear many hats", "do whatever it takes"
  - culture markers: "rockstar", "ninja", "we work hard"
  - red-tape signals: extreme tenure asks at junior levels
Do not flag every "fast-paced" — only when paired with other concerning signals.

ROLE_TITLE AND COMPANY
Extract role_title verbatim from the posting if stated. Use null if not stated.
Extract company name if stated explicitly in the JD body or header. Use null
if you'd be guessing from a URL or domain.

EXAMPLE
Input snippet:
  "Senior Software Engineer at Hex. SF or NYC, hybrid. Required: 5+ years
  building production systems. Strong Python and SQL. Familiarity with
  Kubernetes is a plus. We're a fast-paced startup that ships daily and
  rewards ownership."

Correct partial output:
{
  "role_title": "Senior Software Engineer",
  "company": "Hex",
  "seniority_level": "senior",
  "required_skills": ["Python", "SQL", "5+ years building production systems"],
  "preferred_skills": ["Kubernetes"],
  "keywords": ["fast-paced startup", "ships daily"],
  "culture_cues": ["ownership culture", "hybrid (SF/NYC)"],
  "red_flags": []
}

Return only the JSON object. No prose, no markdown.
"""


_ANALYZER_MAX_TOKENS = 4096


def analyze_jd(jd_text: str) -> tuple[JDAnalysis, LLMResponse]:
    """Run the analyzer on JD text. Returns (analysis, llm_response)."""
    if not jd_text.strip():
        raise ValueError("jd_text is empty")

    response = call_llm(
        model=MODEL_FLASH,
        system=JD_ANALYZER_PROMPT,
        user=jd_text,
        max_tokens=_ANALYZER_MAX_TOKENS,
        response_schema=JDAnalysis,
        thinking_budget=0,
    )

    if not response.text.strip():
        raise RuntimeError("LLM returned empty response")

    try:
        raw = json.loads(response.text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"LLM output was not valid JSON ({e}). "
            f"Output length: {len(response.text)} chars, "
            f"output tokens: {response.output_tokens}/{_ANALYZER_MAX_TOKENS}. "
            "If output_tokens is at the cap, raise _ANALYZER_MAX_TOKENS."
        ) from e

    try:
        return JDAnalysis.model_validate(raw), response
    except ValidationError as e:
        raise RuntimeError(f"LLM output failed schema validation: {e}") from e
