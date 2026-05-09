# JD Analyzer sub-agent.


from __future__ import annotations

from typing import Literal

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field


class JDAnalysis(BaseModel):
    """Structured analysis of a job description. Consumed by Phase 3 (Resume Tailor) and beyond."""

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
