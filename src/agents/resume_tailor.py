# Resume Tailor sub-agent.

# Input: claims ledger + JDAnalysis
# Output: TailoredResume (reworded, reordered, JD-aligned bullets)

from __future__ import annotations

import json
from typing import Iterable

from pydantic import BaseModel, Field, ValidationError

from src.agents.jd_analyzer import JDAnalysis
from src.ledger.db import Claim
from src.utils.llm import MODEL_FLASH, LLMResponse, call_llm


class TailoredBullet(BaseModel):
    """One reworded bullet, with back-pointers to the ledger claims it derives from."""

    text: str = Field(description="The polished, JD-aligned bullet text.")
    claim_ids: list[str] = Field(
        description=(
            "Ledger claim IDs (e.g. 'claim_042') this bullet derives from. "
            "Usually one; two when merging closely related claims."
        ),
    )


class TailoredRoleSection(BaseModel):
    """One job's worth of tailored bullets, grouped under role + company."""

    role_title: str = Field(description="Job title as it appears in the ledger, e.g. 'Senior Backend Engineer'.")
    company: str = Field(description="Employer name as it appears in the ledger.")
    bullets: list[TailoredBullet] = Field(
        description="Tailored bullets for this role, ordered most relevant to JD first.",
    )


class TailoredSkill(BaseModel):
    """A single skill that maps 1:1 to one ledger claim."""

    name: str = Field(description="The skill as it should appear in the resume's Skills section.")
    claim_id: str = Field(description="Ledger claim ID this skill comes from.")


class TailoredResume(BaseModel):
    """A JD-aligned resume, fully traceable to the claims ledger."""

    summary: str | None = Field(
        default=None,
        description="Optional 1-2 sentence headline. Use null if no ledger claim supports a clean summary.",
    )
    summary_claim_ids: list[str] = Field(
        default_factory=list,
        description="Claim IDs backing the summary. Empty list if summary is null.",
    )
    role_sections: list[TailoredRoleSection] = Field(
        description="Experience sections ordered most relevant to JD first.",
    )
    skills: list[TailoredSkill] = Field(
        default_factory=list,
        description="Skills section, ordered by JD relevance.",
    )
    education: list[TailoredBullet] = Field(
        default_factory=list,
        description="Education entries (degree, institution, year). Reuses TailoredBullet shape.",
    )


TAILOR_PROMPT = """\
You are a precise resume tailor. Your job is to assemble a job-aligned resume
by SELECTING, REORDERING, and REWORDING claims from a candidate's claims
ledger to match a target job description (JD). You must NEVER invent claims,
metrics, scope, or facts.

INPUTS (in the user message)
The user message contains two JSON blocks, labelled with headers:
  === CLAIMS LEDGER ===     a JSON array of claim objects, each with
                            id, category, text, source_role, source_company,
                            has_metric, raw_original.
  === JD ANALYSIS ===       a JSON object with role_title, company,
                            seniority_level, required_skills, preferred_skills,
                            keywords, culture_cues, red_flags.

OUTPUT
Return a single JSON object matching the schema you've been constrained to
(TailoredResume). Every assertion in the output (bullet, skill, summary,
education line) must carry the claim_id(s) it derives from. No exceptions.

THE THREE LEGAL OPERATIONS
You may, and only may, do these three things to a ledger claim:
  1. SELECT - include it in the output (or omit it).
  2. REORDER - present claims in JD-relevance order rather than ledger order.
  3. REWORD - change wording, grammar, and emphasis, AS LONG AS the meaning
              and every fact (numbers, scope, technologies) stay identical
              to the source claim.

THE ONE FORBIDDEN OPERATION
INVENT. You may not:
  - Add a metric not present in the source ("improved" → "improved by 30%").
  - Add scope not present ("led migration" → "led 6-person migration team").
  - Add technologies not present ("backend service" → "Go backend service").
  - Add timeframes not present ("shipped feature" → "shipped feature in Q3 2024").
  - Merge two claims into a single claim that asserts something neither
    claim said on its own.
  - Write a summary that asserts things no claim supports.
If a JD keyword has no matching claim, OMIT the keyword. Never invent a
claim to match it. The resume is honest before it is impressive.

SELECTION RULES
- Prioritize claims that match JD required_skills, then preferred_skills,
  then keywords, then everything else.
- A "match" means the claim mentions the skill / keyword / technology by
  name in `text` or `raw_original`. Adjacent skills don't match
  (claim says "Python" does NOT match JD keyword "Ruby").
- Omit claims with no relevance signal. A focused resume beats a
  comprehensive one for ATS scoring.
- For role_sections: include only roles that have at least one selected
  claim. Drop empty role sections.

REORDERING RULES
- Within each role_section, order bullets most JD-relevant first.
- Order role_sections by relevance to the JD, NOT strictly by recency.
  (Recency still matters; treat it as a tiebreaker when relevance is equal.)
- Skills: order JD required_skills first, then preferred, then the rest.

REWORDING RULES
- You MAY integrate JD keywords into reworded bullets when the underlying
  claim already supports them. Example:
    Claim: "Built REST API serving 10M requests/day."
    JD requires: "high-throughput backend services"
    OK to reword as: "Built high-throughput REST API serving 10M requests/day."
    (The phrase "high-throughput" describes 10M req/day, which the claim
    explicitly states.)
- You MAY NOT integrate a JD keyword that the claim doesn't support.
    Claim: "Built REST API."
    JD requires: "high-throughput"
    NOT OK: "Built high-throughput REST API." (claim doesn't establish scale)
- Keep bullets tight: <= 2 lines, action-led, past tense for prior roles.

CLAIM_IDS RULE (CRITICAL)
- Every TailoredBullet must list the claim_id(s) it derives from.
- Most bullets derive from ONE claim — emit a single-element list.
- A bullet may merge TWO closely related claims into one (e.g., "Designed
  payments service in Go" + "Cut transaction failures 40%" → "Designed Go
  payments service, cutting transaction failures 40%"). Emit BOTH claim_ids.
- NEVER emit a bullet with empty claim_ids. NEVER claim a claim_id you
  didn't actually use as a source.
- TailoredSkill: exactly one claim_id per skill (1:1 with ledger skills).
- TailoredResume.summary: if present, summary_claim_ids must list every
  claim the summary draws from. If null, summary_claim_ids must be [].

SKILLS SECTION
- Draw from ledger claims with category == "skill".
- Order JD required_skills first, then preferred, then the rest.
- Use the skill name as it appears in the claim's `text`. Do not rename
  ("Python" stays "Python"; do not change to "Python 3" unless the claim
  said "Python 3").

EDUCATION SECTION
- Include all ledger claims with category == "education".
- Each entry is one TailoredBullet with the institution / degree / year
  exactly as the claim states it. No rewording for education — just copy
  the claim's `text` (or a near-verbatim version) and cite claim_ids.

SUMMARY (OPTIONAL)
- Write a 1-2 sentence headline ONLY if you can back it with specific
  ledger claims. List those claims in summary_claim_ids.
- A summary cannot make a claim no ledger claim supports. "5+ years of
  fintech experience" is only valid if a claim establishes both the years
  and the fintech context.
- If no claim cleanly supports a summary, return summary = null and
  summary_claim_ids = []. Empty is the correct answer when there's no
  honest summary available.

WHAT TO IGNORE
- Claims with no relevance to this JD. Better to omit than stretch.
- JD red_flags. They're for the candidate's human review, not for tailoring.

EXAMPLE
Ledger excerpt:
  {"id": "claim_042", "category": "experience", "text": "Reduced API
   latency 60% by migrating REST to gRPC", "source_role": "Backend Engineer",
   "source_company": "Acme", "has_metric": true,
   "raw_original": "Reduced API latency 60% by migrating REST→gRPC (2024)"}
  {"id": "claim_073", "category": "skill", "text": "Go",
   "source_role": null, "source_company": null, "has_metric": false,
   "raw_original": "Languages: Go, Python, Rust"}

JD analysis excerpt:
  {"required_skills": ["Go", "distributed systems"],
   "keywords": ["low-latency", "platform engineering"]}

Correct partial output:
{
  "role_sections": [
    {
      "role_title": "Backend Engineer",
      "company": "Acme",
      "bullets": [
        {
          "text": "Reduced API latency 60% by migrating REST to gRPC",
          "claim_ids": ["claim_042"]
        }
      ]
    }
  ],
  "skills": [
    {"name": "Go", "claim_id": "claim_073"}
  ],
  "education": []
}

Note: "low-latency" appears in the JD keywords, but the claim says
"latency 60% reduction" — already a low-latency claim, so the keyword
naturally fits the rewording. "Platform engineering" has no supporting
claim, so we did NOT invent one to match it.

Return only the JSON object. No prose, no markdown.
"""

_TAILOR_MAX_TOKENS = 16384


def _format_user_message(jd_analysis: JDAnalysis, claims: Iterable[Claim]) -> str:
    """Serialize ledger + JD analysis as the labeled JSON blocks the prompt expects."""
    claims_payload = [
        {
            "id": c.id,
            "category": c.category,
            "text": c.text,
            "source_role": c.source_role,
            "source_company": c.source_company,
            "has_metric": c.has_metric,
            "raw_original": c.raw_original,
        }
        for c in claims
    ]
    return (
        "=== CLAIMS LEDGER ===\n"
        f"{json.dumps(claims_payload, indent=2)}\n\n"
        "=== JD ANALYSIS ===\n"
        f"{jd_analysis.model_dump_json(indent=2)}\n"
    )


def tailor_resume(
    jd_analysis: JDAnalysis,
    claims: Iterable[Claim],
) -> tuple[TailoredResume, LLMResponse]:
    """Generate a JD-aligned TailoredResume from the ledger. Returns (resume, llm_response)."""
    claim_list = list(claims)
    if not claim_list:
        raise ValueError("claims is empty — nothing to tailor from")

    user_msg = _format_user_message(jd_analysis, claim_list)

    response = call_llm(
        model=MODEL_FLASH,
        system=TAILOR_PROMPT,
        user=user_msg,
        max_tokens=_TAILOR_MAX_TOKENS,
        response_schema=TailoredResume,
        # thinking_budget=None (default) — leave Flash's reasoning ON for
        # the tailoring judgment calls (claim selection, reorder, reword).
    )

    if not response.text.strip():
        raise RuntimeError("LLM returned empty response")

    try:
        raw = json.loads(response.text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"LLM output was not valid JSON ({e}). "
            f"Output length: {len(response.text)} chars, "
            f"output tokens: {response.output_tokens}/{_TAILOR_MAX_TOKENS}. "
            "If output_tokens is at the cap, raise _TAILOR_MAX_TOKENS."
        ) from e

    try:
        return TailoredResume.model_validate(raw), response
    except ValidationError as e:
        raise RuntimeError(f"LLM output failed schema validation: {e}") from e
