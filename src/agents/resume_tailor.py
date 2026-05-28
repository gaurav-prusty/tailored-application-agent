# Resume Tailor sub-agent.

# Input: claims ledger + JDAnalysis
# Output: TailoredResume (reworded, reordered, JD-aligned bullets)

from __future__ import annotations

from pydantic import BaseModel, Field


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
