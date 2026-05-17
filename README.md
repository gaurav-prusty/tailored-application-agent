# Tailored Application Agent

A multi-agent AI system that turns a job posting URL into:
- A tailored resume (`.docx`, ATS-friendly)
- A cover letter written in your actual voice (mined from your sent mail / writing samples)
- A pre-filled JSON of likely application form fields

Built as a learning project to understand agentic AI architecture from first principles.

## Architecture

Six sub-agents orchestrated by LangGraph:

| Sub-agent | Model | Role |
|---|---|---|
| JD Analyzer | Gemini 2.5 Flash | Extracts requirements, keywords, seniority, culture cues |
| Resume Tailor | Gemini 2.5 Pro | Reorders/rewords bullets against JD — never invents |
| Voice Profiler | Gemini 2.5 Pro | Builds style fingerprint from writing samples |
| Letter Drafter | Gemini 2.5 Pro | Writes in that voice, not generic LLM voice |
| Field Mapper | Gemini 2.5 Flash | JSON of likely application form fields with answers |
| Fabrication Auditor | Gemini 2.5 Pro | Diffs output against claims ledger; halts on hallucination |

## Stack - Tentative
- **Google GenAI SDK** (Gemini 2.5 Pro + Flash, free tier via AI Studio)
- **LangGraph** (state graph, conditional edges, checkpointing)
- **SQLite** (claims ledger, voice profile cache)
- **python-docx** (ATS-compatible resume output)
- **Streamlit** (frontend)
- *Stretch:* Google Drive MCP, Gmail MCP

## Setup - Phase 0

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit .env with your GEMINI_API_KEY (https://aistudio.google.com/apikey)
python tests/phase0_smoke_test.py
```

## Status

| Phase | Name | Status |
|---|---|---|
| 0 | Environment & Mental Model | 🟢 Done |
| 1 | Claims Ledger & Resume Parsing | 🟢 Done |
| 2 | JD Analyzer | 🟢 Done |
| 3 | Resume Tailor + Fabrication Auditor | 🟡 In Progress |
| 4 | Voice Profiler | ⬜ Not started |
| 5 | Cover Letter Drafter | ⬜ Not started |
| 6 | LangGraph Orchestration | ⬜ Not started |
| 7 | Field Mapper + Output Packaging | ⬜ Not started |
| 8 | Streamlit Frontend | ⬜ Not started |
| 9 | Google Drive MCP (stretch) | ⬜ Not started |
| 10 | Gmail MCP (stretch) | ⬜ Not started |