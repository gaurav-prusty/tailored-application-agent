# JD Analy Operations

Practical runbook for the sub-agents that live in this package. Assumes the project venv is active and `GEMINI_API_KEY` is set in `.env`.

For *why* the sub-agent shape looks the way it does (single LLM call, strict input/output types, no side effects), see project root `CLAUDE.md`.

---

## Files in this package

| File | Role |
|---|---|
| `jd_analyzer.py` | Phase 2. Fetches a JD (URL or file) and produces a `JDAnalysis` Pydantic object: role/company, seniority, skills, keywords, culture cues, red flags. |

---

## JD Analyzer

### When to use which mode

| Source | Use this |
|---|---|
| Static-rendered JD page (Greenhouse, Lever, Ashby, most company careers pages) | URL mode |
| JS-rendered page (LinkedIn, Workday, Glassdoor, Indeed) | Save the JD text to a file, use file mode |
| Hand-edited JD text | File mode |

Quick test before running the analyzer: open the JD in your browser, View Source, Ctrl-F for a sentence from the JD body. If the sentence is in the source, URL mode will work. If it isn't, the page is JS-rendered, use file mode.

### Usage

```powershell
# URL mode
python -m src.agents.jd_analyzer https://hex.tech/careers/software-engineer-fullstack/

# File mode (save the JD as plain text first)
python -m src.agents.jd_analyzer data\jd_paste.txt
```

The CLI auto-routes: if the argument parses as `http(s)://...`, it fetches; otherwise it reads as a file path.

### Output

A pretty-printed JSON object matching `JDAnalysis`, plus a token-usage block. The JSON contains:

| Field | Type | What it's for |
|---|---|---|
| `role_title` | `str \| null` | Job title verbatim from the posting |
| `company` | `str \| null` | Hiring company name |
| `seniority_level` | enum | `intern \| junior \| mid \| senior \| lead \| principal` |
| `required_skills` | `list[str]` | Skills the JD says are non-negotiable |
| `preferred_skills` | `list[str]` | Bonus / nice-to-have skills |
| `keywords` | `list[str]` | ATS-searchable terms (industry, methodology, stage). NOT skills, NOT team names |
| `culture_cues` | `list[str]` | Concrete tradeoffs (remote/in-office, async, on-call). Empty if the JD has only value statements. |
| `red_flags` | `list[str]` | Real warning signs (unpaid trial, vague comp, "rockstar"). Empty for polished corporate JDs. |

### Programmatic use

```python
from src.agents.jd_analyzer import analyze_jd, fetch_jd, JDFetchError

# URL mode
try:
    jd_text = fetch_jd("https://...")
except JDFetchError as e:
    print(e)            # contains a paste-fallback hint
    raise

# File mode
jd_text = Path("data/jd_paste.txt").read_text(encoding="utf-8")

analysis, response = analyze_jd(jd_text)
print(analysis.seniority_level)     # e.g. "senior"
print(analysis.required_skills)     # list[str]
print(response.input_tokens)        # for cost accounting
```

`analyze_jd` always returns a validated `JDAnalysis` instance, invalid LLM output raises before this function returns.

---

## Common errors

| Symptom | Cause | Fix |
|---|---|---|
| `JDFetchError: HTTP error fetching <url>: ... 4xx/5xx` | Site is down, URL is wrong, or bot-walled even with browser headers | Verify URL in browser; if bot-walled, fall back to file mode |
| `JDFetchError: ... extracted only N chars of usable text` | Page is JS-rendered (LinkedIn, Workday, etc.) | Save the JD text to a file, re-run in file mode |
| Binary garbage in fetched text | Brotli decompression failed | Re-install: `pip install -r requirements.txt` (ensures `brotli` is present) |
| `LLM output was not valid JSON ... output_tokens at the cap` | Response truncated mid-string | Raise `_ANALYZER_MAX_TOKENS` in `jd_analyzer.py`; verify `thinking_budget=0` is still being passed |
| `LLM output failed schema validation: ...` | Model emitted shape the schema rejects (rare with `response_schema`) | Paste the error; either tighten the prompt or relax the schema |
| `RuntimeError: GEMINI_API_KEY not set` | `.env` missing or placeholder value | Copy `.env.example` to `.env`, fill in a real key |

---

## When to iterate `JD_ANALYZER_PROMPT`

The prompt has been tuned against polished corporate JDs (JPMC) and startup JDs (hex.tech). It will probably need adjustment if you regularly hit one of these patterns:

| Pattern | Likely issue | Lever to pull |
|---|---|---|
| Wrong required/preferred split | The JD doesn't use the standard "required" / "preferred" signal phrases | Add the JD's own signal phrases to the REQUIRED VS PREFERRED rule |
| Seniority consistently miscalled | The years-to-seniority map doesn't fit your target market (e.g., FAANG inflated titles, startup deflated titles) | Adjust the map in the SENIORITY rule |
| Team / business-unit names showing up in `keywords` | The Q1+Q2 test isn't catching them | Add the specific name pattern to the EXCLUDED list in KEYWORDS |
| DEI / values text in `culture_cues` | The values-vs-tradeoffs definition is being gamed | Add the offending phrasing to the value-marker blacklist in CULTURE CUES |

When you iterate, re-run on **both** a corporate JD and a startup JD before committing, the prompt has to work on both shapes, not just the one you tuned for.
