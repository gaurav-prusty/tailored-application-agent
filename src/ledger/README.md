# Claims Ledger — Operations

Practical runbook for the claims ledger: how to build it, parse a resume into it, inspect it, and reset it. Assumes the project venv is active and `GEMINI_API_KEY` is set in `.env`.

For *why* the ledger exists, see project root `README.md`.

---

## Files in this package

| File | Role |
|---|---|
| `schema.sql` | SQL DDL for the `claim` table + indexes |
| `db.py` | The only module that writes SQL. Exposes `init_db`, `insert_claim(s)`, `get_claims`, `count_claims`, `delete_all_claims` |
| `parser.py` | Resume → atomic claims → ledger. Owns the `PARSE_PROMPT` and `ParsedClaim` schema |

---

## Sequence: from empty repo to a populated ledger

```powershell
# 1. Initialize the database file and schema. Idempotent.
python -m src.ledger.db init
# -> creates db/ledger.sqlite with the `claim` table + indexes
# -> safe to run repeatedly; no-op if already initialized

# 2. Parse the master resume into atomic claims.
python -m src.ledger.parser data\master_resume.txt
# -> calls Gemini Flash with the schema-constrained prompt
# -> validates each claim via Pydantic
# -> assigns IDs (claim_001, claim_002, ...) in Python
# -> bulk-inserts in a single transaction
# -> prints a summary: counts by category, metric count, token usage
# -> refuses to run if the table already has rows (see --reset below)
```

---

## Inspect

```powershell
# Quick row count.
python -m src.ledger.db count
# -> "47 claims in C:\...\db\ledger.sqlite"

# Categories, metric count, roles found. Use this first.
python tests\phase1_verify.py overview

# Dump every claim for a specific role (case-insensitive match on source_role).
python tests\phase1_verify.py role "Backend Engineer"

# Dump every metric-bearing claim. Audit these first — numbers are the highest
# fabrication-risk surface.
python tests\phase1_verify.py metrics

# Raw SQL when you want it.
sqlite-utils db\ledger.sqlite "SELECT id, category, source_role, text FROM claim LIMIT 10"
sqlite-utils tables db\ledger.sqlite --schema
```

---

## Reset / re-parse

The parser refuses to run on a non-empty table to prevent accidental duplicates. To re-parse after a resume update or a prompt tweak:

```powershell
# Clears every row first, then parses fresh. Reuses claim_NNN IDs.
python -m src.ledger.parser data\master_resume.txt --reset
```

To wipe the file entirely (rare; mostly for tests):

```powershell
Remove-Item db\ledger.sqlite
python -m src.ledger.db init
```

---

## Common errors

| Symptom | Cause | Fix |
|---|---|---|
| `error: ledger already contains N claims` | Re-running parser without `--reset` | Add `--reset` if intentional |
| `LLM output was not valid JSON ... output_tokens at the cap` | Response truncated mid-string | Raise `_PARSE_MAX_TOKENS` in `parser.py`, or verify `thinking_budget=0` is being passed |
| `LLM output failed schema validation: ...` | Model emitted something the schema rejects | Paste the error; either tighten the prompt or relax the schema |
| `IntegrityError: CHECK constraint failed: claim` | Category outside the four allowed | Schema constraint should prevent this; treat as a parser bug |
| `RuntimeError: GEMINI_API_KEY not set` | `.env` missing or placeholder value | Copy `.env.example` to `.env`, fill in a real key |
