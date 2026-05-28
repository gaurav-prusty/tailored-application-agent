# Fabrication Auditor sub-agent.
#
# Receives a TailoredResume and the claims ledger. Verifies every bullet
# traces back to its claim_ids without drift (no invented metrics, scope,
# or facts). Hard gate: if any bullet is flagged, the pipeline halts.

from __future__ import annotations
