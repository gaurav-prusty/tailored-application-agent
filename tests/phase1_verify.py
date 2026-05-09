"""Phase 1 verification helper. Read-only audit tool for the claims ledger.

CLI:
    python tests/phase1_verify.py overview
    python tests/phase1_verify.py role "<role title>"
    python tests/phase1_verify.py metrics
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ledger.db import Claim, count_claims, get_claims


def _print_claim(c: Claim) -> None:
    """Render a claim with its source line, indented for readability."""
    metric_flag = " [METRIC]" if c.has_metric else ""
    print(f"\n  [{c.id}] {c.category}{metric_flag}")
    print(f"    role:    {c.source_role or '(none)'} @ {c.source_company or '(none)'}")
    print(f"    text:    {c.text}")
    print(f"    source:  {c.raw_original}")


def overview() -> None:
    """Print high-level counts: total claims, breakdown by category, list of roles."""
    total = count_claims()
    if total == 0:
        print("Ledger is empty. Run: python -m src.ledger.parser data\\master_resume.txt")
        return

    all_claims = get_claims()
    by_category = Counter(c.category for c in all_claims)
    metric_count = sum(1 for c in all_claims if c.has_metric)

    print(f"Total claims: {total}")
    print("\nBy category:")
    for cat in ("experience", "skill", "education", "metric"):
        print(f"  {cat:<11} {by_category.get(cat, 0)}")
    print(f"  with metrics: {metric_count}")

    roles = Counter(
        (c.source_role, c.source_company)
        for c in all_claims
        if c.source_role is not None
    )
    print("\nRoles found (claim count):")
    for (role, company), n in roles.most_common():
        print(f"  {n:>3}  {role} @ {company}")


def show_role(role_title: str) -> None:
    """Dump every claim where source_role matches `role_title` (case-insensitive)."""
    matches = [
        c for c in get_claims()
        if c.source_role and c.source_role.lower() == role_title.lower()
    ]
    if not matches:
        print(f"No claims found with source_role matching '{role_title}'.")
        print("Try `overview` to see available roles.")
        return

    print(f"{len(matches)} claims for role '{role_title}':")
    for c in matches:
        _print_claim(c)


def show_metrics() -> None:
    """Dump every claim with has_metric=True. These are the highest-risk audit targets."""
    matches = [c for c in get_claims() if c.has_metric]
    if not matches:
        print("No metric-bearing claims in the ledger.")
        return

    print(f"{len(matches)} metric-bearing claims (audit these first):")
    for c in matches:
        _print_claim(c)


def _cli() -> None:
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help"}:
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    cmd = args[0]
    if cmd == "overview":
        overview()
    elif cmd == "role":
        if len(args) < 2:
            print("error: `role` requires a role title argument", file=sys.stderr)
            sys.exit(2)
        show_role(args[1])
    elif cmd == "metrics":
        show_metrics()
    else:
        print(f"error: unknown command '{cmd}'", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    _cli()
