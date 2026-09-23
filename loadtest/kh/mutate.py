"""Mutation-test the v3 scope-arm module.

A new test that has never failed is weak evidence. Each mutation below breaks
one specific rule in the provider; the suite must go red for it, and the script
reports any mutation that survives -- a survivor means the corresponding
assertion is not actually load-bearing.

The provider is restored from a saved copy after every run, including on error.

Run:  python loadtest/kh/mutate.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROVIDER = REPO / "backend/python/app/services/graph_db/neo4j/neo4j_provider.py"
BACKUP = Path(__file__).resolve().parent / "results" / "neo4j_provider.backup.py"
TARGET = [
    "tests/integration/graph_permissions/test_provider_v3_scope_arms.py",
    "tests/integration/graph_permissions/test_provider_v3_hop_rule.py",
]

# (name, old, new, what it should break)
MUTATIONS = [
    (
        "arm2_ignores_hideChildren",
        "          AND NOT coalesce(dg.hideChildren, false)\n",
        "",
        "a hideChildren group would enter the declared scope, leaking v3-hidden-r1",
    ),
    (
        "arm2_admits_any_declaration",
        "          AND (dg.id IN regionA OR dg.id IN $grantedIds)",
        "          AND (dg.id IS NOT NULL)",
        "a declaration granted to nobody would open its contents",
    ),
    (
        "arm4_ignores_declared_scope",
        "        WHERE mg.id IN declaredScope AND NOT coalesce(mb.isDeleted, false)",
        "        WHERE mg.id IS NOT NULL AND NOT coalesce(mb.isDeleted, false)",
        "every group's BELONGS_TO contents would be admitted",
    ),
    (
        "arm5_drops_the_grant_test",
        "          AND sd.id IN $grantedIds\n",
        "",
        "any OPEN node in the connector would become a seed",
    ),
    (
        "hop_naive_compaction_fails_open",
        """                    AND (coalesce({child}.accessRule, 'OPEN') IN ['STRICT', 'OPEN']
                      OR (coalesce({child}.accessRule, 'OPEN') = 'RESTRICTED'
                          AND {child}.id IN $grantedIds)))""",
        """                    AND (coalesce({child}.accessRule, 'OPEN') <> 'RESTRICTED'
                      OR {child}.id IN $grantedIds))""",
        "decision 77: the tempting compaction returns a node whose accessRule "
        "is corrupt, instead of failing closed",
    ),
    (
        "restricted_accepts_inheritance_alone",
        """                      OR (coalesce({child}.accessRule, 'OPEN') = 'RESTRICTED'
                          AND {child}.id IN $grantedIds)))""",
        """                      OR (coalesce({child}.accessRule, 'OPEN') = 'RESTRICTED')))""",
        "AC-16: RESTRICTED would admit on inheritance without a grant",
    ),
    (
        "hop_ignores_inheritance_and_grant",
        "AND (appOpensEverything\n                OR (EXISTS {",
        "AND (true\n                OR (EXISTS {",
        "the App walk would admit every descendant",
    ),
]


def run_suite() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *TARGET, "-q", "--no-header",
         "-p", "no:warnings", "--tb=no"],
        cwd=REPO / "backend/python", capture_output=True, text=True, timeout=1800,
    )
    return proc.returncode == 0, (proc.stdout or "")[-400:]


def main() -> None:
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROVIDER, BACKUP)
    original = PROVIDER.read_text(encoding="utf-8")
    survivors: list[str] = []
    try:
        print("baseline (unmutated) ...")
        ok, tail = run_suite()
        if not ok:
            print("BASELINE IS RED -- fix that before trusting any mutation")
            print(tail)
            return
        print("  green\n")

        for name, old, new, expected in MUTATIONS:
            if old not in original:
                print(f"SKIP {name}: pattern not found (provider changed?)")
                survivors.append(f"{name} (pattern not found)")
                continue
            # Replace EVERY occurrence, not the first. Several of these strings
            # appear both in the live page query and in the standalone
            # `get_knowledge_hub_visible_set_v3`, which nothing calls -- and the
            # standalone copy comes first in the file. Mutating only that one
            # changes no behaviour the tests can see and reports a false
            # "SURVIVED".
            occurrences = original.count(old)
            PROVIDER.write_text(original.replace(old, new), encoding="utf-8")
            ok, tail = run_suite()
            verdict = "SURVIVED" if ok else "caught"
            print(f"{verdict:>9}  {name}  ({occurrences} site(s) mutated)")
            print(f"           expected to break: {expected}")
            if ok:
                survivors.append(name)
            PROVIDER.write_text(original, encoding="utf-8")
    finally:
        PROVIDER.write_text(original, encoding="utf-8")
        assert PROVIDER.read_text(encoding="utf-8") == original

    print(f"\n{len(MUTATIONS) - len(survivors)}/{len(MUTATIONS)} caught")
    for s in survivors:
        print("  SURVIVED:", s)
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
