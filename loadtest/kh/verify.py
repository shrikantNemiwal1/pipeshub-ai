"""Prove the landed changes are semantics-preserving, across the matrix.

The three changes are text-level, so they can be *reverted* at text level: this
captures whatever the provider generates now and rebuilds the pre-change query
from it. Both arms therefore come from the same live provider output, which
means this keeps working as the surrounding query evolves -- unlike a checked-in
copy of the old Cypher, which is exactly how `docs/kh-v3/*.cypher` drifted.

Compares ordered page ids, `total` and per-type counts for every
(user x connector x sort x page) cell. Any difference is a failure.

Run:  python loadtest/kh/verify.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import Harness  # noqa: E402
from replay import run_once, to_legacy  # noqa: E402


def summarise(payload: dict[str, Any]) -> tuple:
    rows = payload.get("rows") or []
    return (
        tuple(str(r.get("id")) for r in rows),
        payload.get("total"),
        payload.get("nRecord"), payload.get("nFolder"), payload.get("nGroup"),
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emails", default=(
        "shrikant.nemiwal@pipeshub.com,darshan.godase@pipeshub.com,"
        "harshit.jaiswal@pipeshub.com,vansh.gupta@pipeshub.com,"
        "shrikant@pipeshubinc.onmicrosoft.com,admin@example.com"))
    parser.add_argument("--sorts", default="name:ASC,name:DESC,updatedAt:DESC,"
                                           "createdAt:ASC,sizeInBytes:DESC,nodeType:ASC")
    args = parser.parse_args()

    # Filters matter more than sorts here: `slim` now builds only the fields
    # something downstream reads, and which fields those are is derived from
    # the generated filter text. A filter whose field went missing would read
    # as null and match nothing -- silently. Each of these names a different
    # slim field.
    filter_cases: list[tuple[str, dict]] = [
        ("none", {}),
        ("search_query", {"search_query": "a"}),
        ("record_types", {"record_types": ["FILE"]}),
        ("indexing_status", {"indexing_status": ["COMPLETED", "FAILED"]}),
        ("origins", {"origins": ["CONNECTOR"]}),
        ("updated_at", {"updated_at": {"gte": 0}}),
        ("size", {"size": {"gte": 1}}),
        ("node_types", {"node_types": ["record"]}),
        ("only_containers", {"only_containers": True}),
    ]

    failures: list[str] = []
    checked = 0

    for email in [e.strip() for e in args.emails.split(",") if e.strip()]:
        harness = Harness(log_level=logging.WARNING)
        try:
            await harness.connect(email=email)
        except SystemExit as exc:
            print(f"SKIP {email}: {exc}")
            continue
        access = await harness.access("verify")
        apps = access["gated_app_ids"]
        print(f"\n=== {email}: {len(apps)} gated apps, "
              f"{sum(len(v) for v in access['by_connector'].values())} grants")

        for spec in [s.strip() for s in args.sorts.split(",") if s.strip()]:
            field, _, direction = spec.partition(":")
            before = len(failures)
            for fname, filters in filter_cases:
                for app_id in apps:
                    try:
                        query, params = await capture_sorted(
                            harness, app_id, access, field, direction,
                            filters=filters)
                    except (SystemExit, KeyError) as exc:
                        failures.append(f"{email} {app_id} capture: {exc}")
                        continue
                    legacy, applied = to_legacy(query)
                    if not applied:
                        failures.append(
                            f"{email}/{app_id}/{spec}/{fname}: NO REVERT "
                            f"APPLIED -- the transforms no longer match the "
                            f"generated query")
                        continue
                    _, new_payload = await run_once(harness, query, params)
                    _, old_payload = await run_once(harness, legacy, params)
                    checked += 1
                    if summarise(new_payload) != summarise(old_payload):
                        failures.append(
                            f"{email}/{app_id[:8]}/{spec}/{fname}: DIFFERS "
                            f"new_total={new_payload.get('total')} "
                            f"old_total={old_payload.get('total')}")
            new_failures = len(failures) - before
            print(f"  sort {spec:<18} {len(apps) * len(filter_cases):>4} cells"
                  f"{'  OK' if not new_failures else f'  {new_failures} FAILED'}")
        await harness.close()

    print(f"\n{checked} cells compared, {len(failures)} failures")
    for f in failures:
        print("  FAIL", f)
    sys.exit(1 if failures else 0)


async def capture_sorted(harness: Harness, app_id: str, access: dict,
                         sort_field: str, sort_dir: str,
                         filters: dict | None = None):
    from app.services.graph_db.neo4j.neo4j_client import Neo4jClient

    captured: dict[str, Any] = {}
    original = Neo4jClient.execute_query

    async def intercept(self, query, parameters=None, txn_id=None):  # noqa: ANN001
        if "$gatedAppIds" in query and "allIds" in query:
            captured["query"] = query
            captured["params"] = dict(parameters or {})
        return await original(self, query, parameters=parameters, txn_id=txn_id)

    Neo4jClient.execute_query = intercept
    try:
        await harness.provider.get_knowledge_hub_connector_page_v3(
            app_id=app_id, org_id=harness.user["org_id"],
            grantee_ids=access["grantee_ids"],
            gated_app_ids=access["gated_app_ids"],
            granted_ids=access["by_connector"].get(app_id) or [],
            limit=50, flatten=True, sort_field=sort_field,
            sort_dir=sort_dir or "ASC", include_total=True,
            filters=filters or {},
        )
    finally:
        Neo4jClient.execute_query = original
    return captured["query"], captured["params"]


if __name__ == "__main__":
    asyncio.run(main())
