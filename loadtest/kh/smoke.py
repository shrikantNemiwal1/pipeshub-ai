"""Prove the in-process harness reproduces the live global-flatten cost.

Run:  python loadtest/kh/smoke.py --email someone@example.com
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import Harness  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", default=None)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--quiet", action="store_true",
                        help="provider logger at WARNING (the 'nolog' arm)")
    args = parser.parse_args()

    harness = Harness(log_level=logging.WARNING if args.quiet else logging.INFO)
    await harness.connect(email=args.email, user_id=args.user_id)
    print("user:", json.dumps(harness.user, indent=None))

    before = await harness.counts()
    print("corpus:", before)

    for rep in range(args.reps):
        record = await harness.run_global_flatten(limit=args.limit)
        row = record.as_row()
        print(f"\nrep {rep}: {json.dumps(row)}")
        for c in sorted(record.connectors, key=lambda c: -c.wall_ms):
            server = "n/a" if c.server_ms is None else f"{c.server_ms:7.1f}"
            print(f"   {c.app_id[:8]}  wall={c.wall_ms:8.1f}ms  server={server}ms"
                  f"  data={c.data_ms:6.1f}ms  rows={c.rows}  granted={c.n_granted_ids}")

    after = await harness.counts()
    if before != after:
        print(f"WARNING corpus changed {before} -> {after}")
    await harness.close()


if __name__ == "__main__":
    asyncio.run(main())
