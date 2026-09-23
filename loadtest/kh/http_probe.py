"""Time the Knowledge Hub endpoint over real HTTP.

This is the level the user actually feels, and the one the in-process harness
cannot see: it adds FastAPI, the `require_scopes(KB_READ)` auth dependency,
dependency injection and JSON serialisation on top of the query.

    KH_TOKEN=<bearer> python loadtest/kh/http_probe.py --reps 100 --sleep 0.5

The token is read from the environment and never written to disk or echoed --
it is a real credential, and `results/` is gitignored but not secret.

Notes on method:
  * One `requests.Session`, so connections are reused the way a real client's
    are. Per-call connection setup would otherwise be measured as latency.
  * `--sleep` puts a gap between calls so each one meets a settled server
    rather than queueing behind its predecessor. Back-to-back calls measure
    throughput under self-inflicted load, which is a different question.
  * Warm-up calls are discarded: the first request after a restart pays plan
    cache and connection-pool costs no later one does.
  * Non-200s are counted and excluded from the percentiles, and reported --
    a wall of 401s otherwise reads as a suspiciously fast run.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

DEFAULT_URL = (
    "http://localhost:8088/api/v1/knowledge-hub/nodes?flattened=true&limit=50"
)


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile on a sorted list."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(1, min(len(ordered), int(round(pct / 100.0 * len(ordered)))))
    return ordered[k - 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="seconds between calls")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    token = os.environ.get("KH_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set KH_TOKEN to a bearer token.")

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})

    for _ in range(args.warmup):
        session.get(args.url, timeout=60)

    samples: list[float] = []
    failures: dict[int, int] = {}
    total_items = None
    rows = None

    started = time.time()
    for i in range(args.reps):
        t0 = time.perf_counter()
        response = session.get(args.url, timeout=60)
        elapsed = (time.perf_counter() - t0) * 1000.0
        if response.status_code == 200:
            samples.append(elapsed)
            if total_items is None:
                body = response.json()
                total_items = (body.get("pagination") or {}).get("totalItems")
                rows = len(body.get("items") or [])
        else:
            failures[response.status_code] = failures.get(response.status_code, 0) + 1
        if args.sleep and i < args.reps - 1:
            time.sleep(args.sleep)
    wall = time.time() - started

    if not samples:
        raise SystemExit(f"No successful calls. Statuses: {failures}")

    stats = {
        "n": len(samples),
        "failures": failures,
        "min": min(samples),
        "p50": percentile(samples, 50),
        "p80": percentile(samples, 80),
        "p90": percentile(samples, 90),
        "p95": percentile(samples, 95),
        "p99": percentile(samples, 99),
        "max": max(samples),
        "mean": statistics.fmean(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "totalItems": total_items,
        "rowsReturned": rows,
        "sleep_between_s": args.sleep,
        "wall_s": round(wall, 1),
        "url": args.url,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    print(f"\n{args.label or 'http'}  n={stats['n']}  "
          f"failures={stats['failures'] or 'none'}  "
          f"totalItems={total_items}  rows={rows}")
    print(f"  wall {stats['wall_s']}s with {args.sleep}s between calls")
    print("  ---------------------------------------------")
    for key in ("min", "p50", "p80", "p90", "p95", "p99", "max", "mean", "stdev"):
        print(f"  {key:>6} {stats[key]:8.1f} ms")

    if args.label:
        out = Path(__file__).resolve().parent / "results" / "http"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{args.label}.json").write_text(
            json.dumps({**stats, "samples": [round(s, 2) for s in samples]},
                       indent=2),
            encoding="utf-8",
        )
        print(f"\n  raw samples -> results/http/{args.label}.json")


if __name__ == "__main__":
    main()
