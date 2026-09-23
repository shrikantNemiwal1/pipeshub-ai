# Knowledge Hub v3 query harness

Measures and A/B-tests the **v3 global flatten** path against a real store.

Everything here is diagnostic tooling, following the `loadtest/` convention:
**nothing under `backend/` is modified to switch an arm.** The one exception is
`mutate.py`, which deliberately edits the provider and restores it in a
`finally` — it is a test-quality tool, not a measurement one.

Findings and numbers live in **`docs/kh-v3/optimization-log.md`**. This file is
just how to run it.

---

## Quick start

```bash
# reproduce the live cost, with a per-connector breakdown
python loadtest/kh/smoke.py --email you@example.com --reps 5 --quiet

# A/B the current provider against the pre-optimization query
python loadtest/kh/replay.py --email you@example.com \
    --arms control,legacy --reps 9 --warmup 3 --profile \
    --out loadtest/kh/results/run1

# prove the landed changes are semantics-preserving across users and sorts
python loadtest/kh/verify.py

# mutation-test the v3 scope-arm module
python loadtest/kh/mutate.py
```

Credentials are read from `backend/python/.env` (`NEO4J_*`) or the real
environment. The password is never printed.

---

## Files

| file | what it does |
|---|---|
| `env.py` | credentials, user resolution, and the **read-only guards** |
| `instr.py` | per-call Neo4j timing, attributed across the concurrent fan-out |
| `harness.py` | builds a real `Neo4jProvider` in process; runs one decomposed request |
| `smoke.py` | reproduce the live cost, print the per-connector breakdown |
| `replay.py` | capture the real query, replay text-transform **arms** of it |
| `verify.py` | compare optimized vs reconstructed-legacy across users x sorts x connectors |
| `mutate.py` | break one provider rule at a time and confirm the tests go red |

## Arms

`replay.py --arms all` lists everything. The useful groups:

- **preserving** — must return a byte-identical page. `control`, `legacy`,
  `dedup_allids`, `slim_min`, `hop_factored`, `anchor_belowseeds`, `planner_dp`.
- **breaking** — diagnostic only, expected to change the result:
  `depth1/5/13`, `kill_A1..A7`, `no_inherit_probe_*`, `no_hidden_guard`,
  `no_notin_tests`, `no_enrich`, `no_counts`, `no_sort`.

An arm whose page differs from `control` is reported **BROKEN**, never as a
speedup. An arm whose transform does not match the current query text is
**SKIPPED loudly** — a silently-inapplicable edit would otherwise read as
"1.0x, no effect", which looks like a finding.

---

## Things this harness had to learn the hard way

- **Measure the ratio, not the absolute.** Medians drift between batches (a
  control measured 578 ms in one and 505 ms in another). Every batch carries its
  own interleaved control, and only within-batch ratios are quoted.
- **Warm up per arm.** Each distinct query text gets its own plan-cache entry.
  A warm plan for one arm and a cold one for another fabricates a speedup.
- **A PROFILEd run is never a timing sample.** Profiling adds counters and
  forces materialisation; `--profile` writes separate files.
- **Read wall time beside db hits.** `x IN <runtime list>` is pure CPU with
  *zero* db hits, so a db-hits-only view can miss the cost entirely — and can
  also mislead in the other direction: `no_notin_tests` removed work and made
  the query **slower** (0.57x).
- **Instrument the driver, not the provider.** `Neo4jClient._run_autocommit`
  returns `await result.data()` and never calls `consume()`, so the
  `ResultSummary` is discarded before the provider sees it.
- **Mutate every occurrence.** Several rule strings appear both in the live page
  query and in `get_knowledge_hub_visible_set_v3`, which nothing calls — and the
  dead copy comes first in the file. Mutating only the first reports a false
  "SURVIVED". `mutate.py` prints how many sites it changed.
- **Assert ordered ids, not a set.** A reordering silently breaks keyset paging
  even when the set matches.
- **Check the corpus did not move.** The live stack runs on the same host and
  can be indexing; every run compares node and relationship counts at both ends.
