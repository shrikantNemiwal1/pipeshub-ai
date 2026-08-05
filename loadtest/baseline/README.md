# Measurement evidence

Everything behind `../PERFORMANCE.md` (results) and `../PROFILE.md` (CPU
attribution). Captured on Play — 8 CPU, 31 GB, Neo4j, `internal_search`,
gpt-5.4-mini on Azure.

**Design:** 2 arms × {1,2,4} workers × {8,16,32} users × 3 trials, 300 s each,
plus a 3-run drift check. 7,760 requests, **zero failures**. Queries drawn per
user from a seeded PRNG over `../queries.txt`, so both arms saw the *same*
workload rather than merely the same distribution.

`baseline_*` is this branch's parent commit, taken from the deployed image and
verified byte-identical to it. `current_*` is the branch. Nothing else differs.

## Files

| file | what it is |
|---|---|
| `matrix.txt` | summary table — median and min-max per configuration, plus before/after |
| `matrix.csv` | every run, one row each |
| `prof_baseline.txt` / `prof_current.txt` | leaf (self) CPU, top 25, 1w/32u × 3 trials |
| `cum_baseline.txt` / `cum_current.txt` | cumulative CPU by application frame |
| `*_cpu.svg` | flame graphs |
| `*_mem.svg` | memory over the run |
| `*_cpu.raw.gz` | raw py-spy profiles for the 1w/32u cells, for your own analysis |

## Suggested reads

- **`baseline_w1_u32_t1_cpu.svg` vs `current_w1_u32_t1_cpu.svg`** — same
  configuration. Search `iterencode`, `_compile`, `_renumber_citation_links`:
  each shrinks sharply or disappears.
- **`baseline_w4_u16_t1_cpu.svg` vs `current_w4_u16_t1_cpu.svg`** — the deployed
  shape; 455 vs 301 Python samples per turn.
- The `w1_u8` / `w1_u16` / `w2_u16` graphs show how the mix shifts with load.

## Re-analysing a raw profile

```bash
python3 - <<'PY'
import gzip, collections
leaf = collections.Counter(); total = 0
with gzip.open("current_w1_u32_t1_cpu.raw.gz", "rt", errors="replace") as fh:
    for line in fh:
        stack, _, cnt = line.strip().rpartition(" ")
        if not cnt.isdigit(): continue
        frames = [f for f in stack.split(";") if f]
        if not frames: continue
        total += int(cnt); leaf[frames[-1]] += int(cnt)
for f, c in leaf.most_common(25):
    print(f"{c*100/total:6.2f}%  {f}")
PY
```

Or `python3 ../instr/top_functions.py <uncompressed .raw>` for the subsystem view.

## Caveats

`timed_run (backend_timing.py:107)` at ~5.5% cumulative is the load-test probe
wrapping every Neo4j and HTTP call — these profiles are slightly inflated versus
production. Absolute req/min depends on this corpus, whose indexed portion is
~3,300 records with 1,010 of them copies of one document (the queries avoid it);
the before/after **ratio** is the transferable number.

Open an `.svg` in a browser: width is time, colour means nothing, click to zoom,
search box (top right) totals a subsystem.

A point-in-time record, not a fixture — nothing reads these. Regenerate with
`./perftest.sh <label> <users> <seconds> <workers>` then `./aggregate.py results`.
