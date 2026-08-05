# Query-service performance

What was slow, what changed, and what it is worth — measured before and after across
9 configurations, 3 trials each, on a realistic query mix.

Evidence in `baseline/`: `matrix.csv` (every run), `matrix.txt` (the summary table), and
flame graphs and memory graphs for representative cells. Reproduce with `./perftest.sh`;
see `README.md`.

## Result

| workers | users | before | after | change |
|---|---|---|---|---|
| 1 | 8 | 13.2 | **17.7** | +34% |
| 1 | 16 | 12.9 | **17.7** | +37% |
| 1 | 32 | 12.3 | **18.2** | +48% |
| 2 | 8 | 20.6 | **25.0** | +21% |
| 2 | 16 | 21.9 | **29.2** | +33% |
| 2 | 32 | 20.9 | **29.7** | +42% |
| 4 | 8 | 24.3 | **29.3** | +21% |
| 4 | 16 | 29.5 | **41.6** | +41% |
| 4 | 32 | 31.2 | **40.1** | +29% |

Median throughput, req/min, counted server-side. **Positive in all 9 configurations,
+21% to +48%, median ~+34%.** Zero failed requests across 7,760.

Latency falls in step — 4 workers / 16 users p50 **24.3 s → 17.5 s**; 1 worker / 32 users
**130.7 s → 87.3 s**.

The mechanism is visible directly in the profiles: **Python CPU per turn drops 28–35% in
every cell** (455 → 301 samples/turn at 4w/16u, 274 → 198 at 1w/8u). The gain grows with
concurrency at every worker count, which is what a genuine CPU fix looks like — it pays most
where CPU is the binding constraint.

Measured on Play (8 CPU, 31 GB, Neo4j, `internal_search`, gpt-5.4-mini on Azure), 300 s per
run, 10 queries drawn per user from a seeded PRNG so both arms saw the same workload.

**Baseline = this branch's parent commit**, taken from the deployed image and verified
byte-identical to it, so the two arms differ only by the change under test.

## The problem

A chat turn re-did work proportional to the **whole answer** on **every streamed token** —
roughly 700 times per turn — in three independent places:

1. `TerminalAnswerStreamer._emit_state_delta` re-ran citation normalization over the entire
   accumulated answer, per token.
2. Each emit produced a `STATE_DELTA` carrying the whole answer plus every citation's full
   text — one measured turn sent 1,030 frames averaging 37 KB, **39 MB of SSE to deliver a
   1.5 KB answer**, all encoded here and relayed by the single-threaded Node gateway.
3. Jinja templates were recompiled from constant sources inside per-record loops, and tool
   JSON schemas regenerated on every LLM call.

## What changed

| change | effect |
|---|---|
| Rate-limit live citation resolution (`PIPESHUB_ANSWER_DELTA_INTERVAL_MS`, default 100 ms) | citation normalization no longer per token |
| Per-key pending slots in `QueueEventSink` so coalescing actually engages | SSE 39.4 MB → 1.4 MB per turn |
| `lru_cache` the constant Jinja templates | removes recompilation from per-record loops |
| Cache bound tools per tool set, per request | binds/turn 3.3 → 1.0 |
| Pass the **live** virtual-record map to the fetch tool | repeat fetches hit the map instead of re-downloading |
| Sniff base64 images from a 204-byte prefix | no measurable gain; kept as strictly less work |

Two deserve a note. **The SSE coalescing was a no-op before this change** — one shared
pending slot, and the answer stream interleaves two coalescable kinds per token, so each
write evicted the other and neither ever merged. And **the record cache never worked on the
navigate/lookup path**: `CitationCollector.virtual_records` returns `... or {}`, handing the
write-back cache a throwaway dict whenever the map is still empty, so every repeat fetch
re-downloaded the document.

## Scaling, and where the ceiling is

| workers | best observed (after) | vs 1 worker |
|---|---|---|
| 1 | 18.2 | 1.0× |
| 2 | 29.7 | 1.6× |
| 4 | 41.6 | 2.3× |

Workers scale **sublinearly** — 4 workers gives 2.3×, not 4×. Each count saturates: 1 worker
pins at ~18 req/min and 2 at ~30 regardless of whether 8, 16 or 32 users are offered; only
4 workers still gains between 8 and 16 users. Beyond that, more concurrency buys latency,
not throughput.

CPU per turn *rises* with worker count (198 → 221 → 301 samples/turn at 16–32 users), which
is contention showing up as extra work rather than more work being done.

## Honest limits

- **The workload is a 10-query mix, not production traffic.** It is drawn from a corpus whose
  indexed portion is ~3,300 records — and 1,010 of those are copies of a single document,
  which these queries deliberately avoid. A different corpus will give different absolutes;
  the before/after ratio is the transferable number.
- **Absolute figures here supersede earlier ones.** Previous numbers (23 → 43 req/min) were
  measured with a single repeated query, which the provider prefix-caches and which keeps
  retrieval warm; that overstated throughput by roughly 1.6×. The improvement is real, the
  old capacity figures were not.
- **The arms ran in blocks over ~6 hours.** A drift check re-ran one baseline cell four hours
  later and reproduced it inside its original spread (29.5 [29.3-31.6] → 31.1 [30.3-31.2]),
  so the comparison reflects the code change rather than time of day.
- Throughput divides by elapsed time including the drain after load stops. The report also
  prints a steady-state count; the two diverge most at 1 worker / 32 users (12.3 vs 9.8),
  where turns outlast the load window.

## What is left, in priority order

1. **Node cluster mode.** One process on one core in front of every SSE stream and document
   download — 384 ms p50 against Neo4j's 8 ms at the same concurrency.
2. **Stop fetching whole documents.** A turn downloads and decodes ~96 complete documents to
   quote a few paragraphs. Block-scoped fetch removes both the CPU and the Node traffic.
3. **Fewer LLM calls per turn.** The largest remaining lever on both latency and token spend.

Not worth doing, measured: more than 4 workers, removing the record-escalation judge, and
msgspec for SSE encoding.

A follow-up round has since landed — an accessible-record permission cache, a
memoized citation tick, a shared HTTP session, batched record lookups, and a
compression threshold. It is **unmeasured**: the table above is still the last
verified before/after. See PROFILE.md for what each change targets, and note
that the permission cache cannot be A/B'd with a single load-test identity.

## Method notes

Things that produced wrong conclusions before they were caught:

- **One repeated query is not a workload.** It was the single largest source of error here —
  ~1.6× inflation, plus turns that returned nothing at all still counting as throughput.
- **Count throughput server-side**, and know whether the drain is in your denominator.
- **Attribute a profile frame by caller before believing a hypothesis.** `b64decode` looked
  like image sniffing; 29 of 35 samples were record decompression.
- **Verify a fix engaged.** The first SSE throttle measured as a complete no-op.
- **Track errors.** Until this round nothing counted non-200s: a run that mostly failed and a
  run that was merely slow both showed up only as low throughput.
- **A single trial is ±10%**; medians over 3 trials with the spread shown are the minimum for
  a quotable number.
