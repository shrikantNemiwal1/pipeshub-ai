# Query-service CPU profile

Where the query service actually spends CPU, before and after the changes on this
branch, and what the next optimisation should be.

Measured on Play, **1 worker / 32 users / 300 s, 3 trials aggregated per arm**
(baseline 61,409 samples, current 59,461). One worker is deliberate: it is the
cleanest attribution, because a single GIL means samples map to one timeline.
py-spy `--gil`, so a sample is counted only while Python is actually executing.

Raw data in `baseline/`: `prof_*.txt` (leaf), `cum_*.txt` (cumulative),
`*_cpu.svg` (flame graphs), and `*_cpu.raw.gz` for re-analysis.

## Headline: record fetch and decode is the largest consumer

The single biggest cost is one chain — **download a blob, base64-decode it,
decompress, JSON-parse**:

| frame | leaf % | what it is |
|---|---|---|
| `b64decode` (base64.py:88) | 6.05% | decoding record blobs |
| `_process_downloaded_record` (blob_storage.py:168) | 3.77% | decompress + parse |
| `raw_decode` (json/decoder.py:354) | 3.35% | JSON parse |

**~13% of all executing Python in leaf terms**, before the I/O around it. In
cumulative terms the entry points are unambiguous:

| frame | cumulative % |
|---|---|
| `get_record` (chat_helpers.py:2033) | **15.27%** |
| `get_record_from_storage` (blob_storage.py:945) | **10.18%** |
| `_process_downloaded_record` (blob_storage.py:166) | 6.22% |

A turn downloads and decodes on the order of 100 whole documents to quote a few
paragraphs from them.

## Self CPU, top 25 (current arm)

| % | function |
|---|---|
| 6.05 | `b64decode` (base64.py:88) |
| 4.20 | `read` (asyncio/streams.py:716) |
| 3.77 | `_process_downloaded_record` (blob_storage.py:168) |
| 3.35 | `raw_decode` (json/decoder.py:354) |
| 2.90 | `_run` (asyncio/events.py:88) |
| 1.51 | `write` (asyncio/selector_events.py:1075) |
| 1.48 | uvicorn process frame |
| 1.39 | `iterencode` (json/encoder.py:258) |
| 1.09 | `_unpack` (neo4j packstream v1:368) |
| 1.01 | `new_hydration_scope` (neo4j hydration_handler:101) |
| 0.96 | `reschedule` (asyncio/timeouts.py:71) |
| 0.94 | `transform` (neo4j/_data.py:359) |
| 0.82 | `_unpack` (neo4j packstream v1:357) |
| 0.79 | `wait_for` (asyncio/tasks.py:519) |
| 0.78 | `extract_start_end_text` (chat_helpers.py:3429) |
| 0.76 | `_read_ready__data_received` (asyncio/selector_events.py:1013) |
| 0.75 | `read` (asyncio/streams.py:720) |
| 0.71 | `isawaitable` (inspect.py:473) |
| 0.71 | `receive_into_buffer` (neo4j/_async/io/_common.py:340) |
| 0.69 | `data` (neo4j/_async/work/result.py:793) |
| 0.67 | `data` (neo4j/_data.py:318) |
| 0.65 | `__init__` (pydantic/main.py:250) |
| 0.63 | `call_at` (asyncio/base_events.py:787) |
| 0.53 | `__aiter__` (neo4j/_async/work/result.py:409) |
| 0.49 | `_could_become_confidence_line` (streaming.py:469) |

## Cumulative, application frames (current arm)

Generic scaffolding (asyncio, click, uvicorn, starlette) filtered out.

| % | frame |
|---|---|
| 15.27 | `get_record` (chat_helpers.py:2033) |
| 10.18 | `get_record_from_storage` (blob_storage.py:945) |
| 8.90 | `data` (neo4j result.py:793) |
| 8.84 | `execute_query` (neo4j_client.py:284) |
| 7.88 | `_wrapper` (hooks/retry_with_status.py:77) |
| 7.71 | `bound` (hooks/middleware/wrapper.py:59) |
| 7.57 | `call_model_wrapped` (agent/hook_dispatch.py:157) |
| 7.52 | `inner` (neo4j/_async/io/_common.py:204) |
| 7.41 | `_call_llm` (agent/__init__.py:666) |
| 6.43 | `__aiter__` (neo4j result.py:409) |
| 6.30 | `fetch_message` (neo4j _bolt.py:865) |
| 6.24 | `_wait_for_io` (neo4j _bolt_socket.py:173) |
| 6.22 | `_process_downloaded_record` (blob_storage.py:166) |
| 5.64 | `receive_into_buffer` (neo4j _common.py:340) |
| 5.51 | `timed_run` (backend_timing.py:107) — **the profiling probe itself** |
| 5.44 | `recv_into` (neo4j _bolt_socket.py:221) |
| 5.00 | `_buffer_one_chunk` (neo4j _common.py:54) |
| 4.95 | `_on_delta` (answer_streamer.py:175) |
| 4.93 | `on_event` (answer_streamer.py:117) |
| 4.79 | `_produce` (chat_modes/bridge.py:519) |
| 4.68 | `_emit_state_delta` (answer_streamer.py:198) |
| 4.36 | `stream` (langchain_transport.py:418) |
| 4.32 | `_build_linked_record_context_metadata` (chat_helpers.py:628) |
| 3.99 | `normalize_citations_and_chunks` (citations.py:633) |
| 3.97 | `_get_virtual_ids_for_connector` (neo4j_provider.py:4397) |

## What the branch changed, visible in the profile

Four costs present in the baseline arm are **gone from the current top 25**:

| frame | baseline | current |
|---|---|---|
| `iterencode` (SSE JSON encode) | 4.35% | 1.39% |
| `_renumber_citation_links` (citations.py:321) | 1.63% | absent |
| `_compile` (jinja2/environment.py:709) | 1.57% | absent |
| `_wrap_bare_refs` (citations.py:147) | 1.46% | absent |

That is the SSE coalescing fix, the citation rate limit, and the Jinja template
cache, each showing up exactly where predicted. Total executing Python per turn
fell about 28–35% across the matrix.

## Where turn time goes (4 workers / 16 users, current)

| phase | mean | % of turn |
|---|---|---|
| `llm_stream` | 14,597 ms | **72.7%** |
| `retrieval_wait` | 4,210 ms | 20.9% |
| `agent_create` | 1,160 ms | 5.7% |
| everything else | <50 ms each | <0.3% |

## Backend calls per turn

| | calls/turn | p50 |
|---|---|---|
| Neo4j | 341–397 | **5–14 ms** |
| Node API | 94–108 | **73–244 ms** |

Neo4j p50 stays flat (5–14 ms) across every worker/user combination — it is not
queueing and it is **not the constraint**. The Bolt frames in the profile
(`_unpack`, `new_hydration_scope`, `transform`, `receive_into_buffer`, ~5%
combined) are protocol decode work, a symptom of **call volume**, not slow
queries.

Node is the opposite: its p50 grows with load — 116 ms at 8 users, 236 ms at
16 — while Neo4j stays flat. That is queueing at the single-threaded gateway,
and most of those ~100 calls per turn are whole-document downloads.

## Recommended next step

**Block-scoped fetch — stop downloading whole documents.**

It is the only change that hits both dominant terms at once:

- It removes the largest CPU consumer in the service (`get_record` chain, ~15%
  cumulative, ~13% leaf).
- It cuts the ~100 Node calls per turn, which is what makes the single-threaded
  gateway queue.
- Fetched document text is re-sent to the model on every LLM call of the turn,
  so it should also shrink `llm_stream` — the 72.7% term.

Ranked alternatives, and why they come second:

2. **Fewer / cheaper LLM calls per turn** — the largest single term, but a
   bigger design change.
3. **Node cluster mode** — cheap and config-level; its latency demonstrably
   grows with throughput.
4. **Batch the Neo4j fan-out** — ~370 calls per turn is an N+1 smell and it
   loads the box for other tenants, but at 9 ms p50 it is not the bottleneck.
   `get_nodes_by_field_in` already exists
   (`services/graph_db/interface/graph_db_provider.py:798`) and is used by
   Salesforce and one agent route, but **not** on this path —
   `utils/fetch_full_record.py` still does `get_document` and
   `check_record_access_with_details` per record.

Ceiling check for anything in `retrieval_wait`: eliminating that phase entirely —
impossible — takes a turn from 20.1 s to 15.9 s, about +26% throughput. That
bounds how much Neo4j/Node work alone can be worth.

## Follow-up round

Landed after this profile, targeting the terms above and one it missed. The
percentages below are the profile shares each change aims at, **not** results.
The first has since been measured under load — see the A/B section that follows;
the rest are still unmeasured.

| change | targets | switch |
|---|---|---|
| Redis cache of the accessible-record maps, per KB / per app-level connector / per (user, connector) | `_get_kb_virtual_ids` + `_get_virtual_ids_for_connector`, ~8.3% of all CPU, recomputed per search | `PIPESHUB_ACCESSIBLE_RECORDS_CACHE=off` |
| Memoized `generate_text_fragment_url`; live-citation tick 100 ms → 250 ms | `generate_text_fragment_url`/`extract_start_end_text`, ~2.4% | `PIPESHUB_ANSWER_DELTA_INTERVAL_MS` |
| Shared aiohttp session + cached config reads in `blob_storage` | one TCP handshake and 3 etcd round trips per record fetch (~100/turn) | — |
| Batched virtual-record → document lookups | the per-record `get_nodes_by_filters` + `get_document` pair, ~1.7% | — |
| Compress stored records only above 20 MB | the `b64decode` + JSON-envelope parse on read, ~9% — **only for records indexed after the change** | `PIPESHUB_RECORD_COMPRESSION_THRESHOLD_BYTES` |

Seed accounts for multi-identity runs with `./seed_users.py` and set
`PIPESHUB_USERS`. Two things to keep in mind when doing so: the Node rate
limiter is per user, so an N-user run has N× the request budget of a
single-token one, and on a stack whose corpus comes from real connectors the
seeded users will have a far smaller corpus than the account that owns those
connections — see the multi-identity note below before comparing anything
across those two shapes.

### Load-test A/B — accessible-records cache, 1 worker / 8 users

Full `perftest.sh` runs on the local dev stack, cache off vs on, 300 s each,
with flame graphs, memory graphs, phase timings and backend call counts in
`results/`. **Read the CPU share, not the throughput** — see the caveats.

Per run, because averaging hides what matters here:

| run | cache | turns | permission traversal, % of executing Python |
|---|---|---|---|
| cacheon t1 | on | 9 | 5.41% |
| cacheon t2 | on | 7 | 20.86% |
| **cacheon t3** | **on** | **19** | **0.39%** |
| cacheoff t1 | off | 23 | 14.25% |
| cacheoff t2 | off | 26 | 15.90% |
| **drift** | **on** | **21** | **0.45%** |

**Once the cache is warm the permission traversal essentially disappears —
~15% → ~0.4%, replicated across two independent runs** (`cacheon t3` and the
drift cell) at turn counts comparable to the cache-off runs, so this is not an
artifact of one arm completing more turns than the other.

Trials 1 and 2 are the outliers, and they are informative: cold cache *and* cold
system, only 9 and 7 completed turns, so cold fills dominated a small sample —
t2 even exceeds the cache-off share. The TTL is 300 s and a run is 300 s, so
every trial refills from scratch; a TTL longer than the run, or a longer run,
removes that. Averaging all three cache-on trials gives 8.56%, which understates
the steady-state effect by mixing in the fill-dominated runs.

The service is CPU-pinned at 1 worker: every run produced ~14–15k `--gil`
samples (14404 / 13975 / 15461 / 14896 / 14803) regardless of arm. The profile
therefore measures how a *fixed, saturated* CPU budget is spent, which is what
makes the share the meaningful number. Other groups rising (record fetch +
decode reaches 38% in `cacheon t3`) is the renormalisation this document already
warns about, not a regression.

**Throughput from these runs is not usable, and is deliberately not quoted.**
Both arms were run once each in sequence rather than interleaved — the mistake —
and the stack was still warming throughout. In wall-clock order:

| # | run | cache | p50 turn | truncated |
|---|---|---|---|---|
| 1 | cacheon t1 | on | 208 s | 4/13 |
| 2 | cacheon t2 | on | 162 s | 5/12 |
| 3 | cacheon t3 | on | 132 s | 0/19 |
| 4 | cacheoff t1 | off | 99 s | 1/24 |
| 5 | cacheoff t2 | off | 64 s | 1/27 |

A monotonic decline straight through the arm boundary between #3 and #4, with no
step at the change. `aggregate.py` independently rejected runs #1 and #2 (41.7%
error rate) — the truncations, not the cache, are what its throughput column was
reporting.

A drift-check cell (`drift_w1_u8_t1`, cache **on**, run last on the warmest
system) landed at a 130 s p50 with 0 truncations: far better than its own
cold-start 208 s, but not down at the 64 s of the last cache-off run. So warmup
is real and large, and it is **not** the whole story — the cache-off arm alone
swung 99 s → 64 s between two consecutive identical runs.

The honest reading is that wall-clock variance here (64–208 s p50 for what is
nominally the same configuration) is larger than any effect this change could
have. Comparing only the two runs that were both warm and error-free:

| | req/min | samples/turn | errors |
|---|---|---|---|
| cacheoff (t1+t2) | 4.0 | 598 | 4.2% |
| drift cell, cache **on** | 4.0 | **497** | 0.0% |

**Throughput is identical, and executing Python per turn drops ~17%.** That is
the expected shape: at 8 users this stack is bound by the LLM and by record
fetching, not by the CPU the cache frees. The freed CPU is real — it is simply
not what limits throughput at this saturation. It would matter at a worker count
or user count where Python CPU is the binding constraint, which is what the
1-worker profile in this document was originally chosen to expose.

The flame graphs show this without any analysis: search `cpu.svg` for
`_get_virtual_ids_for_connector` and it is present in every cache-off run and
absent from the warm cache-on ones, where `accessible_records_cache` frames
appear instead.

Two things the run surfaced that have nothing to do with the cache:

- **8 users over-saturates this stack.** Turns take 100–300 s and some SSE
  streams are cut at curl's 300 s limit. They still record HTTP 200, because SSE
  sends the status line immediately and streams afterwards — only `curl_exit=28`
  reveals the truncation. Anything past ~4 users here is beyond the knee.
- **`record fetch + decode` is 15–38% with `b64decode` alone at ~9%.** That is
  the single largest item in the profile and exactly what the 20 MB compression
  threshold targets — but only for records indexed *after* that change, so this
  corpus still pays it in full. Re-indexing is what would show that win, and on
  this evidence it is the biggest of the remaining levers.

**What a multi-identity run can and cannot show here.** `multiuser_w1_u8_t1`
drives 9 real accounts instead of 9 connections on one. It is not comparable to
the cells above on throughput: seeded users reach only the KB and S3, so their
turns average 50–72 s against 100–300 s for the admin's 3,687-record corpus.

It also cannot exercise the per-user side of the cache. The record-level
connectors (Drive, Gmail, Jira, Confluence, Slack, Notion) carry ACLs synced
from the real SaaS sources, which no synthetic account can be granted — so every
`cusr` hash stays at one field no matter how many users are driving load. S3 is
`APP_LEVEL` and lands in a shared `capp` entry instead. Multi-user fanout of the
hash is covered by unit tests
(`test_accessible_records_cache.py::test_user_connector_entries_are_per_user`),
not by this run; what the run does show is many identities sharing the `kb` and
`capp` entries concurrently without interference.

### Load-test A/B — 1 worker / 32 users, 3 trials interleaved

The original profile in this document was taken at 1w/32u, so the A/B was
repeated there properly: three arms × three trials, interleaved
`off, cold, warm` per trial, each arm restarted through a switcher that verifies
the running process reports the arm that was asked for.

- **off** — `PIPESHUB_ACCESSIBLE_RECORDS_CACHE=off`.
- **cold** — cache on with the shipped 300 s TTL. The load window is also 300 s,
  so this arm refills mid-run by construction.
- **warm** — cache on, TTL 3600 s, preceded by a 60 s throwaway load run, so the
  maps are already built when the measured window opens.

A first attempt at 1w/32u (recorded in an earlier revision of this section) read
1.35× and was discarded: it ran one trial per arm, and the arms completed 33 vs
15 turns for near-identical CPU, so share and per-turn normalisation disagreed.
Three interleaved trials fix that, and the two normalisations now agree.

**Traversal CPU, the frames the cache exists to remove** (`_get_kb_virtual_ids`,
`_get_virtual_ids_for_connector` and the two user-independent loaders):

| trial | | off | cold | warm |
|---|---|---|---|---|
| t3 | share of executing Python | 11.56% | 5.44% | **2.35%** |
| t3 | samples per completed turn | 30.8 | 15.9 | **6.9** |
| t2 | share | 13.26% | 7.99% | **5.42%** |
| t2 | samples per completed turn | 77.7 | 30.4 | **15.8** |

**Warm cache cuts the traversal 4.5–4.9× at 32 users, and share and per-turn
agree** — in t3 the three cells completed 49 / 48 / 49 turns with zero errors, so
neither number is being manufactured by a turn-count difference. Trial 3 is the
one to read: it is the warmest, all three cells are error-free, and the arms ran
consecutively (12:59 / 13:07 / 13:19) in the same drift regime.

The cold arm sitting halfway between is the TTL doing exactly what it says: with
TTL 300 s and a 300 s window it refills mid-run, so it pays roughly half the
traversal the off arm does. A TTL longer than the run removes the rest.

**The cache's own overhead is negligible.** Redis round-trips, JSON decode,
single-flight locks and the live KB-access probe that still runs on every
request, all together:

| | off | cold | warm |
|---|---|---|---|
| cache machinery | 0.00% | 0.11% | 0.17% |
| live KB-access probe | 0.00% | 0.01% | 0.06% |

(t3; the probe is `_get_accessible_kb_ids`, deliberately left live so KB
*membership* changes are never served stale.) The one exception is the first
cold fill under concurrency, where cache machinery reached 3.27% (`cold_t1`).

**Throughput is not separable at this config and is not claimed.** Nominally
identical cells ranged 3.4–8.0 req/min:

| | t1 | t2 | t3 |
|---|---|---|---|
| off | 5.40 | 3.40 | 7.70 |
| cold | 3.40 | 4.20 | 7.10 |
| warm | 4.70 | 8.00 | 7.10 |

No consistent ordering. In t3, warm completed the same 49 turns as off while
burning 10% more executing-Python samples (14,393 vs 13,075) — the
renormalisation this document warns about: with less time blocked on Neo4j the
process spends more of each second executing Python, at a near-identical sample
rate (34.6 vs 34.2/s). The freed CPU is real; at 32 users on one worker it is
not what limits throughput, which is the same conclusion the 8-user run reached.

**A global warmup drift dominates this matrix**, which is why interleaving was
required: completed turns rose monotonically with trial index in *every* arm
(off 24→49, cold 28→48, warm 33→49). Comparisons are only valid within a trial.

**The cold-fill stampede is real but not fully isolated.** The cold arm produced
the only significant truncations — 38.6% and 19.2% error rates in t1 and t2
against 0% for warm in all three trials — consistent with 32 concurrent requests
missing at once and queueing behind the single-flight lock. t1 is confounded by
system drift; t2 is not, and there the ordering still holds (off 5.6%, cold
19.2%, warm 0%). Two trials is suggestive, not conclusive. The mitigation is the
same either way: **a TTL longer than the traffic pattern's idle gap**, so fills
are rare rather than periodic.

At 8 users the same change measures 35×. The 32-user figure is smaller because
the box is saturated and the traversal is a smaller slice of a longer turn — not
because the cache works less well.

Caveat on `off_t1`: it was re-run last (13:28) because a concurrent `pytest`
invocation stole CPU from the original, so it sits at the *end* of the
wall-clock order, not at trial-1 drift level. It is excluded from the
within-trial comparisons above. Its own sample rate (22.1/s against ~34/s
everywhere else) marks it as an outlier regardless.

Three bugs in the harness itself, found while getting these runs to work:

- `instrument.sh` wires the backend probe by inserting an import *before the
  first `def`/`class` line* of `runtime_threads.py` — a file that contains
  neither, so the insert silently no-ops while still printing "wired", and
  `BACKENDAGG`/`LOOPLAG` never appear. Appending the import instead works.
- `perftest.sh`'s multi-user token resolution used `mapfile -t`, which strips
  `\n` but not `\r`. On Windows the helper's stdout is CRLF, so every token
  carried a trailing carriage return, the `Authorization` header was malformed,
  and the auth probe aborted the run with a bare `HTTP 400`. Fixed here by
  piping through `tr -d '\r'`.
- An arm-switching helper that located the service by scraping a pid out of a
  log file measured **both arms against the same process** once the service was
  restarted onto a different log: it killed a stale pid, the real service
  survived, and the replacement failed to bind port 8000 with `Errno 10048`
  while the run carried on regardless. Any switcher must find the service by
  port and then *verify the running process reports the arm that was asked for*
  — a silent mis-switch produces confident, wrong numbers. `PIPESHUB_QUERY_LOG`
  pointing at the same stale file also made throughput read zero, which looked
  like saturation rather than a broken measurement.

### Measured earlier — isolated timings

Local dev stack (Neo4j, Redis, 3 seeded users, one org with ~3,700 indexed
records). Two measurements, because they answer different questions.

**The call itself**, timed in-process against the live graph, cache off vs on,
median of 7 calls per user:

| user's accessible corpus | cache off | cold fill | cache on | |
|---|---|---|---|---|
| 3,687 records | 175.5 ms | 620 ms | **18.6 ms** | 9.5× |
| 31 records | 19.8 ms | 16.3 ms | **7.1 ms** | 2.8× |

The saving tracks corpus size, which is what the design predicts: the cost is
proportional to the rows the permission traversal returns.

**End to end**, on `/api/v1/search`, 3 users × 12 rounds per arm (n=36):

| arm | median | p25–p75 |
|---|---|---|
| cache off | 9,412 ms | 8,399–10,707 |
| cache on | **8,645 ms** | 8,239–9,885 |

8.1% faster, but the run-to-run spread is wider than the difference — about
1.7σ, so treat this as *consistent with* the isolated measurement rather than
as a result in its own right. A search here is dominated by vector search and
record fetching; 157 ms of permission traversal is a small slice of it. The
CPU-share framing in this document, not wall-clock latency, is where this
change is meant to pay.

Correctness, verified against the same stack:

- The `APP_LEVEL` scan returns **exactly** what the per-user 8-path traversal
  returns — 31/31 records for three different users on the S3 connector, and
  identical map sizes at 3,687 records. No over- or under-sharing.
- Repeat searches vary run to run (45/48/45 records) **with the cache off too**
  (46/46/43). That is approximate vector search, not the cache.
- Creating a KB folder dropped that KB's entry immediately, far inside the 300 s
  TTL — event invalidation reaches across service boundaries.
- `PIPESHUB_ACCESSIBLE_RECORDS_CACHE=off` writes zero Redis keys and logs that
  it is disabled.
- **Redis is a hard dependency of the query service regardless of this cache.**
  With Redis stopped, searches hang identically whether the cache is on or off —
  the config store holds the JWT secret every request needs, so requests die in
  auth long before reaching any of this. Pre-existing, and worth its own fix.

## Caveats

- `timed_run (backend_timing.py:107)` at 5.5% cumulative is the load-test probe
  wrapping every Neo4j and HTTP call. It inflates these profiles slightly versus
  an uninstrumented production run.
- The one-worker configuration is the clearest attribution, not the deployed
  shape. Per-turn CPU rises with worker count (198 → 221 → 301 samples/turn at
  1/2/4 workers) as contention appears.
- Percentages do not add: removing one cost renormalises the rest. Judge a change
  by its own signature, not a running total.
- The `llm_stream` share is measured; the claim that document fetching inflates
  prompt size is **inferred** from fetch volume, not measured here. Logging
  prompt tokens per LLM call for one run would confirm or kill it.

## Reproducing

```bash
cd loadtest
cp .env.example .env          # add TOKEN
./instrument.sh on
./perftest.sh w1_u32 32 300 1
./aggregate.py results
```

Flame graphs open in a browser: width is time, colour means nothing, click to
zoom, search box (top right) totals a subsystem. To re-derive the tables above
from a raw profile, see `instr/top_functions.py`.
