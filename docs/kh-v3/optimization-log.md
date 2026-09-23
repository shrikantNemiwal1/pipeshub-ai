# KH v3 global-flatten optimization log

A running record of what was tried, what it measured, and what was kept.
Newest section last. Every number here came from `loadtest/kh/`; nothing is
estimated or carried over from an earlier branch.

**Status: 3 changes landed in `neo4j_provider.py`.**

- Global flatten request **1.46x - 1.49x faster**, measured back-to-back twice
  (824.7 -> 565.5 ms busy; 471.0 -> 316.4 ms quiet).
- The dominant connector's query **1.51x - 1.60x**, and db hits
  **749,077 -> 435,609 (-42%)** — the db-hit figure is the durable one, since
  absolute timings move by ~2x with machine load and the ratio does not.
- Result set **byte-identical**: **1,782** (user x sort x filter x connector)
  cells compared on real data, 0 differences; integration failure set unchanged
  from pristine HEAD.
- Section 8 records **pre-existing v3 gaps** found while running the suite as a
  gate. Those matter more than the latency work -- most of v3's scope block is
  not exercised by the acceptance fixture at all, and nothing gated the v3 hop
  rule. Two new modules (`test_provider_v3_scope_arms.py`,
  `test_provider_v3_hop_rule.py`) add **20 tests** (19 pass, 1 xfail), with
  **7/7 mutations caught**.
- **One security finding, 8.1c:** the v3 scope block never re-checks `orgId` on
  the nodes it collects, so a node with another org's `orgId` but this
  connector's `connectorId` and a grant is returned. **Latent** on this store
  (every node matches its app's org, one Organization) — but it needs your
  decision, not a silent fix.

**If you read two things:** section **8.1** (five of v3's seven scope arms
cannot match a single node anywhere in the integration suite — so "423 tests
green" was never evidence that the scope block works) and section **8.1c** (the
cross-org over-share). The latency win is real but bounded; those two are the
findings with consequences.

### What is on disk (nothing is committed)

| path | status | what |
|---|---|---|
| `backend/python/app/services/graph_db/neo4j/neo4j_provider.py` | modified | the three changes (+65/-24) |
| `backend/python/tests/integration/graph_permissions/test_provider_v3_scope_arms.py` | new | 14 tests, arms 2-6 + deleted + cross-org |
| `backend/python/tests/integration/graph_permissions/test_provider_v3_hop_rule.py` | new | 6 tests, the hop truth table |
| `docs/kh-v3/optimization-log.md` | new | this file |
| `loadtest/kh/` | new | the harness (8 files) |
| `loadtest/.gitignore` | modified | ignore `captures/` — they hold a real user's grant ids |

---

## 1. Scope and ground rules

- **v3 only.** v2, v1 and Arango are out of scope. (v2 is unreachable from the
  service anyway: `KH_VISIBLE_SET_V3` / `KH_CHILDREN_V3` in
  `deployment/docker-compose/docker-compose.yml:125,132` are read by **nothing**
  -- exhaustive repo grep -- and `ba8d4e6e8` routes `search_page`
  unconditionally through `get_knowledge_hub_connector_page_v3`.)
- **Global flatten only** (`start_is_app=True, flatten=True`). Scoped browse is
  acceptable today and was not optimized.
- Measured on the **dev store only**: `neo4j-container`, Neo4j **2025.12.1
  Community**.
- Every candidate must return a **byte-identical page** (ordered ids + `total` +
  per-type counts) or it is reported as broken, not as a speedup.

### Measurement environment

| | |
|---|---|
| Neo4j | 2025.12.1 **Community**, slotted runtime only |
| Page cache / store | 512 MB / **58 MB** -- everything fits, so IO is not the constraint |
| Host | 16 cores, 11.68 GiB, native stack (Node 3000, Next 3001, query 8000, connectors 8088) |
| Data | 16,445 Record, 516 RecordGroup, 893 Group, 664 User, 16 App |
| Edges | NODE_RELATION 17,004 · INHERIT_PERMISSIONS 16,964 · BELONGS_TO 18,157 · PERMISSION 4,481 |
| Test user | `shrikant.nemiwal@pipeshub.com` — 29 grantees, 9 gated apps, 378 grants |

### What Community Edition takes away

This shaped the whole approach, so it is worth recording:

- **Query logging is Enterprise-only.** `db.logs.query.enabled=VERBOSE` and
  `db.logs.query.threshold=0s` *are set* on this container and `/logs/query.log`
  is **0 bytes**. That channel is dead; do not plan around it.
- `PROFILE`'s **`Time (ms)`** column needs the pipelined runtime (Enterprise) and
  **`Page Cache Hits/Misses`** is Enterprise. Neither is available.
- `runtime=pipelined|parallel` and `operatorEngine` are Enterprise.
- Usable signals: **DB Hits, Rows vs Estimated Rows, Memory (Bytes), Indexes
  Used**, client wall time, and `SHOW TRANSACTIONS`.
- **db hits alone can mislead here**: `x IN <runtime list>` is pure CPU with
  *zero* db hits. Always read wall time beside db hits.

---

## 2. The harness (`loadtest/kh/`)

Follows `loadtest/` convention: diagnostic tooling, committed, **nothing under
`backend/` is modified to switch an arm**.

| file | what it does |
|---|---|
| `env.py` | credentials (from `backend/python/.env`, never printed), user resolution, and the **read-only guards** -- `ensure_schema()` is never called and node/rel counts are asserted unchanged at the end of every run |
| `instr.py` | patches `AsyncSession.run` and `AsyncResult.data`; a `contextvar` attributes each driver call to (request, phase, connector) across the concurrent fan-out |
| `harness.py` | builds a real `Neo4jProvider` in process and runs one global flatten, decomposed |
| `replay.py` | captures the real query, replays text-transform **arms** of it, interleaved, with identity assertion and `PROFILE` |
| `smoke.py` | reproduce the live cost and print the per-connector breakdown |
| `verify.py` | compare optimized vs reconstructed-legacy across users x sorts x filters x connectors |
| `mutate.py` | break one provider rule at a time and require the tests to go red (restores in `finally`) |

Two design points that earned their place:

- **The baseline arm is generated by the provider**, not hand-copied:
  `replay.capture()` monkeypatches `Neo4jClient.execute_query` and calls the real
  method. This is what stops arms drifting from shipped code the way
  `docs/kh-v3/*.cypher` already did.
- **Every transform asserts it matched.** A silently-inapplicable edit would
  otherwise report "1.0x, no effect", which reads like a finding.

Why the instrumentation sits on the driver and not the provider:
`Neo4jClient._run_autocommit` (`neo4j_client.py:599-601`) returns
`await result.data()` and **never calls `consume()`**, so the `ResultSummary` --
and with it `result_available_after` / `result_consumed_after` -- is discarded
before the provider can see it.

### Protocol

- Arms **interleaved in one process**, order rotated per rep. Never
  arm-A-then-arm-B (warm-up drift made sequential arms unusable in an earlier effort).
- **Per-arm warm-up**: each distinct query text gets its own plan-cache entry;
  a warm plan for one arm and a cold one for another fabricates a speedup.
- Identity asserted on **ordered** ids, not a set -- a reordering silently breaks
  keyset paging even when the set matches.
- Corpus counts checked before and after; the live stack shares this host.

---

## 3. Baseline

`PROFILE` of the real GitLab 2 query (scope + listing + tail, no enrich):

```
Total database accesses: 746,128   memory: 54,773,888   ~893 ms
returns total = 12,931  (12,907 record + 24 recordGroup)
```

Per-request decomposition of a global flatten (in-process, quiet loop):

```
request_ms 717   access_ms 9   fanout_wall_ms 708   merge_ms 5   other_ms 0
  GitLab 2      703 ms  <-- the critical path
  next largest   81 ms
```

### Finding 1 — it is one connector, not sixteen queueing

`fanout_wall_ms` (708) is within 5 ms of `max_conn_ms` (703). The fan-out is
already bounded by its single slowest query.

**This falsifies the concurrency hypothesis.** `DEFAULT_MAX_CONCURRENCY = None`
(`kh_search.py:38`) fires all 9 connectors at once, and the prior note that
"Neo4j saturates at about two effective parallel queries" suggested capping it
would help. It cannot: GitLab 2 alone *is* the request. Capping concurrency was
therefore **not** pursued. `access_ms` (9 ms) and `merge_ms` (5 ms) are also too
small to matter, and `other_ms ~ 0` confirms the decomposition is complete.

### Finding 2 — roughly half the cost is not the traversal

| operator | rows | db hits | memory |
|---|---|---|---|
| `CacheProperties` (13 properties) | 12,931 | **168,103** | |
| `Expand(All)` NODE_RELATION | 12,931 | 87,890 | |
| `Expand(Into)` INHERIT_PERMISSIONS ×3 | 0* | **59,772 each** | 816 |
| `NodeUniqueIndexSeek` Record(id) | 25,813 | 51,662 | |
| `Filter` isDeleted/isPlaceholder | 25,813 | 51,626 | |
| `Sort` | 12,931 | 0 | **45,377,376** |
| `Unwind allIds` | **25,849** | 0 | |

\* `Rows = 0` on a semi-apply's right-hand side is short-circuit reporting, not
"finds nothing" -- **16,762 of 17,004** hierarchy edges (98.6%) do have the
matching `INHERIT_PERMISSIONS` edge, so the probe mostly succeeds.

### Finding 3 — `allIds` is ~2x duplicated

`Unwind` emits **25,849** ids for **12,931** distinct nodes, and the dedup
(`WITH DISTINCT node`) happens only *after* the seeks. Each duplicate pays a
`Record` seek **and** a `RecordGroup` seek (the label union plans as a `Union` of
two `NodeUniqueIndexSeek`s), so ~4 seeks per distinct node.

### Finding 4 — cardinality is underestimated ~3,200x

`Sort` estimates **4** rows against **12,931** actual. The planner believes the
id list is tiny.

---

## 4. Ablation results (GitLab 2, 5-7 interleaved reps, median)

Control = 578-606 ms depending on the batch; each batch has its own control.

| arm | kind | median | x | result |
|---|---|---|---|---|
| control | — | 605.9 ms | 1.00x | — |
| `depth13` (`{1,50}`→`{1,13}`) | breaking | 609.1 ms | 0.99x | same |
| `depth5` | breaking | 315.8 ms | 1.92x | **same** |
| `depth1` | breaking | 306.9 ms | 1.97x | **same** |
| `no_inherit_probe_true` | breaking | 497.2 ms | 1.22x | same |
| `no_inherit_probe_false` | breaking | 303.8 ms | 1.99x | DIFFERENT (12,930) |
| `no_hidden_guard` | breaking | 586.5 ms | 1.03x | same |
| `no_notin_tests` | breaking | **1057.1 ms** | **0.57x** | same |
| `planner_dp` | preserving | 648.7 ms | 0.93x | same |
| `kill_A1_regionA` | breaking | 649.0 ms | **0.92x** | same |
| `kill_A2_declared` | breaking | 477.3 ms | 1.24x | same |
| `kill_A3_nested` | breaking | 601.5 ms | 0.99x | same |
| `kill_A4_belowDeclared` | breaking | 479.0 ms | 1.24x | same |
| `kill_A5_seeds` | breaking | 618.7 ms | 0.96x | same |
| `kill_A6_belowSeeds` | breaking | 600.6 ms | 0.99x | same |
| `kill_A7_collections` | breaking | 600.0 ms | 0.99x | same |
| `no_enrich` | breaking | 602.3 ms | 0.99x | same |
| `no_counts` | breaking | 601.9 ms | 0.99x | same |
| `no_sort` | breaking | 569.5 ms | 1.04x | DIFFERENT |

### What this says

**The seven arms are redundant with each other, and the query computes all of
them every time.** Note that *every* single-arm kill still returns 12,931:

- Kill the declared-group path (`A2` or `A4`) -> the App walk still produces all
  12,931. 1.24x.
- Cap the App walk at depth 1 -> the declared-group path still produces all
  12,931. 1.97x.

The hierarchy is genuinely 13 deep (measured: 4 nodes at depth 1, peak 4,323 at
depth 10, 35 at depth 13), so `depth1` cannot be producing 12,931 via the walk.
**Two independent arms each produce the same id set.** That is exactly the
25,849-for-12,931 duplication in Finding 3.

**Counter-intuitive results worth keeping:**

- `no_notin_tests` is **0.57x -- removing work made it slower.** The three
  `NOT sd.id IN ...` tests are an optimization: without them more nodes qualify
  as seeds and arm 6's walk does more work. This falsifies the "hidden quadratic
  in the `IN` tests" hypothesis.
- `kill_A1_regionA` is **0.92x** for the same reason -- an empty `regionA` makes
  `NOT sd.id IN regionA` admit more seeds.
- So **single-arm ablation under-reports coupled arms**, and any conclusion drawn
  from one kill in isolation would have been wrong.

**Hypotheses falsified:** concurrency (Finding 1); the `*1..20` hidden-ancestor
guard (`no_hidden_guard` 1.03x); the `{1,50}` slack bound (`depth13` 0.99x);
`IN`-list linearity; `enrich`; the exact-total counts. `planner=dp` is *slower*.

---

## 4b. What the per-node permission rule itself costs

Measured **after** the three changes landed, against the shipped query
(control 280.0 ms / 435,609 db hits, 7 interleaved reps). These are
semantics-breaking ablations — they bound what the rule could ever save, they
are not proposals.

| arm | median | x | db hits | ids |
|---|---|---|---|---|
| control (shipped) | 280.0 ms | 1.00x | 435,609 | 12,931 |
| drop the STRICT / OPEN / RESTRICTED branching, keep inheritance + grant | 251.2 ms | 1.11x | 409,747 | 12,931 |
| remove the per-hop rule entirely (structural guards only) | 226.3 ms | 1.24x | 337,043 | 12,931 |

**The whole per-node permission check is ~19% of the query (54 ms of 280 ms);
the `accessRule` branching on its own is ~10% (~29 ms).** That is the ceiling on
any rewrite of the predicate, and it is why the effort went to the listing layer
instead — where the 2x duplication and the 13-field projection were.

**Both ablations return the identical 12,931 ids, and that proves nothing about
redundancy.** It is specific to this user and connector: access is broad, and the
redundant declared-groups arm reaches the same nodes anyway. For a restricted
user the rule changes the result. The 1.24x is the rule's *cost*, not a licence
to drop it.

### The 2x2: shipped changes vs removing the rule, both against the original

Nine interleaved reps, all four arms in one process. All return the identical
12,931 ids.

| arm | median | vs original | db hits |
|---|---|---|---|
| **legacy** (the code this work started from) | 422.8 ms | 1.00x | 749,077 |
| legacy + per-hop rule removed entirely | 366.3 ms | **1.15x** | 530,967 |
| **shipped** (the three changes, rule intact) | 279.4 ms | **1.51x** | 435,609 |
| shipped + rule removed | 206.5 ms | 2.05x | 337,043 |

**Removing the per-hop rule from the original would have bought 1.15x — 56 ms.**
The three shipped changes bought 1.51x (143 ms), and unlike the rule removal
they are keepable.

**This is the sharpest example of db hits diverging from time.** Removing the
rule from legacy cuts **29% of the db hits** (218,110) but only **13% of the
time** — the work it removes is index-backed `EXISTS` probes, cheap per hit. A
db-hits-only view ranks the rule about twice as important as it is. Read both.

**The unshippable ceiling is 2.05x**, which confirms from a second direction
what section 4c says: what is left is the traversal *structure* — deriving
12,931 nodes on every page — not any predicate inside it.

The rule was the obvious suspect and was never the lever. The listing-layer
duplication, which no part of the initial plan pointed at, was worth more than
twice as much and could actually ship.

### The inheritance probe, re-measured after the factoring

| arm | median | x | ids |
|---|---|---|---|
| control | 266.9 ms | 1.00x | 12,931 |
| both probes -> `true` | 250.8 ms | **1.06x** | 12,931 |
| both probes -> `false` | 163.4 ms | 1.63x | **12,930 — different** |

Against the *unfactored* query the same arm measured **1.22x**. It is 1.06x now,
which is the direct confirmation that section 5.3 banked the duplicate
evaluation rather than merely moving it.

Worth noting how this surfaced: after the factoring these arms **skipped
loudly**, because they expected to find the probe text twice and there was only
one. That is the "every transform asserts it matched" rule earning its keep —
without it they would have reported a silent "1.0x, no effect".

---

## 4c. Synthesis — where the compute actually goes

### The request is one query

Of a 717 ms request: `access_ms` 9, `merge_ms` 5, `other_ms` ~0, and
`fanout_wall_ms` 708 — within 5 ms of the single slowest connector. GitLab 2
**is** the request; the other eight connectors run inside its shadow.

### The work is per-visible-node, not per-page

The query derives all **12,931** visible nodes to return **50** — a **259x
amplification**, and the root cause of everything below it.

| evidence | result | what it says |
|---|---|---|
| page 1 / page 2 / page 5, by cursor | 271.4 / 281.3 / 264.8 ms | **paging does not reduce work** |
| `no_enrich` (touches only the 50 returned rows) | 0.99x | per-page work is free |
| `no_counts` / `no_sort` | 0.99x / 1.04x | even the exact-total tail is nearly free |

Every expensive operator scales with the visible set; nothing scales with page
size. This is the single most important structural fact about the query, and it
is why the initial suspicion of the `Sort`/`collect` tail was wrong.

### The db-hit budget (current, 435,609)

| layer | db hits | share |
|---|---|---|
| Hierarchy walk + rule — `Expand(All)` NODE_RELATION 87,890, `Expand(Into)` INHERIT_PERMISSIONS 59,772, two rule filters 51,724 | ~199,400 | **46%** |
| Listing — seek each id, filter, build `slim` | ~129,200 | **30%** |
| Declared-groups path — arm 4 `BELONGS_TO` 13,002 + filter 25,812 | ~38,800 | 9% |
| everything else | ~68,200 | 15% |

### How the bottleneck moved

| | before | after |
|---|---|---|
| Listing layer | ~336k (45%) | ~129k (30%) |
| Scope / walk | ~410k (55%) | ~306k (70%) |
| **total** | **746,128** | **435,609** |

The three changes took ~62% out of the listing layer. **The bottleneck is now
decisively the traversal**, and it has a hard floor: ablating the entire
permission rule — unshippable — buys only 1.24x (section 4b).

So the remaining large win is **structural, not a query rewrite**: the visible
set is recomputed from scratch on every page request. Materialised ancestry or a
cached per-(user, connector) visible set attack that; predicate cleverness
cannot, and is capped at ~19%.

---

## 4d. The endpoint itself (HTTP), and a correction

`loadtest/kh/http_probe.py`, 100 calls, 0.5 s between them, one
`requests.Session` (connection reuse, like a real client), against the shipped
code. `GET /api/v1/knowledge-hub/nodes?flattened=true&limit=50`.

| | ms |
|---|---|
| min | 293.8 |
| **p50** | **320.5** |
| **p80** | **339.1** |
| p90 | 351.4 |
| **p95** | **369.4** |
| p99 | 406.8 |
| max | 430.0 |
| mean / stdev | 325.2 / 24.2 |

100/100 succeeded, `totalItems: 14910`, 50 rows — the same total the in-process
harness reports, which confirms both levels measure the same request.

The distribution is **tight**: stdev 24 ms, p95 only 15% above p50, no tail
pathology. The 3x swings seen elsewhere in this document were machine load, not
the query.

### Correction: the HTTP layer costs ~35 ms, not ~204 ms

An earlier measurement in this session used `curl` in a shell loop and put the
endpoint at 541 ms, implying a ~204 ms HTTP layer. **That was a measurement
artifact** — curl spawned a process and opened a fresh TCP connection per call.
With a session and matched pacing:

| | in-process | HTTP | HTTP layer |
|---|---|---|---|
| p50 | 285.9 | 320.5 | **+34.6** |
| p80 | 298.2 | 339.1 | +40.9 |
| p95 | 329.0 | 369.4 | +40.4 |

The conclusion drawn from the bad number — "even reducing the Cypher to zero
only reaches ~204 ms, so query work has little headroom at the API level" — was
**wrong and is retracted**. The real split at p50:

```
320 ms  API call
├─ 286 ms  in-process request          (89%)
│   ├─ ~275 ms  the dominant connector's query
│   ├─    9 ms  access resolution
│   └─    5 ms  merge
└─  35 ms  FastAPI + auth + serialisation  (11%)
```

**~89% of API latency is the Cypher**, so the remaining query work (the
traversal at 46% of db hits, and the ~76k db hits of redundant re-seeking)
does pay through to the endpoint.

The lesson is the one this harness already encodes elsewhere and that I broke
here: *the client is part of the measurement*. A per-call process spawn and TCP
handshake was 2/3 of a number I then reasoned from.

### Still open

Legacy has **not** been measured over HTTP. The connectors service runs without
`--reload`, so a true before/after at the endpoint needs the provider reverted
**and the service restarted** — an interruption to a running stack, so it was
not done unasked.

---

## 5. Landed changes

All three are semantics-preserving, verified by byte-identical ordered page ids,
`total` and per-type counts.

| arm | median | x | db hits |
|---|---|---|---|
| control | 578.1 ms | 1.00x | 749,077 |
| `dedup_allids` | 512.8 ms | 1.13x | — |
| `slim_min` | 527.4 ms | 1.10x | — |
| `dedup_slim` | 418.6 ms | 1.38x | 555,153 |
| `hop_factored` | 523.2 ms | 1.10x | 629,533 |
| **`dedup_slim_factored`** | **379.3 ms** | **1.52x** | **435,609** |

### 5.1 Dedup `allIds` before the seeks

`neo4j_provider.py`, `{listing}`. Was `UNWIND allIds AS vid`; now dedups into
`dedupedIds` first. The arms overlap by construction, so this removes ~12,900
redundant double index-seeks plus their property reads.

### 5.2 Build only the `slim` fields something reads

`neo4j_provider.py`, the `slim` projection. It built **13 fields for every
visible node** (12,931 of them) to return 50. Downstream, `{tail}` collects only
`id`/`nodeType`/`sortKey`, the sort reads the sort property, and `{enrich}`
re-MATCHes the page rows and rebuilds the full projection from the real node --
so the other nine fields existed only for the filters.

The needed set is now **scanned out of the generated filter text**
(`re.findall(r"node\.(\w+)", conditions)`) rather than hardcoded, so adding a
filter cannot silently drop its field. `CacheProperties` fell 168,103 -> 38,793
db hits and `Sort` memory 45.4 MB -> 17.3 MB.

### 5.3 Name the inheritance probe once

`_kh_v3_granted_hop`. The rule spelled
`EXISTS { (c)-[:INHERIT_PERMISSIONS]->(p) }` **twice** -- once in the RESTRICTED
arm, once in the STRICT/OPEN arm -- and the planner emitted two `Expand(Into)`
operators at **59,772 db hits each**.

With `R` = accessRule, `I` = probe, `G` = grant test:

```
was:  (R='RESTRICTED' AND I AND G) OR (R IN ['STRICT','OPEN'] AND (I OR G))
now:  (I AND (R IN ['STRICT','OPEN'] OR (R='RESTRICTED' AND G)))
      OR (R IN ['STRICT','OPEN'] AND G)
```

Equal for all three declared values **and** both yield `false` for an undeclared
one. That last clause is the point: decision 77 records that the obvious
three-conjunct compaction **fails open** on an undeclared `accessRule`. This
factors the probe out without touching the three-branch structure that keeps it
failing closed.

### Final numbers

**Per-connector A/B** (GitLab 2, both arms interleaved in one process, 9 reps
after 3 per-arm warm-ups, quiet machine). The `legacy` arm is rebuilt from the
current provider output by inverting the three edits, so both arms come from the
same live code:

| arm | median | min | db hits | result |
|---|---|---|---|---|
| `legacy` | 763.8 ms | 730.2 ms | 749,077 | — |
| **optimized** | **505.3 ms** | **466.9 ms** | **435,609** | identical |

**1.51x, 42% fewer db hits.**

**End-to-end request**, measured back-to-back by reverting the provider to HEAD,
measuring, restoring, and measuring again. Run twice, on a busy machine and
again on a quiet one:

| | legacy median | optimized median | ratio |
|---|---|---|---|
| busy (8 reps each) | 824.7 ms | 565.5 ms | 1.46x |
| quiet (10 reps each) | 471.0 ms | 316.4 ms | 1.49x |

`total = 14,910` and `ids_digest = 9358f981c9b41ece` unchanged throughout.

**Absolute medians drift by a factor of ~2 with machine state; the ratio does
not.** The same control measured 578 ms, 505 ms and 259 ms in three different
batches. Only the **ratio within a batch** is meaningful, which is why every
batch carries its own interleaved control — and why **db hits (749,077 ->
435,609, -42%) are the more durable number**, since they do not move with load
at all.

Across all batches: **per-connector 1.51x - 1.60x, end-to-end 1.46x - 1.49x.**

The first request after the change is ~1,100 ms: the query text changed, so the
plan cache misses once. That is a replan, not a regression.

---

## 6. What remains, and what was deliberately not done

Remaining hot operators after the three changes (435,609 db hits):

| operator | db hits |
|---|---|
| `Expand(All)` NODE_RELATION | 87,890 |
| `Expand(Into)` INHERIT_PERMISSIONS | 59,772 |
| `CacheProperties` | 38,793 |
| `Projection` (slim) | 38,745 |

The `Expand(All)` is the walk itself and is not reducible without a structural
change.

**The one candidate left with a real number on it: stop re-seeking nodes the
arms already hold.** Every arm ends in `collect(DISTINCT x.id)`, and the listing
then looks each id back up with
`MATCH (node:Record|RecordGroup {id: vid})` — which plans as a `Union` of **two**
`NodeUniqueIndexSeek`s, one of which always returns zero rows. Measured after
the dedup, that is `NodeUniqueIndexSeek` **25,838** + its `Filter` **25,814** +
the RecordGroup half ≈ **76k db hits, ~17% of what remains**. Collecting node
references alongside the id lists (the ids are still needed for the `IN` tests)
would remove it.

Not attempted here because it touches all seven arms plus the listing, and a
text-level transform across that many sites is exactly where a silent
result-set change would hide. It wants a real code change with `verify.py` run
across the full matrix afterwards, not an arm.

**Not done, and why:**

- **Anchoring the seed walk on a seek** (wrap arm 6 in
  `CALL (seedIds) { UNWIND seedIds AS sid MATCH (sc {id: sid}) ... }` so the
  walk starts at the seeds instead of filtering `sc.id IN seedIds` after the
  pattern). This was the single most-recommended rewrite going in. Measured
  **0.97x with byte-identical db hits (435,609)** -- the planner already pushes
  that filter below the `Repeat`, so there was nothing to win. Falsified.
- **Capping fan-out concurrency** -- falsified by Finding 1.
- **Lowering the `{1,50}` bound** -- `depth13` measured 0.99x.
- **Removing the hidden-ancestor guard** -- 1.03x, and it is a P0 over-share.
- **`planner=dp`** -- measured *slower* (0.93x).
- **Dropping either redundant arm** -- both are needed for correctness on other
  tenants; only this user/connector has both paths covering the same set.

**Open for a decision (not taken unilaterally):**

- The exact total forces full materialisation of the connector on page 1
  (`Sort` + `EagerAggregation`). A cheaper or approximate total for very large
  connectors would remove the largest remaining memory cost.
- Re-checking the winning changes on **5.26** (what `install.sh` and Helm ship)
  before release. This phase was scoped to 2025.12.1 only.

---

## 7. Correctness verification

### Semantic verification on real data — 1,782 cells, 0 failures

`loadtest/kh/verify.py` rebuilds the **pre-change** query from whatever the
provider generates now (reverting the three edits at text level) and runs both
against the same parameters, comparing ordered page ids, `total` and per-type
counts. Both arms come from live provider output, so this keeps working as the
surrounding query evolves — unlike a checked-in copy of the old Cypher, which is
exactly how `docs/kh-v3/*.cypher` drifted.

6 users x 6 sorts (`name` ASC/DESC, `updatedAt`, `createdAt`, `sizeInBytes`,
`nodeType`) x **9 filter cases** x every gated connector = **1,782 cells, 0
differences**.

The filter dimension is the one that matters most. `slim` now builds only the
fields something downstream reads, and *which* fields those are is derived from
the generated filter text — so a filter whose field went missing would read as
null and match nothing, **silently**. The nine cases (`search_query`,
`record_types`, `indexing_status`, `origins`, `updated_at`, `size`,
`node_types`, `only_containers`, and none) each name a different `slim` field.

The provider also falls back to the full projection if a filter ever names a
field the map cannot build — slower, but it cannot lose rows.

### Mutation testing — 7/7 caught

The new v3 modules passed on their first run, which is weak evidence. So
`loadtest/kh/mutate.py` breaks one provider rule at a time and requires the
suite to go red for each. It restores the provider in a `finally`.

| mutation | sites | caught |
|---|---|---|
| arm 2 ignores `hideChildren` | 3 | yes |
| arm 2 admits any declaration (drops the `IN regionA OR granted` gate) | 2 | yes |
| arm 4 ignores `declaredScope` | 2 | yes |
| arm 5 drops the grant test | 2 | yes |
| **the hop's naive compaction (decision 77's fail-open)** | 1 | yes |
| **RESTRICTED accepts inheritance alone (AC-16)** | 1 | yes |
| the hop ignores inheritance and grant | 1 | yes |

**The first run scored 1/5 — and the four "survivors" were a harness bug, not a
test weakness.** `str.replace(old, new, 1)` edits the *first* occurrence, and
for four of these five rules the first occurrence is inside
`get_knowledge_hub_visible_set_v3` (`:14942`), which **nothing calls**. A
mutation applied to dead code changes nothing the tests can see, so it reads
exactly like a passing suite. `mutate.py` now replaces every occurrence and
prints the site count — and those counts (3, 2, 2, 2, 1, 1, 1) are themselves
the measure of how much duplicated rule text the provider carries.

**A second near-miss, in my own test.** `test_an_absent_rule_reads_as_open` was
vacuous as first written: `rec()` always writes `accessRule`, defaulting to
`"OPEN"`, so the "absent" rows were ordinary OPEN rows and the test asserted
nothing beyond the OPEN case it already had. Fixed by popping the property, and
`test_the_absent_case_really_omits_the_property` now guards that premise — the
same trick as `test_the_fixture_can_exercise_the_declared_arms`. Both exist
because a test whose *setup* is wrong passes loudly and proves nothing.

### Test suites

- **KH unit tests** — all pass (2 pre-existing xfails). These assert on query
  *source text*, so they were the most likely to break; they did not.
- **`tests/integration/graph_permissions/`, neo4j** — 260 collected tests
  (including the 18 new ones). The failure set with my changes is **identical
  to the failure set on pristine HEAD**: the same 5 tests in
  `test_provider_v2_search.py`. Verified by reverting the provider to `HEAD`,
  re-running, and restoring. So they are pre-existing, from the v3 rewrite in
  `ba8d4e6e8`, not from this work. The full suite was run to completion **four
  times** over the course of this work, with the same result each time.

  **One gap, stated plainly:** the last two tests added to
  `test_provider_v3_scope_arms.py` (deleted-node coverage and the cross-org
  xfail) were verified **at module level only** — 19 passed, 1 xfailed. The
  whole-suite run that would have covered them was killed part-way by the
  machine running out of memory, and was not restarted. Re-run
  `pytest tests/integration/graph_permissions/ -k "not arango"` to close it;
  nothing about those two tests depends on the rest of the suite, so the
  expected result is the same 5 pre-existing failures.
- **arango** — `test_provider_v2_deleted` and others fail with
  `'ArangoHTTPProvider' object has no attribute 'get_knowledge_hub_access_v3'`.
  `kh_search` is v3-only now and Arango has no v3 implementation. Pre-existing
  and by design of the current branch state.

---

## 8. Pre-existing v3 gaps found while running the suite

These are **not** caused by the optimization. They were found because the suite
had to be run as a gate, and they matter more than the latency work.

### 8.1 The acceptance fixture cannot exercise five of v3's seven scope arms

v3 keys arms 2-6 on `connectorId = app.id`:

- arm 2 `WHERE dg.connectorId = app.id` -> `declaredIds`
- arm 3 nested groups, driven by `declaredIds`
- arm 4 `belowDeclared`, driven by `declaredScope`
- arm 5 `WHERE sd.connectorId = app.id` -> `seedIds`
- arm 6 `belowSeeds`, driven by `seedIds`

In `fixture_graph.py`:

- **`rg()` never sets `connectorId` at all** (`fixture_graph.py:60-66` sets
  `connectorName`, not `connectorId`). So `RecordGroup.connectorId` is NULL for
  every group in the fixture and arm 2 can never match.
- **Records use a connector id that is not an app id**: `a, conn = "pl-app",
  "drive-conn"` (and the same for `gp-app`, `swm-app`, `gate-app`). So arm 5
  can never match either.

On the **real store this convention does not hold** — checked directly:

```
Record      : connectorId = App id   16,445  (100%)
RecordGroup : connectorId = App id      516  (100%)
```

So production always satisfies `connectorId = app.id`, and the fixture never
does. The consequence is that **arms 2-6 are dead for every `drive-conn` app in
the entire integration suite** — only arm 1 (the App walk) and arm 7
(collections) are exercised. The suite still reports green because the App walk
covers most fixture cases, which is precisely why this was invisible.

This is the most important thing in this document: "423 tests green" is not
evidence that v3's scope block works, because most of it never runs.

**What was done about it.** Rewriting the shared fixture would touch a file
400+ tests assert exact id sets against, and each app block would need per-node
ownership (the declarations block alone has two apps sharing one `conn`). So
instead a new, additive module was added:

**`backend/python/tests/integration/graph_permissions/test_provider_v3_scope_arms.py`**

It builds its own small graph with the production convention
(`connectorId = app.id` on every node) so arms 2-6 actually run, and covers:

| test | arm |
|---|---|
| a declared group's contents reached by `BELONGS_TO` | 2 + 4 |
| a nested group carries the declaration down | 3 + 4 |
| a granted node below a gap is visible | 5 |
| content below a seed is visible | 6 |
| the gap itself is not admissible | 1 |
| an undeclared group's contents stay out | — |
| a declaration granted to nobody opens nothing | 2 |
| `hideChildren` hides contents but not the group | 2 |
| the exact visible set, and that nothing hidden leaks | all |
| `total` counts the whole result, not the page | PG-32 |
| a user with no grants gets an empty page | gate |

`rg(**props)` already forwards to `_node`, so `connectorId` could be set without
modifying the shared fixture at all.

The module also carries **`test_the_fixture_can_exercise_the_declared_arms`**,
which asserts every node in *its own* graph satisfies `connectorId = app.id`.
Without that guard this module could rot into exactly the hole it was written to
escape — passing while testing nothing.

Assertions are against **known values**, not against the other backend: two
engines generated from one set of builders share that set's mistakes.

### 8.1b Nothing gated the v3 hop rule

`test_rule_equivalence.py` pins **`_kh_v2_rule_cypher`** against an oracle. It
does not touch **`_kh_v3_granted_hop`** — the rule the shipped global-flatten
query actually evaluates, and the one section 5.3 rewrote.

That is the exact situation decision 77 warns about: the obvious compaction of
this rule agrees on every *declared* `accessRule` and **fails open on an
undeclared one**, so no amount of scenario coverage separates the two forms.

Added: **`test_provider_v3_hop_rule.py`**, which enumerates the rule's inputs
rather than sampling scenarios —

```
accessRule in {OPEN, STRICT, RESTRICTED, <absent>, <unrecognised>}
  x inherits in {yes, no}  x  granted in {yes, no}   =  20 children
```

— and asserts which the walk admits, with the load-bearing cases called out
separately: AC-16 (RESTRICTED refuses a grant without inheritance), corruption
fails closed, and **absent is not the same as unrecognised** (absent reads as
OPEN; unrecognised hides).

Worth recording: the real store is **not** all-OPEN — it holds 16,347 OPEN,
**564 STRICT and 50 RESTRICTED** nodes, so `verify.py` exercises all three
branches on real data too.

Fixing the shared fixture is still worth doing and is left as a recommendation —
it would turn arms 2-6 on for the other 400+ tests too, and only then would
8.3 below tell us anything.

### 8.1c Cross-org over-share: the tenant boundary rests on one predicate

Found by writing a test for it, on the newly-live arms.

The v3 scope block gates the **App** on `app.orgId = $org_id` and then **never
re-checks `orgId` on any node it collects**. Arms 2 and 5 key on
`connectorId = app.id` alone, and the walk arms key on reachability alone. So a
node carrying this connector's id, a direct grant, and **another org's
`orgId`** is returned in the wrong tenant's page. Demonstrated:
`test_a_node_from_another_org_does_not_leak` plants exactly that node and it
comes back.

**Latent, not live.** On the dev store every one of 16,445 Records and 516
RecordGroups has `orgId` equal to its App's `orgId`, and there is a single
`Organization`. So nothing is leaking today — but the boundary is resting
entirely on the sync never writing a mismatched `orgId`, with no defence in the
read path. It is the same family as the recorded org-isolation issues
(org-blind lookups).

**Not fixed here.** Adding an `orgId` predicate to the arms changes what a node
with a **NULL** `orgId` does, which is a permission-model decision rather than
a cleanup — the kind that goes to you, not into a performance branch. The test
is marked `xfail(strict=True)`, so it stays visible, stays green, and **fails
loudly if someone fixes the query** and forgets to remove the marker.

Deleted-node handling, tested at the same time on all four admitting positions
(declared contents, nested group, nested group's contents, seed), is **correct**
— every arm carries its own `isDeleted` guard and so does the listing.

### 8.2 v3 global search never returns App rows (D45)

`test_a_global_search_returns_each_node_once` asserts
`"pl-app" in placed and placed["pl-app"] is None, "Apps are a partition too (D45)"`.
v3's `{listing}` matches `(node:Record|RecordGroup {id: vid})` only, and the page
query's `allIds` does not include `app.id` — note the standalone
`get_knowledge_hub_visible_set_v3` **does** prepend `[app.id]`, so the two
disagree. Consequences:

- `counts_by_type["app"]` raises `KeyError` (`test_the_total_and_counts_...`).
- `test_a_filter_applies_in_every_partition` collapses to a single expected row.

Whether v3 *should* return Apps is a design call (D45 says yes), so this is
reported, not fixed.

### 8.3 `pl-r6` — a granted node below a gap — is missing from search

Expected `{pl-r6, kb-r3, kb-r4}`, got `{kb-r3, kb-r4}`. `pl-r6` is
"Granted below gap": `App -> pl-rg1 -> pl-r3 (gap) -> pl-r6` with a direct grant.
It is a seed, and seeds are arm 5 — which 8.1 shows cannot fire in this fixture.
So this failure is a *symptom* of 8.1, not an independent bug. It would be
resolved by fixing the fixture, and only then would it tell us whether v3's seed
handling is actually correct.

---

## 9. How to reproduce any of this

Harness usage is in `loadtest/kh/README.md`. The specific commands behind the
numbers above:

```bash
# Section 3 -- the per-connector breakdown that showed it is one connector
python loadtest/kh/smoke.py --email <you> --reps 5 --quiet

# Section 4 -- the ablation matrix (breaking arms, diagnostic only)
python loadtest/kh/replay.py --email <you> --reps 5 --warmup 2 \
  --arms control,depth13,depth5,depth1,no_inherit_probe_true,\
no_inherit_probe_false,no_hidden_guard,no_notin_tests,planner_dp
python loadtest/kh/replay.py --email <you> --reps 5 --warmup 2 \
  --arms control,kill_A1_regionA,kill_A2_declared,kill_A3_nested,\
kill_A4_belowDeclared,kill_A5_seeds,kill_A6_belowSeeds,kill_A7_collections,\
no_enrich,no_counts,no_sort

# Section 5 -- the landed changes, A/B against the reconstructed old query
python loadtest/kh/replay.py --email <you> --reps 9 --warmup 3 --profile \
  --arms control,legacy,anchor_belowseeds --out loadtest/kh/results/final

# Section 7 -- semantics, across users x sorts x filters x connectors
python loadtest/kh/verify.py

# Section 7 -- mutation testing (edits the provider, restores in `finally`)
python loadtest/kh/mutate.py

# Section 8 -- the new v3 scope-arm tests
cd backend/python && python -m pytest \
  tests/integration/graph_permissions/test_provider_v3_scope_arms.py -q
```

The integration harness needs its throwaway containers up:

```bash
docker compose -p kh-perm \
  -f backend/python/tests/integration/graph_permissions/docker-compose.yml up -d
```

Two things that will bite:

- **`cd` into `backend/python` in the same command as pytest.** A `cd` earlier
  in the shell leaves the working directory at the repo root, `tests/...`
  collects **nothing**, and pytest still exits 0 — a "0 tests" run that reads as
  success.
- **Restart `kh-perm-neo4j-1` before any long run.** It degrades badly as it
  approaches its 2 GiB cap while still reporting `healthy`.

---

## 10. Incidental findings (not perf)

- `docker-compose.yml` advertises `KH_VISIBLE_SET_V3` / `KH_CHILDREN_V3` as
  working switches; **nothing reads them**. The comments describe behaviour that
  no longer exists.
- `docker-compose.yml:503-505` sets `NEO4J_dbms_memory_heap_*` and
  `NEO4J_dbms_memory_pagecache_size`. Those are Neo4j **4.x** setting names; the
  5.x/2025.x names are `server.memory.*`. Worth confirming they are not being
  silently ignored in the shipped stack.
- `knowledge_hub_service.py:487` calls the **v2** `get_knowledge_hub_access_context_v2`
  on every request on top of v3's own access resolution. Measured at only ~9 ms
  here, so it was not touched, but it is a redundant round trip.
- The query names `sharingStatus`, which **does not exist** anywhere in the
  database (server notification `01N52` on every call).
- **`only_containers` silently drops every leaf container.** The filter tests
  `node.hasChildren`, but `slim` hardcodes `hasChildren: false` for every row —
  the real value is only computed later, in `{enrich}`, for the page. So the
  filter can only ever pass nodes whose `nodeType` is app/recordGroup/folder,
  and a container that is a *record* never survives it. Pre-existing; not
  touched, because fixing it means computing `hasChildren` for every visible
  node, which is the opposite of what section 5.2 just did.
- **`get_knowledge_hub_visible_set_v3` (`:14942`) is dead code** and a
  near-duplicate of the live `{scope}` block. Nothing calls it (the flags that
  would have are dead too). It did **not** receive the three optimizations, so
  the two copies have now diverged further. It is also a trap for tooling: a
  mutation harness that replaces the *first* occurrence of a rule string
  silently edits this copy instead of the live one and reports a false
  "mutation survived". `loadtest/kh/mutate.py` mutates every occurrence and
  prints the site count for exactly this reason.
- `_walk_pages` (`knowledge_hub_service.py:139`) replays every prior page
  serially when given a page number, so page 2 by `?page=2` costs page 1 + page
  2. Any paging measurement must use the **cursor**.
