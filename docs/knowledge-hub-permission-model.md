# Knowledge Hub — New Permission Model and Traversal

**Status:** requirements and discussion. Only what appears in the [Decisions log](#8-decisions-log)
is agreed.
**Sources:** Confluence page "New Permission Model" (export of 2026-09-11); requirements
discussion on 2026-09-11 and 2026-09-12; code read of `main` and of the prior attempt.
**Test cases:** [`knowledge-hub-permission-model-test-cases.md`](knowledge-hub-permission-model-test-cases.md),
maintained alongside this doc (decision 23).
**Prior attempt:** PR #1522 and the local `refactor/permission-hierarchy` branch — not reused as
code; see [§5](#5-prior-attempt--what-to-carry-what-to-change).
**PR #3230** (retrieval "search with filters") is a separate effort. The one thing shared is the
`permissionModel` property it adds to App and RecordGroup, which this work reads —
[§3.8](#38-permission-model-declarations), decisions 31 and 39.

---

## 1. Problem

The current permission model and knowledge hub queries cannot express:

- a record shared on its own, when the user has no access to its parent;
- connectors where every ancestor must be accessible (strict), next to connectors where it need not be.

Serving both means parent permission checks have to be **optional** — applied where the source
demands them, skipped where it does not.

## 2. Requirements

### 2.1 First goal — knowledge hub queries

| ID | Requirement | Source |
|---|---|---|
| R1 | Knowledge hub browse, search and filter traverse the graph with the rules in [§3](#3-permission-model-and-traversal). | Doc §1.1; discussion |
| R2 | Parent permission is checked per node, and only where the node requires it (strict). Strictness is per node, mixable within a connector, and non-strict by default. | Decisions 1, 17, 24, 38 |
| R3 | Breadcrumbs come from the same single query as the listing, and follow the navigation the knowledge hub shows — not the physical path. | Decisions 5, 58 |
| R4 | Every node in results carries its parent id — the parent it appears under in navigation, never a node the user cannot access. | Decision 14 |
| R5 | Knowledge base items carry the user's role, taken from explicit grants on the collection. Connector items carry no role. | Decisions 9, 21, 34, 52 |
| R6 | Cursor-based pagination with nearly the same structure as the prior design; the agreed contract is [§3.9](#39-agreed-api-and-pagination-contract). | Decisions 8, 58 |
| R7 | Parallel queries per **record group that is a direct child of the App**, and one per knowledge base — not per App/connector — merged into one sorted page. | Decision 7 |
| R8 | The permission model and traversal queries are written from scratch; from the prior attempt only the QPP algorithm carries over. | Discussion |
| R9 | Neo4j and Arango both support the algorithm, before anything else is built on it. | Decision 11 |
| R10 | New `v2` query functions beside the current ones. The service calls v2 in code; switching back is a code change. Later v1 is removed and v2 takes v1's name. | Decision 20 |
| R11 | Every frontend feature that uses the knowledge hub API moves to the new contract. | Decisions 10, 22 |
| R12 | The new write path applies to every connector from the start. Confluence Cloud, SharePoint Online and GitLab get focused testing and any connector-specific changes; other connectors keep working as before. No special-casing; production only after every connector is tested. | Decisions 18, 19 |
| R13 | Remove the ORGANIZER, COMMENTER, FILEORGANIZER and OTHERS roles from the whole code base; a commenter is written as READER. | Decision 41 |
| R14 | A test-case document, listing normal and edge cases, is maintained from now on. | Decision 23 |
| R15 | v2 has to be correct on data written by the new write path; older data is the migration's job. | Decision 32 |
| R16 | Keep the logic simple: only the conditions that are actually needed go into the query. | Decision 39 |
| R17 | Remove the DOMAIN, ANYONE and ANYONE_WITH_LINK grant types from the whole code base; organisation-wide sharing is an org grant. | Decision 59 |

### 2.2 Also in the Confluence doc — outside the first goal unless agreed

Numbered CD1–CD4 so they do not clash with decision numbers (D1–D58).

| ID | Item | Status |
|---|---|---|
| CD1 | Simple permission for GitLab, GitHub, Bitbucket. | **Decided** — this is the `permissionModel` declaration, §3.8. |
| CD2 | One edge type for the App → RecordGroup → Record hierarchy. | **Decided** — `NODE_RELATION`, decisions 15, 30, 56. |
| CD3 | RecordGroup / folder / record level filters in chat. | After the first goal (decision 12). |
| CD4 | Rework of the other queries: connector deletion, full-sync edge deletion, connector stats, collection CRUD, reindex and indexing-status, get record by path, data source entities methods. | After the first goal; CD2 makes much of it necessary. |

## 3. Permission model and traversal

### 3.1 Terms

- *node* — App, record group or record (a folder is a record).
- *direct grant* — a permission on the node that reaches the user, through any of the five paths:
  user, group, role, team, org (decision 42).
- *inherits* — the node has an inheritance link to its direct parent. It is a per-node fact: a
  node may have a parent and not inherit from it (decision 25).
- *access rule* — `accessRule` on the node, one of three values. Set per node, mixable within a
  connector (decisions 1, 17), and **`OPEN` unless the connector says otherwise** (decision 38).
  It applies to the node it is on and is never inherited by its children (decision 37).
  - `OPEN` — parents do not matter; inheritance from an accessible parent, or a direct grant, is
    enough.
  - `STRICT` — **every** ancestor up to the App must be accessible as well (decision 24).
  - `RESTRICTED` — implies strict, and inheritance alone is not sufficient: the node's own grant is
    required on top of it.
  Two booleans (`isStrict`/`hasRestriction`) previously encoded this. Restriction was only ever read
  on a strict node (decision 3), so the fourth combination was inert; it maps to `OPEN`, and the
  three-valued field makes it unrepresentable.
- *gap* — an ancestor the user cannot access, with an accessible node somewhere below it.

### 3.2 Rules (agreed)

| Node | Accessible when |
|---|---|
| `STRICT` | every ancestor up to the App is accessible, **and** the node inherits or holds a direct grant |
| `RESTRICTED` | every ancestor is accessible, **and** the node inherits **and** holds a direct grant |
| `OPEN` | holds a direct grant — parents do not matter — **or** inherits from an accessible parent |
| under a permission-model declaration | reaching the App or the record group is enough; nothing below it is checked — §3.8 |
| inside a collection (knowledge base) | the user holds an explicit grant on the collection — a direct user grant or a team share (decision 52). Everything in it is then visible and carries that role. |
| App (connector) | the connector gate: any permission that reaches the user, directly or through another entity, or a user–app relation (decision 43) |

- A direct grant without access to the parents does not count for a strict node (decision 2).
- Neither inheritance nor a grant means no access, whatever the node's flags (decision 25).
- Records with `isDeleted = true` are never returned (decision 54).

### 3.3 Where nodes below a gap appear (agreed)

A node the user can access, whose parent the user cannot access, is the top of a
continuous-permission chain.

- A **record** appears under **its own record group** if the user can access that group; otherwise
  directly under the **App** (decision 13). Its own group, not the nearest accessible group above it.
  The App fallback applies only when that group is under the App and cannot be opened. It never
  applies beneath a `hideChildren` group or to a deleted group (decision 78).
- A **record group** follows the same rule: its parent group if the user can access it, otherwise
  the App (decision 27).
- A node whose parent is **not in the graph at all** falls back the same way: its record group, or
  the App when it has none (decision 53).
- **Drive "Shared with Me"** is a real second hierarchy edge: a shared record appears in **both**
  places when the user can reach both — its drive location and the Shared-with-Me group
  (decisions 29, 55). A flat result returns it **once**, under the drive location when both are
  reachable and otherwise under whichever parent is (decision 67).
- Its accessible descendants appear under it as usual.
- Inaccessible nodes are never shown (decision 4). Placeholder stubs **are** shown, so the
  hierarchy stays intact (decision 44).
- `parentId` is the node it appears under (decision 14); breadcrumbs follow the same placement
  (decision 5).

Worked cases:

| Graph (✓ accessible, ✗ not) | Appears under | Breadcrumbs |
|---|---|---|
| Example 2: App → RG 1 ✓ → record 3 ✗ → record 6 (grant) | RG 1 (its group) | App › RG 1 › 6 |
| App → RG 1 ✓ → RG 2 ✗ → record 6 (grant, belongs to RG 2) | App | App › 6 |
| App → RG 1 ✓ → record 6 ✓ → record 10 ✗ → record 11 (grant, belongs to RG 1) | RG 1 — not record 6 | App › RG 1 › 11 |
| App → RG 1 ✓ → RG 2 ✗ → RG 3 (grant) | App (RG 3's parent group is RG 2) | App › RG 3 |

### 3.4 Traversal steps

1. Start at the current node: the connector root, or the folder/page/space the user selected.
2. **Connector gate** — if the subtree belongs to a connector the user has no access to, stop.
3. **Check the current node** against §3.2.
4. If it passes, collect descendants — *browse*: direct children only; *flatten*: every node in the
   subtree, listed flat (decision 28).
5. Going down, apply §3.2 to each child against its direct parent. Pass → include it (and
   recurse when flattening). Fail → skip the child **and everything under it**.
6. Apply search filters, sorting and pagination only to nodes that passed step 5.

Steps 4–5 prune top-down, so on their own they never reach a node below a gap. The second way in
is the user's own grants — §3.6, arm 2.

### 3.5 Worked examples (Doc §3)

**Example 1 — Confluence: every node needs all of its parents.**
App → RG 1, RG 2 (restricted, direct grant). RG 1 → record 3, record 4 (restricted, no direct
grant). Record 4 → record 6. RG 2 → record 5. Every node inherits from its parent; the user has
app access.

| Accessible | Not accessible |
|---|---|
| 1, 2, 3, 5 | 4 (restricted, no direct grant), 6 (under 4) |

**Example 2 — SharePoint: parent permission not required.**
App → RG 1 (site), RG 2 (site). RG 1 → record 3, record 4. Record 3 → record 6.
Record 6 → records 7, 8. RG 2 → record 5.

| Node | Result | Why |
|---|---|---|
| 1, 2 | accessible | inherited |
| 4 | accessible | direct grant, parent accessible |
| 6 | accessible | direct grant, **parent 3 not accessible** |
| 5, 7, 8 | accessible | inherited, parent accessible |
| 3 | not accessible | no direct grant, no inheritance |

**Example 3 — permission only at record group level** (GitHub, GitLab, Bitbucket). This is the
`permissionModel` declaration — §3.8.

### 3.6 Strictness: reading (b), and how it runs in one query

Decision 24 picks reading **(b)**: a strict node needs **every** ancestor up to the App to be
accessible — Confluence's rule. Two other readings were considered; they differ only in mixed
chains (✓ accessible, ✗ not):

| Graph | (a) needs its parent | **(b) chosen** | (c) blocks |
|---|---|---|---|
| RG 1 ✓ → folder F (strict, ✗) → file X (non-strict, direct grant) | X visible | **X visible** | X hidden |
| RG 1 ✓ → folder G (non-strict, ✗) → file Y (strict, direct grant) | Y hidden | **Y hidden** | Y visible |
| RG 1 ✓ → R3 (non-strict, ✗) → R6 (non-strict, own grant) → R7 (strict, inherits) | R7 visible | **R7 hidden** | R7 visible |

**(b) needs no upward walk — for listings.** Two traversals with a per-hop rule cover them:

- **Arm 1 — from the partition root** (the App, or a top-level group the user can reach). Walking
  down from an accessible parent `p`, a child `c` passes when one of three mutually exclusive
  branches holds:
  - `accessRule(c) = RESTRICTED` → `inherits(c→p) AND grant(c)`
  - `accessRule(c) = STRICT` → `inherits(c→p) OR grant(c)`
  - `accessRule(c) = OPEN` → `inherits(c→p) OR grant(c)`, with ancestors irrelevant (decision 3)

  Written as boolean algebra, never a nested `CASE`: Neo4j 5.26 crashes its planner on an
  `EXISTS { }` referencing two pattern-internal variables inside a `CASE`.

  An earlier wording applied a restriction ternary to *every* node, which contradicted §3.2's
  `OPEN` row and hid AC-19. That class of error is now **structurally impossible** — with one
  three-valued field there is no "restricted but not strict" state to mis-read, which is the
  strongest argument for the collapse.

  Since the walk started at the App and every hop was accessible, "every ancestor is accessible"
  holds by construction — nothing is checked upward.
- **Arm 2 — from the user's own grants.** The entry points are the nodes the user holds a direct
  grant on whose `accessRule` is `OPEN`; each is accessible on its own (decision 3). Expanding down
  from an entry uses the same rule plus one condition: **`STRICT` and `RESTRICTED` nodes are
  skipped**, because something above the entry is inaccessible and (b) can never admit them.

Union the arms, de-duplicate, and place the arm-2 tops as in §3.3. Each arm is one Quantified Path
Pattern in Neo4j and one AQL traversal with `PRUNE` in Arango (S8). Both arms filter the hierarchy
edge on its relationship type, so non-hierarchy links never act as navigation edges (decision 56).

**Browse and scoped requests are the exception.** A downward walk cannot decide whether the *start
node itself* is admissible, because that depends on ancestors the listing never visits — AC-57 is
the case. Those requests need one bounded walk from the start node up to the App, which also yields
`currentNode`, `parentNode` and the breadcrumbs in the same query, applying placement at each level
so an inaccessible ancestor is replaced by the fallback rather than named (NV-28).

**Ancestry over a DAG.** "Every ancestor up to the App" is ambiguous once a record has two parents
(Drive "Shared with Me", decision 55). A strict node is accessible when **at least one** full
hierarchy path from the App is accessible (decision 72); the union over partitions already produces
exactly this, so it costs nothing in the query.

**What it means for connectors.** "Strict" means "a gap anywhere above hides this node", so it
belongs only where the source works that way — Confluence pages and spaces. GitLab is the
counter-example: a direct member of a project who is not in its group can still see the project's
issues, so GitLab's issues are written **non-strict**. They still need the project, because a
non-strict node needs its own grant or inheritance from an accessible parent, and the project is
that parent. Connectors that declare nothing are non-strict (decision 38).

### 3.7 Hierarchy and inheritance edges (agreed)

- `NODE_RELATION` is `RECORD_RELATION` **renamed** (decision 56). It carries every relationship
  type, and the hierarchy is the subset with `PARENT_CHILD` or `ATTACHMENT`; attachments therefore
  stay navigation children. Every traversal filters on the relationship type, so a Jira link or a
  database foreign key never becomes a navigation edge.
- It points **parent → child** (decision 30), and it now also links App → record group, record
  group → record group and record group → record. Those three were not merely un-renamed before —
  the write path never emitted them at all, writing only child → parent `BELONGS_TO`, so a graph
  built by a real sync had no downward path from an App to any record.
- **Inheritance stays a separate `INHERIT_PERMISSIONS` edge**, child → parent (decision 30).
- `BELONGS_TO` stays, so a record group's records can be fetched directly without a deep traversal
  (decision 15).
- One hierarchy parent per node, except Drive "Shared with Me", which is a real second parent
  (decisions 16, 55).
- Knowledge base items get the same edges as anything else — `BELONGS_TO` and inheritance — because
  the permission-model declaration is a shortcut, not a replacement for the real graph (decision 50).

### 3.8 Permission-model declarations

PR #3230 adds a `permissionModel` property on the App and on record groups; this work reads the
same property (decision 31). A declaration **wins inside its scope**: nothing below it is checked
and per-node flags there are not consulted (decision 39).

The enum has three values (`config/constants/arangodb.py:96`), not two:

- `APP_LEVEL` on an App: a user who can reach the App reaches **every record and record group in it**.
- `RECORD_LEVEL`: every record is verified individually — the full check, i.e. no skip. This is the
  safe default and the traversal treats it as "no declaration".
- `RECORD_GROUP_LEVEL` on a record group: a user who can reach the group reaches **its records and
  the groups nested under it**.
- A node's `accessRule` or inheritance below a declaration is ignored **while the
  declaration stands** — but the write path still records the real values and inheritance edges
  (decision 66), so removing or changing a declaration leaves a graph that answers correctly.
- The hierarchy is still walked for navigation, placement and breadcrumbs; only the permission
  check is skipped.

### 3.9 Agreed API and pagination contract

All agreed on 2026-09-12 (decision 58).

- **Partitions** run for global search, filter and flatten. Browse and subtree-scoped requests are
  a single query.
- **Concurrency** is capped (around 8–10 partition queries at a time); the rest are batched.
- **Cursor** holds an offset only for partitions that have contributed; a partition still at offset
  0 is left out entirely (decision 57).
- **Totals**: the exact total is computed once on the first page and carried in the cursor.
  Per-type counts cover the whole result, not just the page.
- **Breadcrumbs, `currentNode` and `parentNode`** come from the listing query. Items carry
  `parentId`; there are no per-item breadcrumbs.
- **`via_parent_id`** (optional, browse only) names the parent the user navigated through, so a
  record with two parents shows the trail actually followed. Ignored unless it is a real,
  accessible hierarchy parent of the node (decision 79).
- **Ids** are bare everywhere — `id`, `parentId`, breadcrumbs, the URL path, the cursor's parent.
  `parentId` is null at the root, and every item carries its parent's type.
- **An inaccessible start node returns 404**, so the response never confirms that it exists.
- **Validation**: today's parameter contract stays (`q` 2–500 characters, `limit` 1–200, lists of
  at most 100), but invalid enum and sort values return 400 instead of silently falling back.
- **Sorting**: case-insensitive by name, nulls last in both directions, ties broken by id, with the
  same comparator in the query and in the merge.
- **Filters**: `q` matches a literal substring on both backends; the size filter excludes nodes
  with no size; the legacy "KB" connector id means all of the user's collections.
- **Cursor robustness**: malformed or out-of-range cursors return 400; a cursor is bound to user
  and org; when a cursor and explicit parameters disagree the cursor wins, except `include`;
  `page` is no longer accepted.
- **Paging under change**: per-partition offsets, so a change mid-walk can cause a duplicate or a
  gap; every page except the last is full; "previous" returns exactly the earlier page.
- **A failed partition fails the request** rather than returning a page with silent holes.
- **Pagination fields**: `startIndex`, `endIndex`, `currentPageItems`, `nextCursor`, `prevCursor`;
  `page` and `totalPages` are gone.
- **Deep links**: the URL carries the cursor, so a reload restores the page; an invalid cursor
  sends the UI back to the first page.
- **Nodes with no top-level group** get one app-direct partition per app.
- **Performance target**: page cost should not grow with the offset (keyset paging per partition),
  aiming for p95 under a second for a 50-item page on the acceptance graph.
- **Other callers**: agent tool parameters and the enterprise-edition override points stay as they
  are (Q25).

## 4. Current state (main)

Read on 2026-09-11/12. The working tree is `main` plus PR #3230; nothing PR #3230 adds is read by
the knowledge hub today. File key: `knowledge_hub_router.py` and `knowledge_hub_models.py` are in
`backend/python/app/connectors/sources/localKB/api/`; `knowledge_hub_service.py` and `kb_service.py`
in `…/localKB/handlers/`; `neo4j_provider.py` in `backend/python/app/services/graph_db/neo4j/` and
`arango_http_provider.py` in `…/graph_db/arango/`; `data_source_entities_processor.py` in
`backend/python/app/connectors/core/base/data_processor/` and `graph_data_store.py` in
`…/core/base/data_store/`.

### 4.1 API

**Python** (`knowledge_hub_router.py:172-294`, prefix `/api/v1/knowledge-hub`, scope `KB_READ`):

- `GET /nodes` — root listing, or global search.
- `GET /nodes/{parent_type}/{parent_id}` — children; `parent_type` is app, recordGroup, folder or record.

**Node.js** proxies both at `/api/v1/knowledgeBase/knowledge-hub/nodes[/:parentType/:parentId]`
(`kb.routes.ts:182-196`) as a pass-through. It also forwards `parentId`, `view` and `kbIds`, which
Python ignores (`kb_controllers.ts:77-112`).

**Parameters:** `page` (from 1), `limit` (1–200), `sort_by` (name, createdAt, updatedAt, size,
type; default updatedAt), `sort_order`, `q` (2–500 characters), `node_types`, `record_types`,
`origins`, `connector_ids`, `indexing_status`, `created_at` / `updated_at` / `size` ranges,
`only_containers`, `flattened` (true / false / omitted), `include` (breadcrumbs, counts,
availableFilters, permissions). No cursor.

**Mode:** an explicit `flattened` wins; otherwise any filter selects *search* and none selects
*browse*. Browse ignores filters except `origins` and `connector_ids` at root
(`knowledge_hub_service.py:147-208, 446-447`).

**Response:** each item (`knowledge_hub_models.py:59-86`) carries `parentId`,
`permission {role, canEdit, canDelete}`, `hasChildren` and `sharingStatus`. Pagination
(`:108-115`) is `{page, limit, totalItems, totalPages, hasNext, hasPrev}`.

**Other callers and override points**

- Agent tools call the service directly with parameters HTTP does not expose — `record_group_ids`,
  `depth`, `include_typed_records` (`knowledge_hub_service.py:108-110`; callers in
  `agents/actions/knowledge_graph/navigator.py:324-340`,
  `agents/actions/knowledge_hub/knowledge_hub.py:331-347`,
  `agents/actions/knowledge_graph/ops/listing.py:211-227`).
- Some callers skip the service and call v1 provider functions directly:
  `get_knowledge_hub_node_access` (`navigator.py:263`, `resolver.py:264`),
  `get_knowledge_hub_breadcrumbs` (`navigator.py:305`), `get_node_depths_batch` (`navigator.py:348`),
  `get_knowledge_hub_filter_options` (`catalog.py:131`, `agents/chat_modes/bridge.py:174`) and
  `get_knowledge_hub_parent_node` (`localKB/api/kb_router.py:1405`). They move to v2 in a later
  phase (decision 33).
- The enterprise edition can replace the whole service (`edition_config.knowledge_hub_service_factory`,
  `connectors_main.py:887-888`); `_resolve_user` and `_get_user_app_ids` are its override points.

### 4.2 Queries and permission logic

| Operation | How it works today |
|---|---|
| Root | One query over the user's reachable apps. Hides `hideConnector` apps except KBs — a clause decision 63 drops, since only collections carry the flag (§4.9). It does **not** filter on `isActive`, and decision 51 keeps it that way. |
| Browse | One query per parent. Loads every child, checks each with a 20-hop inheritance walk, then sorts and slices. App → its top-level RGs (KB App → its root records). RG → child RGs plus records whose `externalParentId` is null. Folder/record → one hop of `RECORD_RELATION` (PARENT_CHILD, ATTACHMENT). |
| Search | One global query, permission-first: seed RGs from grants, expand through `INHERIT_PERMISSIONS`, add directly granted records and KB-app seeds, scope to the parent, filter, sort, `SKIP`/`LIMIT`, then hydrate the page. Neo4j runs 3 queries, Arango 2. Never returns App nodes. |
| Breadcrumbs | One query per ancestor level, up to 20. `currentNode` and `parentNode` are separate queries again. |

**Deleted records are not filtered anywhere in the hub's root, browse or search blocks** — decision
54 changes that. `isDeletedAtSource` exists only as a schema default (`schema/arango/documents.py:89,
126, 758`) and is read once in a record-group listing; nothing writes it, so the rule uses
`isDeleted` alone.

**Role helper for App nodes** (`neo4j_provider.py:16036-16052`; the same ladder in Arango at
`arango_http_provider.py:17999-18013`): no access path at all → null; a direct permission edge →
its role; a team share → the user's team role; team app relation only → READER (EDITOR for org
admins); org admin → EDITOR on team-scoped apps, OWNER otherwise; creator → OWNER; everyone else →
READER. Collection CRUD uses a different rule — `get_user_kb_permission`
(`neo4j_provider.py:9829-9877`) takes the highest of the direct user grant and team-derived roles
and returns nothing otherwise. Decision 52 makes the hub follow the CRUD rule for collections.

**Role helper for records and record groups** (`neo4j_provider.py:15755`, `arango_http_provider.py:17577`):
the targets are the node plus its `INHERIT_PERMISSIONS` ancestors (1–20 hops), roles come from the
five grant paths, and the highest wins by `OWNER 6 > ADMIN 5 > EDITOR 4 > WRITER 3 > COMMENTER 2 >
READER 1`. ORGANIZER, FILEORGANIZER and OTHERS are dropped.

**Grant types dropped at write time:** connectors produce DOMAIN, ANYONE and ANYONE_WITH_LINK
permissions, but the data processor never stores them — those branches are commented out
(`data_source_entities_processor.py:828-840`). Decision 59 removes the concept outright: the dead
branches, the enum values and the connector code that produces them all go, and organisation-wide
sharing is expressed as an org grant instead.

### 4.3 Gaps against the requirements

| # | Gap | Where |
|---|---|---|
| C1 | Browsing into a node never checks that the user may access it. Names in `currentNode`, `parentNode` and breadcrumbs are returned unchecked. | `knowledge_hub_service.py:698-700` |
| C2 | Search results carry no parent id (Neo4j hard-codes null) and no permission role. | `neo4j_provider.py:16860, 16923` |
| C3 | In browse, `parentId` is a prefixed document id while `id` is bare. | e.g. `neo4j_provider.py:15230` |
| C4 | `hasChildren` ignores permission. | |
| C5 | Pagination by page number; every page sorts the full filtered list. | `knowledge_hub_models.py:108-115` |
| C6 | Breadcrumbs cost one query per level. | §4.2 |
| C7 | With no role found, context permission defaults to READER, so the service's "no permission" branch never runs. | `neo4j_provider.py:14699`, `arango_http_provider.py:15662` |
| C8 | Role vocabularies disagree across the code. | `neo4j_provider.py:15813`, `:9829-9877` |
| C9 | A nested connector RG's `parentNode` is read from `rg.parentId`, which is not in the schema. | `neo4j_provider.py:15006-15157` |
| C10 | Neo4j and Arango disagree in about ten places: KB root children, scoped search under an RG, search depth (5 hops vs 20), error handling. | |
| C11 | Filters differ by backend: Neo4j ignores a range bound of 0; Arango's size filter keeps nodes with no size. | `neo4j_provider.py:15708-15727`, `arango_http_provider.py:14208-14262` |
| C12 | Deleted records are never filtered out of hub results. | §4.2 |

### 4.4 Hierarchy and inheritance edges

Verified against `data_source_entities_processor.py`. Parent links use three edge types in two
directions:

| Link | Edge | Direction | Written at |
|---|---|---|---|
| top-level RG → App | `BELONGS_TO` | child → parent | `:1735-1757` (RGs with no parent only) |
| RG → parent RG | `BELONGS_TO` (tagged `entityType: KB` for every connector) | child → parent | `:1781-1794` |
| record → its RG | `BELONGS_TO` (every record in the group, nested ones too) | child → parent | `:472` |
| parent record → child record | `RECORD_RELATION` (PARENT_CHILD or ATTACHMENT) | parent → child | `:282-333` |
| KB record → KB App | `BELONGS_TO` (`entityType: KB`) | child → parent | `:159-170` |

Inheritance (`INHERIT_PERMISSIONS`, child → parent):

- record → its RG when `inherit_permissions` (default true), deleted otherwise (`:474-477`) —
  **including records nested under other records**;
- RG → parent RG when the RG's `inherit_permissions` (default false); never deleted (`:1796-1803`);
- KB record → KB App when `inherit_permissions` (`:171-182`);
- **never written:** record → parent record (the helper at `graph_data_store.py:601` has no
  callers), and RG → App. RGs auto-created from a record get no App edge at all (`:451`).
- `inherit_permissions` is not stored on the node; it only decides whether the edge exists.

`RECORD_RELATION` also carries the non-hierarchy relation types — the enum has fifteen values in
all, of which PARENT_CHILD and ATTACHMENT are the hierarchy: `SIBLING`, `OTHERS`, `LINKED_TO`,
`BLOCKS`, `DUPLICATES`, `DEPENDS_ON`, `CLONES`, `IMPLEMENTS`, `REVIEWS`, `CAUSES`, `RELATED`,
`FOREIGN_KEY`, `DERIVED_FROM` (`config/constants/arangodb.py:609-633`). After the rename they ride on `NODE_RELATION` and are
excluded from traversal by relationship type (decision 56).

**Drive "Shared with Me"** (`google/drive/team/connector.py`): each synced user gets two record
groups of their own, "My Drive" and "<name>'s Shared with Me", both with an OWNER grant for that
user (`:1196-1216`); the Shared-with-Me group is marked internal because it has no source id. A
file shared with a user gets a `BELONGS_TO` link to that group — no hierarchy link and no
inheritance today (`:1883-1905`, `data_source_entities_processor.py:479-487`). Decision 55 makes it
a real second hierarchy edge. For shares from a personal drive the connector also clears the file's
own drive group (`:1849-1850`); Shared Drive items keep theirs.

**Connector details that matter for the new model**

- *Confluence Cloud:* READ-restricted pages are written with inheritance off; pages with only edit
  restrictions keep inheriting from the space (`confluence_cloud/connector.py:1625-1629`). Decision
  26 changes the first half.
- *Slack:* every channel record group, DMs included, is created with `inherit_permissions=True` and
  `hide_children=True` (`_to_channel_record_group` in both Slack connectors). Decision 40 sets the
  inheritance flag to false for team Slack; `hide_children` is what keeps Slack messages out of the
  tree.
- *SharePoint Online:* `_get_item_permissions` (`sharepoint_online/connector.py:2889`) fetches a
  drive item's permissions and hands them to `_convert_to_permissions` (`:2950`), which reads
  `granted_to_v2`, `granted_to_identities_v2`, `roles` and `link` and **never looks at
  `inheritedFrom`** — so permissions the item merely inherited are written as its own grants. Drive
  files and folders never set `inherit_permissions` either, so they fall back to the model default
  (`True`, `entities.py:233`). Decision 47 fixes both, but **not** through Graph's `inheritedFrom` —
  see §4.10 and decision 64. Pages, lists and list items have no permission endpoint in Graph at
  all: `_get_page_permissions`, `_get_list_permissions` and `_get_list_item_permissions` each
  return an empty list with a comment saying so (`:2911-2948`). A page record is built from id,
  title, timestamps, page layout, promotion kind, eTag and web URL, with `inherit_permissions=True`
  hard-coded (`:2109-2131`); nothing on the Graph `sitePage` object says whether the page inherits.
  Lists and list items set no flag at all and fall to the model default `True`; lists *are* synced
  today (`:1650-1688`, items at `:1822-1836`, capped at 1000 per list). Each library's root folder
  is synced as a folder record, `<drive>:root:<id>` (`:1335`), and stays a node (decision 48).
- *GitLab:* a project's record group parents to the longest **included** group path
  (`gitlab/projects.py:510-515, 525`), and the five per-project groups hang off the project
  (`:527-566`). Group record groups carry no parent of their own; decision 49 leaves that alone.
- *Deletes:* `cascade_children=False` (`data_source_entities_processor.py:1545`) deletes a record
  and keeps its children; decision 53 says where those children land.

The Arango edge definitions drift from what is written: `inheritPermissions` is declared
recordGroups → records (`schema/arango/graph.py:14-18`), the reverse of every written edge;
`belongsTo` does not declare RG → RG, RG → App or record → App; `permission` does not declare Apps
as a target, though KB grants are user/team → App.

### 4.5 Knowledge bases

- A KB is its own App (`type: KB`, personal scope, `hideConnector`) — `kb_service.py:281-301`.
  Creating one writes a user → App grant (OWNER), an org–app relation and a user–app relation.
- Sharing writes user → App grants with a role, and team → App grants **without** a role; a
  member's role comes from the user → team edge.
- There is no record group level: records and folders hang off the KB App, and nesting is
  `PARENT_CHILD` from the parent folder.
- `kb_service` accepts OWNER, ORGANIZER, WRITER, COMMENTER, READER (`kb_service.py:1294, 1382`);
  `get_user_kb_permission` ranks OWNER > WRITER > READER > COMMENTER. The UI offers only OWNER,
  WRITER, READER. Decision 41 removes the rest; decision 52 makes the hub use this rule.
- COMMENTER is also the connector permission model's `PermissionType.COMMENT`
  (`models/permission.py:13`), written by Google Drive (team) (`:787-788`) and returned by
  Confluence Data Center (`confluence_datacenter/connector.py:3307`). Those connectors write READER
  instead (decision 41).

### 4.6 Tests

- Router and service: `tests/unit/connectors/sources/test_knowledge_hub_router*.py` (91 cases),
  `tests/unit/connectors/sources/localKB/test_knowledge_hub_service.py` (80).
- Providers: knowledge hub classes in `tests/unit/services/graph_db/test_arango_http_provider_records.py`
  (14515–15478) and `test_arango_http_provider_full_coverage.py` (846–1466), plus
  `test_neo4j_provider.py` (`TestKnowledgeHubSearchThreePhase`, 2905). The interface inventory test
  `interface/test_graph_db_provider.py` (`TestAbstractMethodInventory.EXPECTED_METHODS`, 173)
  enumerates every abstract method and fails until the v2 names are added.
- Agents: `tests/unit/agents/actions/test_knowledge_hub_summaries.py`, `…/knowledge_graph/test_*.py`.
- Backend integration: `tests/integration/test_graph_navigation_e2e.py` (33).
- Node.js: `kb_controllers.test.ts`, `kb.routes.test.ts`.
- `integration-tests/connectors/confluence/` asserts every item carries `permission.role` — which
  decision 34 changes for connector items.

### 4.7 Frontend consumers of the knowledge hub API

All three paginate by page number today (`page`, `limit`, `hasNext`; next page = `page + 1`).

| Consumer | Endpoint | Where |
|---|---|---|
| Knowledge base page — tree, data area, search, filters | `/knowledge-hub/nodes`, `/knowledge-hub/nodes/:nodeType/:nodeId` | `frontend/app/(main)/knowledge-base/api.ts`, `page.tsx` |
| Chat collection picker | `/knowledge-hub/nodes` | `frontend/app/(main)/chat/api.ts:643` |
| Agent builder apps palette | `/knowledge-hub/nodes`, paged through every root node | `frontend/app/(main)/agents/api.ts:659`, `:696` |

### 4.8 What this means for the first goal

The rules check every node against its **direct parent**. Today's graph cannot answer that: nested
records inherit straight from their record group, nothing marks a node restricted or strict, and
parent links use three edge types in two directions. So the write path changes first (decision 12),
and v2 only has to be right on data it produced (decision 32).

### 4.9 `hideConnector` — what it actually marks

Checked on 2026-09-12, because the root listing filters on it.

- Every connector is built with `hideConnector: False` (`connectors/core/registry/connector_builder.py:54`).
- Exactly one connector sets it true: the knowledge base (`localKB/connector.py:56`). Local FS sets
  it false explicitly (`local_fs/connector.py:275`), and no other connector mentions it.
- Every write of `"hideConnector": True` into the graph creates a **collection App**:
  `kb_service.py:298` (commented "Excluded from main connector management UI"), the Kafka entity
  handler's KB creation (`services/messaging/kafka/handlers/entity.py:602`), and the two
  record-group → App migrations (`arango_http_provider.py:5699`, `neo4j_provider.py:3598`).
- Two things read it: the connector **registry** listings (`connector_registry.py:961, 989`), which
  keep collections out of the connector-management UI, and the hub's root query
  (`neo4j_provider.py:13636`, `arango_http_provider.py:13888`).

So there is no such thing today as a hidden app that is not a collection, and the hub's clause
`app.type = 'KB' OR NOT coalesce(app.hideConnector, false)` can never exclude a row: the only apps
carrying the flag are collections, which the first half of the condition re-admits. It is a
connector-management concern, not a permission or navigation one, so v2 stops reading it
(decision 63).

### 4.10 How SharePoint actually exposes permission inheritance

Checked against Microsoft's documentation on 2026-09-13, because decisions 47 and 62 rested on
what the API offers rather than on what the current connector happens to call.

| Surface | Inheritance signal | Verdict |
|---|---|---|
| Graph `sitePage` | none — the v1.0 property and relationship tables carry no permissions relationship and no inheritance field | unusable |
| Graph `driveItem` permissions | `permission.inheritedFrom` exists in the schema, and `driveItem-list-permissions` says callers can differentiate with it | **unreliable for SharePoint** |
| Graph `listItem` / lists | no `hasUniqueRoleAssignments`; Graph does not expose it in v1.0 or beta | unusable |
| SharePoint REST `HasUniqueRoleAssignments` | a documented boolean on every securable object — web, list, library, folder, item | **the mechanism to use** |

Two Microsoft sources contradict each other on `inheritedFrom`, and the tie-breaker is explicit:

- The `permission` resource page states: *"OneDrive for Business and SharePoint document libraries
  don't return the **inheritedFrom** property."*
- A Microsoft Q&A moderator, answering exactly this question, says of the observed behaviour
  (absent when unique, `{}` when inherited): *"Since it is not documented in the official
  documents, this feature is not reliable to determine the type of permission"*, and points to
  `HasUniqueRoleAssignments`, noting the Graph API does not support that property.

So inheritance has to come from the SharePoint REST API. **The connector is already there:**
`_get_sharepoint_access_token` (`:2350`) requests a token for `https://{host}/.default` —
explicitly "NOT Graph API" — using the same certificate or client-secret credentials, and the
connector already calls `_api/web/roleassignments?$expand=RoleDefinitionBindings,Member` (`:2748`),
`_api/web/sitegroups` (`:3100`), and, for pages, `_api/web/GetFileById('{page_id}')/ListItemAllFields`
(`:3972`). Adding `HasUniqueRoleAssignments` reuses all of it.

**Cost.** Read per item this is an extra REST call per node. It does not have to be: the flag can
be selected for a whole list in one paged call —
`_api/web/lists(guid'<id>')/items?$select=Id,HasUniqueRoleAssignments` — and only items that come
back `true` need their own `roleassignments` lookup. Pages are items in the "Site Pages" library,
so the same batched call covers them.

**Two things to settle before building on this.**

1. *Verify the REST path actually works against a real tenant.* Every existing SharePoint REST call
   swallows its errors: `_get_sharepoint_group_users` logs a warning and returns `[]` for anything
   but a 404 (`:2730-2739`), `_get_custom_sharepoint_groups` returns `[]` on any exception, and
   `_get_page_content` returns `None` (`:3987-3989`). So if the app registration carries only Graph
   permissions and not SharePoint ones, these calls have been failing silently all along and nobody
   would know. Decision 64 must not be built on an untested path — confirm a live `200` for
   `_api/web/lists(guid'…')/items?$select=Id,HasUniqueRoleAssignments` first.
2. *Principal resolution.* `roleassignments` returns SharePoint principals — site groups, security
   groups, and users as login names such as `i:0#.f|membership|user@contoso.com` — not the Graph
   ids the rest of the write path uses. The connector already resolves these at site level
   (`_resolve_site_permissions` and the group lookups at `:2706-2769`), so item-level assignments
   reuse that, but it is real work, not a field rename.

## 5. Prior attempt — what to carry, what to change

PR #1522 (GitHub head `cd9c39ef42`, 2026-04-28) and the newer local commit `f2f0793a3` on
`refactor/permission-hierarchy` (2026-07-17, not pushed). Code is not reused (R8); its frontend
may serve as a reference (decision 22).

### 5.1 Carried forward, as ideas

- **QPP traversal** (Neo4j ≥ 5.9; docker-compose and Helm both ship 5.26.0): a per-hop
  predicate, and a failing hop stops expansion of that branch inside the database.
- **Cursor** (`app/utils/cursor.py` on that branch): URL-safe base64 JSON with `dir`, `sf`/`sd`,
  `lm`, `src` (per-partition offsets), `seen`, `f` (filters), `pi`/`pt` (parent), `tc` (total).
- **K-way heap merge** of per-partition, pre-sorted results.

### 5.2 Problems seen in its pagination and merge

| # | Problem | Where |
|---|---|---|
| P1 | Partition = connector (App); R7 needs top-level record groups and knowledge bases. | `_query_global_search_v3` |
| P2 | `parentId` hard-coded to null in results. | `_query_single_connector_v3` |
| P3 | Breadcrumbs: one query per ancestor level, up to 20 levels. | `get_knowledge_hub_breadcrumbs` |
| P4 | Top-down only — directly shared nodes under an inaccessible parent are never found. | QPP from the App |
| P5 | Every page collects and sorts a partition's whole accessible set, then slices by offset. | `collect(node)` … `allNodes[$offset..]` |
| P6 | "Has more" is true only if one partition alone returned more than the limit. | `_query_global_search_v3` |
| P7 | Previous-page offsets are approximated per partition (offset − limit). | `compute_query_offsets_for_prev` |
| P8 | Descending sort on string fields merges in ascending order. | `_extract_sort_key` |
| P9 | Size sorts as a string in Cypher, so 9 B sorts after 10 B. | sort expressions |
| P10 | A failed partition query is logged and treated as empty. | `_query_single_connector_v3` |

### 5.3 How it modeled restriction and strictness (answers Q1)

- `hasRestriction` is a **per-node** flag on records and record groups, set only by Confluence Cloud.
- A **connector-level** `hasRestriction` is set with `with_has_restriction()` and stored on the
  App. No query reads it.
- **Strictness is not modeled.** The traversal walks top-down and prunes, so it behaves strictly
  for every node. A request-level `strict_permission` parameter is marked "future use".
- **Arango has none of it:** its v3 delegates to the older v2.

### 5.4 How it modeled hierarchy and inheritance (answers Q5)

| Link | Hierarchy edge | Inheritance edge |
|---|---|---|
| App → top-level RG | `RECORD_RELATION` PARENT_CHILD | RG → App |
| RG → child RG | `RECORD_RELATION` PARENT_CHILD | child RG → parent RG |
| RG → top record | `RECORD_RELATION` PARENT_CHILD | record → RG |
| record → child record | `RECORD_RELATION` PARENT_CHILD or ATTACHMENT | record → parent record |
| KB App → KB root item | none | none |

The shape matches decisions 6, 15 and 30 — inheritance to the direct parent on its own edge, one
hierarchy link per node — but it sits on the un-renamed edge, and KB roots are not linked.

### 5.5 Is its QPP query fool-proof? (answers Q20)

No. The mechanism — a per-hop rule that stops expanding at the first node that fails — is sound and
carries over. The query does not meet the requirements and has bugs (from reading the code).

**Requirements it does not meet**

| # | Gap |
|---|---|
| G1 | Non-strict nodes under an inaccessible parent are never reached (no arm 2). |
| G2 | No per-node strictness; `strict_permission` is unused. |
| G3 | No navigation placement; `parentId` is null; breadcrumbs are per-level lookups over the physical path. |
| G4 | Knowledge bases: it walks from the KB App over hierarchy edges that KB items do not have, so no KB content is found, and no KB role. |
| G5 | Partitions per connector, not per top-level record group or knowledge base. |
| G6 | Scoped queries never check the start node. |

**Bugs in the permission rule**

| # | Bug |
|---|---|
| B1 | Org-wide grants never match: the rule compares against `$orgNodeId`, which is never computed. |
| B2 | "The record group inherits from the App and the user has the app relation" counts as a *direct grant*, so a restricted group that inherits from the App passes for every app user. |
| B3 | The app relation is matched only directly, not through a team. |
| B4 | The start-node check accepts an unrestricted node on a direct grant alone, and for inherited access checks only the top ancestor. |
| B5 | A Confluence page with only edit restrictions is marked restricted but keeps no READ grants, so nobody can see it. |

**Query shape**

| # | Problem |
|---|---|
| X1 | The start node is matched without a label, so no index can be used. |
| X2 | It returns paths, not nodes, with no `DISTINCT`. |
| X3 | Deleted, internal, placeholder and hidden-children nodes are not excluded. |
| X4 | Text search is `toLower(name) CONTAINS` over the traversed set, after the traversal. |

## 6. Suggestions — for discussion, not decided

| ID | Suggestion | Relates to |
|---|---|---|
| S1 | Connectors set strictness per node when they write it; a connector-level default in the builder fills it for the rest (non-strict, decision 38). | Decision 38 |
| S2 | Arm 2 (§3.6) looks up entries by the user's grant edges, restricted to the partition through `BELONGS_TO`; entries already reached by arm 1 are de-duplicated. | §3.6 |
| S3 | For searches with a text query, start from index candidates and check permission upward, instead of traversing the whole accessible tree first. | P5, X4 |
| S7 | One acceptance graph — Examples 1–3, a collection, the placement cases of §3.3 and the bugs above — run live against both v2 implementations on throwaway Neo4j 5.26 and Arango 3.12 containers. The same cases decide Neo4j/Arango parity. | R9, R14 |
| S8 | Arango: an AQL traversal per arm from each partition root, with `PRUNE` on the per-hop rule and the same rule as a `FILTER`. | R9, §3.6 |
| S11 | `hasChildren`: one existence check per row **inside the listing query** (decision 46), stopping at the first accessible child; measure it in the acceptance harness. | Decision 46 |
| S12 | ~~Drop group → App inheritance from the model entirely.~~ **Rejected** — decision 66 keeps the real edges as the fallback for a declaration that changes. The bug class it would have removed is handled instead by connectors writing the flag correctly (decisions 26, 40). | Decision 66 |

## 7. Open questions

Everything else is answered and recorded in §8.

| ID | Question |
|---|---|
_None._ Every question raised in this design is answered and recorded in §8. New questions arising
during implementation are added here.

## 8. Decisions log

Cited as D*n* in the test-case document; the log runs to D74.

| # | Decision | Date | From |
|---|---|---|---|
| 1 | Strictness is decided per node. | 2026-09-11 | Q1 |
| 2 | Strict: a node needs access to every ancestor up to the App. A direct grant without the parents does not count. | 2026-09-11 | Q2 |
| 3 | Non-strict (now `accessRule = OPEN`): parents are not required and no restriction applies. The top of a continuous-permission chain below a gap appears as a direct child of the App — or of its record group when the user can access it. | 2026-09-11 | Q3, Q8 |
| 4 | Inaccessible nodes are never shown. Inaccessible top-level record groups are still searched for directly shared nodes; strict content can be skipped. | 2026-09-11 | Q13 |
| 5 | Breadcrumbs follow the navigation the knowledge hub shows. | 2026-09-11 | Q9 |
| 6 | Inheritance goes to the direct parent: record → parent record, top record → record group, record group → parent group, top-level group → App. | 2026-09-11 | Q5 |
| 7 | Each knowledge base is one partition. | 2026-09-11 | Q11 |
| 8 | Cursor pagination keeps nearly the same structure; the contract is §3.9. | 2026-09-11 | Q15 |
| 9 | Roles matter only for knowledge base items; the role is set on the collection and applies unchanged to every file and folder in it. | 2026-09-11 | Q18 |
| 10 | Every frontend feature that uses the knowledge hub API is in scope. | 2026-09-11 | Q19 |
| 11 | Both backends support the algorithm before anything else is built on it. | 2026-09-11 | Q20 |
| 12 | Order of work: **write path first**, then the v2 read queries on both backends, then the frontend; remaining connectors after that; migration/backfill last. | 2026-09-12 | Q21, Q22, Q36 |
| 13 | A record below a gap appears under its own record group if the user can access that group, otherwise directly under the App. | 2026-09-11 | Q26 |
| 14 | `parentId` is the parent the node appears under in navigation — never a node the user cannot access. | 2026-09-11 | Q27 |
| 15 | `BELONGS_TO` stays, so a record group's records can be fetched without a deep traversal. | 2026-09-11 | Q28 |
| 16 | One hierarchy parent per node, except Drive "Shared with Me". | 2026-09-11 | Q29 |
| 17 | Strictness is per node and mixable within a connector. | 2026-09-11 | Q30 |
| 18 | The three connectors: Confluence Cloud, SharePoint Online and GitLab. | 2026-09-11 | Q31 |
| 19 | The new write path applies to every connector from the start; the three get focused testing. Production only after every connector is tested. | 2026-09-11 | Q32 |
| 20 | v2 is chosen in code; later v1 is removed and v2 takes its name. | 2026-09-11 | Q34 |
| 21 | For a collection, the highest role over all paths wins, and everything inside it is visible to anyone with access. For connectors roles do not matter — only whether access exists. | 2026-09-11 | Q35 |
| 22 | The frontend moves to the new contract, after the read queries (decision 12). | 2026-09-11 | Q36 |
| 23 | A separate test-case document is maintained from now on. | 2026-09-11 | — |
| 24 | Strictness means **every ancestor up to the App** must be accessible (reading (b), §3.6). | 2026-09-12 | Q30 |
| 25 | Inheritance is a per-node fact: neither inheritance nor a grant means no access; a restricted node needs both. | 2026-09-12 | Q37 |
| 26 | Confluence Cloud writes READ-restricted pages as **inheriting**, with the restriction flag; edit-only restrictions keep inheriting and are not flagged. | 2026-09-12 | Q37, Q64 |
| 27 | A record group below a gap is placed by the same rule as a record. | 2026-09-12 | Q41 |
| 28 | Flatten returns every node in the subtree of the node it runs on, as a flat list. | 2026-09-12 | Q42 |
| 29 | "Shared with Me" groups are shown in the hub, and a shared record appears in both places when the user can reach both (see decision 55). | 2026-09-12 | Q40, Q44 |
| 30 | The hierarchy edge points parent → child. Inheritance stays a separate `INHERIT_PERMISSIONS` edge, child → parent. | 2026-09-12 | Q28 |
| 31 | The hub reads the `permissionModel` declarations on App and RecordGroup (§3.8). | 2026-09-12 | Q33, Q23 |
| 32 | v2 has to be correct on data written by the new write path; the migration covers older data. | 2026-09-12 | Q66 |
| 33 | Callers that bypass the service move to v2 in a later phase. | 2026-09-12 | Q67 |
| 34 | Connector items carry no role: `permission` is null. | 2026-09-12 | Q49 |
| 35 | Roles are removed from the code base (final list in decision 41). | 2026-09-12 | Q38 |
| 36 | Every case is discussed before it is implemented; nothing is decided by assumption. | 2026-09-12 | — |
| 37 | A node's restriction (now `accessRule = RESTRICTED`) applies to the node it is on and is never inherited by its children. | 2026-09-12 | Q71 |
| 38 | Connectors that declare nothing are **non-strict** by default. | 2026-09-12 | Q39 |
| 39 | A permission-model declaration wins inside its scope; per-node flags below it are not consulted. | 2026-09-12 | Q71 |
| 40 | Team Slack writes `inherit_permissions=False` on channel groups; individual Slack keeps `true`. | 2026-09-12 | Q59 |
| 41 | ORGANIZER, COMMENTER, FILEORGANIZER and OTHERS are removed from the whole code base; connectors write a commenter as READER; stored grants are not migrated. | 2026-09-12 | Q38, Q72 |
| 42 | All five grant paths stay: user, group, role, team, org. | 2026-09-12 | Q4 |
| 43 | The connector gate is any permission that reaches the user — directly or through another entity — or a user–app relation. | 2026-09-12 | Q6 |
| 44 | Placeholder stubs are shown, so the hierarchy stays intact. Slack messages stay hidden through `hide_children`. | 2026-09-12 | Q7 |
| 45 | Search and filter can return every kind of node, Apps included. | 2026-09-12 | Q46 |
| 46 | `hasChildren` may be permission-aware only if the check runs inside the listing query. | 2026-09-12 | Q48 |
| 47 | SharePoint Online follows the source's own inheritance flag: an inheriting item is written as inheriting with no grants of its own; an item with unique permissions is written with its own grants and no inheritance. Part of the first goal. The **mechanism** is decision 64 — not Graph's `inheritedFrom`. | 2026-09-12 | Q60 |
| 48 | A document library's root folder stays a node in the hierarchy. | 2026-09-12 | Q61 |
| 49 | GitLab's nested source groups are not mirrored in the hierarchy. | 2026-09-12 | Q62 |
| 50 | Knowledge base items keep their `BELONGS_TO` and inheritance links; the declaration is a shortcut, not a replacement. | 2026-09-12 | Q63 |
| 51 | `isActive` does not change knowledge hub behaviour. | 2026-09-12 | Q65 |
| 52 | For collections, access and role come from explicit grants only — a direct user grant or a team share, the same rule collection CRUD uses. The admin, creator and READER fallbacks go. Connector apps keep the gate (decision 43) and a null role (decision 34). | 2026-09-12 | Q68 |
| 53 | When a record's parent is not in the graph, its record group becomes its parent; with no record group, the App. The same applies to children that survive a parent deleted with `cascade_children=False`. | 2026-09-12 | Q45 |
| 54 | Records with `isDeleted = true` are never returned. `isDeletedAtSource` is not used — nothing writes it. | 2026-09-12 | Q73 |
| 55 | Drive "Shared with Me" is a **real second hierarchy edge**: a shared record appears both in its drive location and under the Shared-with-Me group, whenever the user can reach both. | 2026-09-12 | Q40 |
| 56 | `NODE_RELATION` is `RECORD_RELATION` renamed. It carries every relationship type; the hierarchy is the subset with PARENT_CHILD or ATTACHMENT, so attachments stay navigation children, and traversals filter on the type. | 2026-09-12 | Q28 |
| 57 | The cursor stores an offset only for partitions that have contributed; a partition still at offset 0 is left out. | 2026-09-12 | Q14 |
| 58 | The API and pagination contract of §3.9 is agreed (Q12, Q16, Q17, Q24, Q25, Q47, Q50–Q58, Q69). | 2026-09-12 | §3.9 |
| 59 | DOMAIN, ANYONE and ANYONE_WITH_LINK grants are removed from the model and the code base, not revived. Organisation-wide sharing is an org grant. A Drive file opened through an anyone-link still reaches the user through "Shared with Me" (decision 55), so nothing is lost. | 2026-09-12 | Q78 |
| 60 | A grant stored with a retired role reads as READER, so access is kept. A later migration rewrites the stored values to READER; nothing is rewritten now. | 2026-09-12 | Q80 |
| 61 | `hasChildren` is navigation-aware: it answers "would browsing this node return anything", so it counts nodes placed under it from below a gap and excludes inaccessible children. It stays inside the listing query (decision 46). | 2026-09-12 | Q81 |
| 62 | ~~SharePoint pages, lists and list items are written as inheriting unconditionally, Graph exposing no flag for them.~~ **Superseded by decision 64** on 2026-09-13: a flag does exist, outside Graph. | 2026-09-12 | Q82 |
| 63 | The hub ignores `hideConnector`. The root listing's `app.type = 'KB' OR NOT hideConnector` clause is dropped: only collections carry the flag (§4.9), so the clause never excludes a row, and it is a connector-management concern. | 2026-09-13 | Q79 |
| 64 | SharePoint inheritance is read from the REST API's `HasUniqueRoleAssignments`, for drive items, pages, lists and list items alike (§4.10). `false` → the node is written as inheriting, with no grants of its own. `true` → its own `roleassignments` are read and written as the node's grants, with inheritance off. Graph's `permission.inheritedFrom` is not used: undocumented for SharePoint, and Microsoft's own guidance calls it unreliable. The flag is selected per list in one paged call, not per item. | 2026-09-13 | Q82, Q60 |
| 65 | When the flag cannot be read, the write path never guesses. A **single item** failing: on a re-sync leave that node's permission state as it is; on a first sync write it as inheriting and log it, the alternative being a node nobody can see. **Every** call failing — the app registration lacks SharePoint API permission, or the token is refused — aborts the sync run instead of marking a whole tenant's content as inheriting, which would over-share it. | 2026-09-13 | Q83 |
| 66 | Real inheritance edges are always written and kept, **including record group → App**, even where an `APP_LEVEL` or `RECORD_GROUP_LEVEL` declaration already covers the scope. A declaration is an optimization layered over a complete graph, not a replacement for it: a connector can change or drop one, and the edges are the fallback. The corollary turns on whether the group is **restricted**. A `RESTRICTED` group may inherit from the App safely, because RESTRICTED demands a grant on top of inheritance: Confluence spaces are written `[R!]` and do inherit, and a user with app access but no space permission still sees nothing — verified on both backends. A group that is **not** RESTRICTED must not inherit from the App, since inheritance alone would then admit every app user; a team Slack channel is that case (decision 40), and so is a Confluence Data Center *personal* space, which is `[S]` and is admitted by its ConnectorGroup grant instead. | 2026-09-13 | Q74 |
| 67 | A record with two hierarchy parents is returned **once** in a flat result (search, flatten). Its `parentId` is the drive location when the user can reach both, and otherwise whichever parent they can reach — there is no App fallback while a reachable parent exists. In browse it is still listed under each parent the user can reach (decision 55). Every item carries its parent's id and type. | 2026-09-13 | Q75 |
| 68 | The missing-parent fallback runs in **both** places. At write time, deleting a node re-parents its surviving children to their record group, or the App when they have none. At read time, a node whose hierarchy parent does not resolve is placed by the same rule, so breaks the write path never saw — a parent that never synced, a filtered subtree, a race — still place correctly. The read-time half is cheap because `BELONGS_TO` is already loaded for partitioning. | 2026-09-13 | Q76 |
| 69 | Every item carries its parent's **id, type and name** — the name too, so a search hit can render "in &lt;folder&gt;" without a second lookup; the parent row is already joined for placement. | 2026-09-13 | Q84 |
| 70 | Paging ships the re-traversal baseline. The traversal cannot emit in sort order, so every page re-traverses its partition: cost is flat in the offset (satisfying §3.9 and PERF-07) but the PERF-06 shape — very large directly-granted sets under inaccessible groups — will miss p95 < 1s. That limit is recorded, not pre-optimised; caching a partition's ordered ids against the cursor, and S3's inversion for text search, stay available if real tenants hit it. | 2026-09-13 | — |
| 71 | A placeholder stub **inherits from its parent**. Without it, a stub has neither inheritance nor a grant, so the read rule would hide it and re-place its children — defeating decision 44. With it, the stub is visible exactly when its parent is, its children stay nested under it, and the query needs no placeholder exception. | 2026-09-13 | — |
| 72 | Ancestry over a DAG: a strict node is accessible when **at least one** full hierarchy path from the App is accessible (§3.6). | 2026-09-13 | — |
| 73 | A child that survives its parent's deletion is re-pointed at its record group for **both** hierarchy and inheritance — decisions 53 and 68 covered only the placement half. The edge sweep removes the `INHERIT_PERMISSIONS` edge that pointed at the deleted parent, and since decision 6 makes a nested record inherit from its parent *record*, the survivor would otherwise have no inheritance at all and be invisible unless it holds its own grant. Applies to the single-record delete path as well as the cascade. | 2026-09-13 | — |
| 74 | The `RECORD_RELATION` → `NODE_RELATION` rename changes the **stored** names too, not just the Python identifiers: the Arango collection `recordRelations` becomes `nodeRelations` and the Neo4j relationship type `RECORD_RELATION` becomes `NODE_RELATION`. This requires a phase-5 migration — Arango renames a collection in place, but Neo4j cannot rename a relationship type, so every edge must be recreated under the new type and the old one dropped. Scope is 289 occurrences across 39 files, reaching the Node.js record constants and the integration-test edge validators. | 2026-09-13 | — |
| 75 | The per-node booleans `isStrict` and `hasRestriction` collapse into **one** field, `accessRule`, with values `OPEN` / `STRICT` / `RESTRICTED`. **Semantics are unchanged** — every case in the companion document keeps its outcome — but the fourth combination becomes unrepresentable. Restriction was only ever read on a strict node (decision 3), so `(non-strict, restricted)` was inert; it maps to `OPEN`. That dead state was not free: the guard neutralising it was repeated in the Cypher rule, the AQL rule and the seed query, and was itself untested until a review found AC-19 and AC-14 inexpressible in the fixture. Stored as a string enum, so the Arango validator rejects a fourth value outright; absent reads as `OPEN` (decision 38, MIG-08) and an *unrecognised* value reads as `RESTRICTED`, so corruption fails closed. Deriving the value from the presence of permission edges was considered and rejected: it changes the outcome of 15 cases, 9 of them P0, 13 of those being over-shares — AC-03/AC-04 and AC-14/AC-16 are identical graphs with opposite expectations, which is the proof the field carries information the edges do not. | 2026-09-15 | — |
| 76 | A permission fetch that fails must never be read as "no restrictions". `_fetch_page_permissions` returns `list[Permission] \| None`, where `None` means *could not determine* and `[]` means *definitively unrestricted*. Previously both a non-SUCCESS status and any exception returned `[]`, so a 403, a rate limit or a timeout wrote a READ-restricted page as `STRICT` — inheriting from its space and visible to every member of it. Callers skip the record instead, leaving the last known-good row untouched for the next sync; writing `RESTRICTED` with an empty grant list would hide the node from everyone *and* overwrite a correct document, because upserts merge. | 2026-09-15 | — |
| 77 | The per-hop rule keeps its **three explicit branches**; the three-conjunct compaction is rejected. The compact form — `(accessRule = 'OPEN' OR allowStrict)` and `(inherits OR granted)` and `(accessRule <> 'RESTRICTED' OR (inherits AND granted))` — is exactly equivalent for all three declared values and diverges on a fourth. An unrecognised `accessRule` matches no branch of the shipped rule and is hidden, but it satisfies `accessRule <> 'RESTRICTED'` and is **returned** by the compact form, inverting decision 75's fail-closed reading. No acceptance scenario can detect this, because every fixture node carries a declared value: measured on both engines, both forms return identical sets across the entire acceptance graph, and only a deliberately corrupted node separates them. A rule rewrite gated on scenario coverage would therefore have shipped the leak. A *corrected* compaction, whose first conjunct reads `accessRule = 'OPEN' OR (accessRule IN ('STRICT','RESTRICTED') AND allowStrict)`, is equivalent including the fourth case and is kept in `test_rule_equivalence.py` as the sanctioned target. Arango's validator refuses the corrupt value at write time, so the exposure is Neo4j-only — Neo4j has no schema, which is why the rule and not the storage layer has to be the thing that fails closed. | 2026-09-15 | — |
| 78 | The App fallback (§3.3, decision 13) takes the **narrow** reading. A granted node lists directly under the App only when its own record group is structurally under that App and **cannot be opened**: not by the rule from the App, not by its own grant below a gap, not by a declaration, and not as part of a collection. The node must also be a chain-top, meaning it has no openable parent it already lists under. It never applies beneath a `hideChildren` group, where such a node stays hidden rather than surfacing one level up, and never to a deleted group. "Cannot be opened" is judged against **all** of the user's grantees, not only the one holding the node's own grant. The broad reading, falling back whenever the group is absent from the listing for any reason, was rejected because it defeats `hideChildren`. Amended the same day: first written as "fails the per-hop rule", which listed a granted record inside a granted group below a gap both under that group and under the App, disagreeing with its breadcrumbs (NV-36, NV-39). | 2026-09-15 | — |
| 79 | Browse accepts an optional **`via_parent_id`**, the parent the user navigated through. When it names a real hierarchy parent of the node that the user can access, `parentNode` and the breadcrumbs follow it. Otherwise they follow placement, with the drive location winning over Shared with Me (decision 67). Without it a record with two parents gets one fixed trail and NV-47 cannot pass. The parameter is additive; the cursor carries it, the Node proxy forwards it, and the frontend starts sending it in phase 3. | 2026-09-15 | — |

## 9. Plan outline (follows decision 12)

| Phase | Work | Waits on |
|---|---|---|
| 0 | **Prove the traversal.** Both query texts written and run against a hand-built fixture on throwaway Neo4j 5.26 and Arango 3.12 containers, returning identical results on the AC and NV P0 cases. Nothing else starts until they are frozen. | — |
| 1 | **Write path.** Rename `RECORD_RELATION` → `NODE_RELATION` and widen it to App and record-group endpoints; inheritance to the direct parent; strict and restricted flags; the missing-parent fallback; Shared-with-Me as a second edge; role removal. Connector work: Confluence restricted pages (decision 26), SharePoint inheritance (decision 47), GitLab flags, team Slack (decision 40); removal of the DOMAIN/ANYONE grant types (decision 59); the SharePoint REST `HasUniqueRoleAssignments` read, batched per list (decision 64, §4.10), preceded by the live-tenant check in §4.10. | — |
| 2 | **Read queries.** v2 on Arango and Neo4j — two arms (§3.6), partitions, cursor pagination, breadcrumbs, parent ids — passing the test cases on both backends (S7). | Phase 1 |
| 3 | **Frontend** on the new contract (§3.9). | Phase 2 |
| 4 | Test every remaining connector on the new model. | |
| 5 | Migration/backfill of existing data. | Decision 32 |
| 6 | Move the bypassing callers to v2; remove v1 and rename v2 to its name. | Decision 33 |

## 10. Changelog

- 2026-09-11 — created; §4 filled from a code read of `main`; first round of answers (decisions
  1–12) and questions Q26–Q37.
- 2026-09-11 — second round (decisions 13–23); placement worked cases; target edges; Q38–Q40.
- 2026-09-11 — test-case document created (281 cases); questions Q41–Q69 raised by the cases.
- 2026-09-12 — third round (decisions 24–36): strictness reading (b) with the two-arm query,
  inheritance as a per-node fact, record groups below a gap, flatten scope, edge directions, the
  `permissionModel` declarations, v2 scope, null role on connector items, role removal.
- 2026-09-12 — fourth round (decisions 37–46): restriction flags do not cascade; non-strict
  default; declarations win inside their scope; team Slack stops inheriting; the five grant paths
  and the gate; placeholders shown; search returns every node kind; `hasChildren` inside the
  listing query. The doc conflict from the applied stash was resolved in favour of this version.
- 2026-09-12 — fifth round (decisions 47–51): SharePoint follows the source's inheritance flag;
  the library root folder stays; GitLab group nesting left alone; collection items keep their real
  edges; `isActive` does not affect the hub.
- 2026-09-12 — sixth round (decisions 52–58): collections use explicit grants only; the
  missing-parent fallback; deleted records excluded; Shared with Me as a real second parent; the
  edge rename carrying every relationship type; cursor offsets only for contributing partitions;
  and the full API contract (§3.9). Plan re-ordered to write path → read queries → frontend.
- 2026-09-12 — decisions 24–58 applied to the test-case document: 63 cases moved off `OPEN`, 16
  added, and four (NV-17, NV-18, PG-24, CN-26) changed result because decision 55 makes Shared
  with Me a real second parent. Five questions the cases exposed were added here: Q78–Q82.
- 2026-09-12 — seventh round (decisions 59–62): the DOMAIN and ANYONE grant types are removed
  rather than revived; retired roles read as READER with a later migration; `hasChildren` is
  navigation-aware; SharePoint pages, lists and list items inherit unconditionally. Two premises
  were corrected against the source while recording these: Graph exposes no inheritance flag for
  SharePoint pages (only drive items have `inheritedFrom`), and SharePoint lists *are* synced
  today. §4.9 records what `hideConnector` actually marks.
- 2026-09-13 — decision 63: the hub stops reading `hideConnector` (Q79 closed).
- 2026-09-13 — the SharePoint API was checked against Microsoft's documentation rather than the
  connector's behaviour (§4.10), which overturned two things: Graph's `inheritedFrom` is
  unreliable for SharePoint, so decision 47's mechanism was wrong, and a documented flag *does*
  exist — REST `HasUniqueRoleAssignments` — so decision 62's "no flag exists, inherit
  unconditionally" was wrong too. Decision 64 replaces both mechanisms and supersedes 62. New
  question Q83: what to write when the flag cannot be read.
- 2026-09-13 — decision 65 closes Q83: a single unreadable item is left alone on re-sync and
  written as inheriting on a first sync; a wholesale failure aborts the run rather than marking a
  tenant's content as inheriting. §4.10 gained the two prerequisites — verify the REST path
  against a live tenant, since the existing calls swallow their errors, and reuse the site-level
  principal resolution for item role assignments.
- 2026-09-13 — decisions 66–68 close the last four questions. Group → App inheritance stays, as
  the fallback when a permission-model declaration changes (S12 rejected); a two-parent record is
  returned once in a flat result, under its drive parent when both are reachable; the
  missing-parent fallback runs at write time on delete *and* at read time on a break; and the
  `NODE_RELATION` rename is confirmed, keeping Jira's link types on the one edge rather than
  adding an edge type per relationship. **Phase 1 is unblocked.** One question remains, Q84,
  which touches only the v2 response shape and so belongs to phase 2.
- 2026-09-15 — decisions 75 and 76. The two per-node access booleans collapse into one three-valued
  `accessRule`, with semantics deliberately unchanged, retiring an inert fourth state whose guard
  was duplicated in three query texts and untested in all three. Deriving the value from permission
  edges was evaluated against the companion document and rejected — it flips 15 cases, 9 of them
  P0, and AC-03/AC-04 are identical graphs with opposite expectations. Separately, a failed
  permission fetch no longer reads as "unrestricted": the fetch distinguishes *unknown* from
  *empty*, because the flag is computed at write time from a call that can fail, which made a
  transient 403 enough to expose a restricted page to its whole space. §3.1, §3.2, §3.6 and §3.8
  updated; the companion document's `[S]`/`[N]`/`[R!]` notation now names the three values.
- 2026-09-15 — decision 77. The per-hop rule keeps its three explicit branches. The planned
  three-conjunct compaction was checked before adoption and proved equivalent for every declared
  value but **not** for an undeclared one, where it returns a node the shipped rule hides. The
  check that matters is in `test_rule_equivalence.py`: both forms agree across the whole acceptance
  graph on both engines — the naive one passes that sweep — and a single corrupted node separates
  them. Gating the rewrite on scenario coverage, as planned, would have blessed a fail-open rule.
  Arango's enum refuses the value at write time; Neo4j has no schema, so the traversal itself must
  fail closed.
- 2026-09-13 — implementation plan written and approved, and a review of the traversal against the
  case list corrected this document in three places. **§3.6 arm 1 was wrong**: it applied the
  restriction ternary to every node, contradicting §3.2's non-strict row and hiding AC-19; the
  strict guard is now explicit. **§3.6's "no upward walk"** was too broad — true for listings, false
  for browse (AC-57), which needs one bounded walk that also yields breadcrumbs. And §3.6 gained the
  DAG ancestry rule. §3.8 gained the third `PermissionModel` value, §4.4 the correct fifteen
  relation types, §4.6 the real Arango test files. Decisions 69–72 recorded; §7 is empty.
- 2026-09-15 — decisions 78 and 79. The App fallback takes the narrow reading, so `hideChildren`
  keeps its contents hidden instead of surfacing them at App level. The reading is pinned by
  behavioural tests on both engines. One of those tests exposed a query-scoping hazard: a grant
  check spliced into a caller's query silently bound to the caller's own grantee variable on
  Neo4j, where Arango raised an error. Browse also gains `via_parent_id`, so a record with two
  parents shows the trail the user actually navigated (NV-47).
- 2026-09-15 — decision 78 amended. "Own group accessible" now means the group can be opened by
  any path: the rule from the App, its own grant below a gap, a declaration, or a collection. It
  no longer means the rule passes from the App. The earlier wording listed a granted record inside
  a granted group below a gap in two places, where its breadcrumbs name one.
