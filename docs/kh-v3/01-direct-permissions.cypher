// ===========================================================================
// KH v3 — Step 1: direct access
// ---------------------------------------------------------------------------
// Every Record, RecordGroup and App the user reaches DIRECTLY — either by a
// PERMISSION edge (their own, or one held by a group, role, team or
// organisation they belong to), or, for Apps, by being connected to the
// connector at all (USER_APP_RELATION).
//
// Both routes are collected here so the app gate is computed once. Later
// steps consume this rather than recomputing it.
//
// This is only the "granted" half of visibility. Most of what a user can see,
// they see by inheritance from a parent — that comes in a later step. This
// query answers one question: what has been handed to them explicitly.
//
// Parameters
//   $user_key : User.id
//   $org_id   : organisation the request is scoped to
// ===========================================================================

MATCH (u:User {id: $user_key})

// --- Who this user counts as ----------------------------------------------
// Membership is a PERMISSION edge tagged type = 'USER', pointing at the
// Group / Role / Teams node. Collected before the next OPTIONAL MATCH so the
// two lookups don't multiply into a cartesian product.
//
// Fixed labels and property equalities go in the pattern; only expressions
// (list membership, coalesce, negation) need a WHERE. On an OPTIONAL MATCH
// both forms behave identically — a WHERE attached to it is part of the
// optional pattern, not a filter on its result — so this is for clarity and
// to hand the planner the labels directly.
OPTIONAL MATCH (u)-[:PERMISSION {type: 'USER'}]->(principal:Group|Role|Teams)
WITH u, collect(DISTINCT principal.id) AS viaMembership

// The organisation the user belongs to, which can itself hold grants.
// Deliberately NOT Group: measured on the real store, all 82 user BELONGS_TO
// edges carry entityType = 'ORGANIZATION' and not one points at a Group, so
// group membership arrives solely through the typed PERMISSION edge above.
// (v1 read user->Group BELONGS_TO and v2 kept it defensively; nothing writes
// it. If a connector ever starts to, those memberships would be lost here.)
OPTIONAL MATCH (u)-[:BELONGS_TO]->(userOrg:Organization)
WITH u, viaMembership, collect(DISTINCT userOrg.id) AS viaOrg

// The user themself is always a grantee.
WITH [u.id] + viaMembership + viaOrg AS granteeIds

// --- What those principals reach directly ---------------------------------
// Two edge kinds in one pattern. PERMISSION can point at a Record, a
// RecordGroup or an App. USER_APP_RELATION only ever points at an App and only
// ever leaves the User — but the user is themself a grantee, so the same
// grantee-driven match picks it up, and non-user grantees simply match none.
// Matching both together avoids a second pass over the same principals.
//
// Driven from the grantees outward, not from the nodes inward.
//
// Written the obvious way -- MATCH (grantee)-[:PERMISSION]->(node) WHERE
// grantee.id IN granteeIds -- the planner has no way into an unlabelled
// `grantee`, so it starts from `node` instead. Profiled on the real store that
// is a UnionNodeByLabelsScan across every Record, RecordGroup and App (16,924
// nodes), an isDeleted read on each, then every incoming PERMISSION edge
// expanded: 123,705 db hits on the expand alone, ~141,000 in total, to return
// 253 rows. Roughly 95% of the work is discarded by the final filter.
//
// UNWIND with a labelled id match turns the list membership into one index
// seek per grantee (29 here) and expands outward from those.
UNWIND granteeIds AS granteeId
MATCH (grantee:User|Group|Role|Teams|Organization {id: granteeId})
MATCH (grantee)-[edge:PERMISSION|USER_APP_RELATION]->(node:Record|RecordGroup|App)
WHERE NOT coalesce(node.isDeleted, false)

RETURN
  node.id AS nodeId,
  CASE
    WHEN node:App         THEN 'app'
    WHEN node:RecordGroup THEN 'recordGroup'
    ELSE                       'record'
  END AS nodeType,
  // 'PERMISSION' — granted to the user or to a principal they belong to.
  // 'USER_APP_RELATION' — the connector is simply connected for this user.
  // The two mean different things to later steps, so they are kept apart
  // rather than flattened into one "has access" flag.
  type(edge) AS via,
  // Null for USER_APP_RELATION: a connection carries no role.
  edge.role  AS role
ORDER BY nodeType, nodeId;


// ===========================================================================
// Four things to decide before this is final
// ===========================================================================
//
// 1. ORG SCOPING. Nothing here filters by $org_id yet, so $org_id is declared
//    and unused. Today org isolation is enforced later, at the App gate
//    (`app.orgId = $org_id`). Two options:
//      (a) leave it — a grant in another org is harmless until that org's App
//          is gated in, and the gate is what stops it;
//      (b) filter here too, belt and braces.
//    (b) is safer but needs orgId to be reliably set on Record and
//        RecordGroup, which I have not verified — only App is confirmed.
//    Org isolation is a P0 case, so I would rather you choose than guess.
//
// 2. MEMBERSHIP EDGE TYPES. This requires `membership.type = 'USER'` on the
//    PERMISSION edge to a Group/Role/Teams. Verified on the real store:
//    134 Group, 37 Role, 30 Teams edges, every one carrying type = 'USER',
//    and no user→Group BELONGS_TO edges exist at all. So on current data the
//    typed path is the only one that matters — but a connector writing an
//    untyped membership edge would silently lose access.
//
// 3. APPS ARE TWO DIFFERENT THINGS. This returns Apps reached by a PERMISSION
//    grant. An App is also reachable by USER_APP_RELATION — the connector
//    being connected for you — which is a different idea: the gate. Both feed
//    "which apps can this user use", so we need to decide whether step 1
//    covers grants only (as here) or the gate as well.
//
// 4. ROLE. Returned because collection items inherit the role held on the
//    collection (Reader / Writer / Owner), so it is needed downstream. Drop it
//    if you would rather keep step 1 to bare ids.
//
// ===========================================================================
// Expected shape on the current tenant: ~209 rows for a user with 29
// grantees — small, which is the point. Everything else a user can see comes
// from inheritance, not from this list.
// ===========================================================================
