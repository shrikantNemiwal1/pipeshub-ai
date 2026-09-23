// ===========================================================================
// KH v3 — Step 3b: everything a user can see inside one connector
// ---------------------------------------------------------------------------
// Step 3a lists the children of a node the user already holds. That is enough
// only while every visible node hangs below an unbroken chain from the App.
// It does not, and the gap is the whole reason this file exists:
//
//   a user can hold a grant on a record inside a folder they cannot open.
//
// SharePoint does this routinely. A rule-applying descent stops at the folder
// and never reaches the record, so the record silently disappears — the user
// sees an empty folder and assumes sync is broken. Measured on the real store,
// that is not an edge case: for Jira the recovered records are 944 of 954
// visible ids, for GitLab 2 they are 12,927 of 12,931. Without this file those
// two connectors show almost nothing.
//
// THREE ANCHORS, ONE UNION
// A node is visible if a rule-passing walk reaches it from any anchor:
//
//   A. the App itself        — the ordinary hierarchy, rule at every hop
//   B. a declared group      — RECORD_GROUP_LEVEL *and granted*; everything
//                              beneath it is open, so the walk is structural
//   C. a seed                — a granted OPEN node the App-rooted walk missed,
//                              i.e. one sitting below a gap
//
// Anchors instead of one clever traversal is the point. "Checking is switched
// off below a declared group" is a property of the *path*, and a quantified
// path pattern cannot carry state down its own walk. Making the declared group
// an anchor in its own right says the same thing with no state at all.
//
// WHY C IS RESTRICTED TO **OPEN**
// The three access rules differ in exactly one respect:
//
//   OPEN        : inherits OR granted                          (no ancestry test)
//   STRICT      : ancestors-reachable AND (inherits OR granted)
//   RESTRICTED  : ancestors-reachable AND inherits AND granted
//
// A seed sits below a gap by definition, so "ancestors-reachable" is already
// false for it. Only OPEN can legitimately be a seed. Admitting a granted
// STRICT or RESTRICTED node here would hand out its whole subtree on half the
// evidence — the same defect as the grant arm in step 2, but amplified.
// For the same reason the walk *below* a seed refuses anything but OPEN.
//
// WHAT THIS FILE ASSUMES ITS CALLER HAS DONE
//   * the connector is one the user may use — the gate is applied upstream
//   * $app_id belongs to $org_id, checked here as a belt-and-braces line
//
// Parameters
//   $user_key : User.id
//   $org_id   : organisation the request is scoped to
//   $app_id   : the connector to compute the visible set for
// ===========================================================================

MATCH (u:User {id: $user_key})

// --- Who this user counts as ----------------------------------------------
OPTIONAL MATCH (u)-[:PERMISSION {type: 'USER'}]->(principal:Group|Role|Teams)
WITH u, collect(DISTINCT principal.id) AS viaMembership

OPTIONAL MATCH (u)-[:BELONGS_TO]->(userOrg:Organization)
WITH u, viaMembership, collect(DISTINCT userOrg.id) AS viaOrg

WITH [u.id] + viaMembership + viaOrg AS granteeIds

// --- The gate --------------------------------------------------------------
// The caller is expected to have narrowed to connectors the user may use, but
// this query must not DEPEND on that: handed an arbitrary $app_id it would
// otherwise return the connector's contents to anyone in the org. That is
// case gate-r3, and it is a leak, so the gate is recomputed here rather than
// assumed. One list lookup.
UNWIND granteeIds AS granteeId
MATCH (grantee:User|Group|Role|Teams|Organization {id: granteeId})
OPTIONAL MATCH (grantee)-[:PERMISSION|USER_APP_RELATION]->(gatedApp:App)
WHERE gatedApp.orgId = $org_id
WITH granteeIds, collect(DISTINCT gatedApp.id) AS gatedAppIds

// --- The connector ---------------------------------------------------------
MATCH (app:App {id: $app_id})
WHERE app.orgId = $org_id
  AND app.id IN gatedAppIds

// A KB has no permissionModel of its own; type = 'KB' is the marker, and it
// means the same thing APP_LEVEL does — everyone who reaches the connector
// sees everything in it.
WITH granteeIds, app,
     ( coalesce(app.permissionModel, '') = 'APP_LEVEL'
       OR app.type = 'KB' ) AS appOpensEverything

// ===========================================================================
// ANCHOR A — the App, walking the ordinary hierarchy
// ===========================================================================
// Bounded at 50 hops (BE-04). Real depth on this store reaches 13, so the
// bound is slack, not a limit — but an unbounded quantifier over a graph that
// may contain a cycle is a hang waiting to happen.
OPTIONAL MATCH pathA = (app)
      ( (pa)-[ha:NODE_RELATION]->(ca)
        WHERE NOT coalesce(ca.isDeleted, false)
          AND NOT coalesce(pa.hideChildren, false)
          AND ( appOpensEverything
             OR ( coalesce(ca.accessRule, 'OPEN') = 'RESTRICTED'
                  AND EXISTS { (ca)-[:INHERIT_PERMISSIONS]->(pa) }
                  AND EXISTS { (ga)-[:PERMISSION]->(ca) WHERE ga.id IN granteeIds } )
             OR ( coalesce(ca.accessRule, 'OPEN') IN ['STRICT', 'OPEN']
                  AND ( EXISTS { (ca)-[:INHERIT_PERMISSIONS]->(pa) }
                     OR EXISTS { (ga)-[:PERMISSION]->(ca) WHERE ga.id IN granteeIds } ) ) )
      ){1,50} (na)
WITH granteeIds, app, appOpensEverything,
     collect(DISTINCT na.id) AS regionA

// ===========================================================================
// ANCHOR B — declared groups, and everything beneath them
// ===========================================================================
// RECORD_GROUP_LEVEL says "whoever can REACH this group sees everything in it".
// Two routes count as reaching it, and an earlier version accepted only the
// first:
//
//   * one of the grantees holds a grant on the group, or
//   * the ordinary walk already arrived — which is what regionA *is*.
//
// Requiring the grant skipped 213 of the 310 declared groups on the real store.
// Every Slack channel is declared and reached by inheritance, never by a direct
// grant; the split is total, 97 granted-and-not-inheriting against 213
// inheriting-and-not-granted, with no group in both.
OPTIONAL MATCH (dg:RecordGroup)
WHERE dg.connectorId = app.id
  AND NOT coalesce(dg.isDeleted, false)
  AND NOT coalesce(dg.hideChildren, false)
  AND dg.permissionModel = 'RECORD_GROUP_LEVEL'
  AND ( dg.id IN regionA
     OR EXISTS { (gd)-[:PERMISSION]->(dg) WHERE gd.id IN granteeIds } )
WITH granteeIds, app, appOpensEverything, regionA,
     collect(DISTINCT dg.id) AS declaredIds

// A group nested under a declared group is covered by that declaration whether
// or not it carries a permissionModel of its own, and its records belong to IT
// rather than to the declared ancestor: 13,803 records sit in nested groups and
// only 14 of them also point at the parent. So the anchor set has to be closed
// downward before membership is read, or those records are simply lost.
//
// The closure walks GROUPS only — at most 516 nodes, against 16,394 if it ran
// over records.
OPTIONAL MATCH (db:RecordGroup)
      ( (pb)-[hb:NODE_RELATION]->(cb:RecordGroup)
        WHERE NOT coalesce(cb.isDeleted, false)
          AND NOT coalesce(pb.hideChildren, false)
      ){1,50} (nb:RecordGroup)
WHERE db.id IN declaredIds
WITH granteeIds, app, appOpensEverything, regionA, declaredIds,
     collect(DISTINCT nb.id) AS nestedDeclaredIds

WITH granteeIds, app, appOpensEverything, regionA,
     apoc.coll.toSet(declaredIds + nestedDeclaredIds) AS declaredScope

// Membership, not traversal. A record's BELONGS_TO points at its owning group
// however deep it sits inside folders, so one indexed hop replaces a 50-deep
// walk — measured equal on all 12 GitLab 2 declared groups, including one of
// 6,228 records.
//
// It also admits a record that does NOT inherit from its group, which the walk
// refuses. That is the point: a declaration is supposed to cover exactly those.
// Only 2 such records exist today, but they are the ones the declaration is for.
OPTIONAL MATCH (mb:Record)-[:BELONGS_TO]->(mg:RecordGroup)
WHERE mg.id IN declaredScope
  AND NOT coalesce(mb.isDeleted, false)
WITH granteeIds, app, appOpensEverything, regionA, declaredScope,
     collect(DISTINCT mb.id) AS belowDeclared

// ===========================================================================
// ANCHOR C — seeds: granted nodes the App-rooted walk never reached
// ===========================================================================
// The diff is the whole trick. A granted node the ordinary walk already found
// needs nothing further — its subtree came with it. Only the ones it missed
// are below a gap, and only those need a walk of their own. That keeps the
// work proportional to what the user can see rather than to the graph.
//
// OPEN only, for the reason in the header. `hideChildren` on the seed's own
// parent is honoured here; an ancestor further up that hides its children is
// NOT yet checked — see the open questions.
OPTIONAL MATCH (sd)
WHERE (sd:Record OR sd:RecordGroup)
  AND sd.connectorId = app.id
  AND NOT coalesce(sd.isDeleted, false)
  AND coalesce(sd.accessRule, 'OPEN') = 'OPEN'
  AND NOT sd.id IN regionA
  AND NOT sd.id IN belowDeclared
  AND NOT sd.id IN declaredScope
  AND EXISTS { (gs)-[:PERMISSION]->(sd) WHERE gs.id IN granteeIds }
WITH granteeIds, app, appOpensEverything, regionA, declaredScope, belowDeclared,
     collect(DISTINCT sd.id) AS seedIds

// Below a seed, ancestry is already broken, so STRICT and RESTRICTED both fail
// their first condition and only OPEN survives. pl-r13 in the fixture is
// exactly this case and is expected to stay out.
OPTIONAL MATCH pathC = (sc)
      ( (pc)-[hc:NODE_RELATION]->(cc)
        WHERE NOT coalesce(cc.isDeleted, false)
          AND NOT coalesce(pc.hideChildren, false)
          AND coalesce(cc.accessRule, 'OPEN') = 'OPEN'
          AND ( EXISTS { (cc)-[:INHERIT_PERMISSIONS]->(pc) }
             OR EXISTS { (gc)-[:PERMISSION]->(cc) WHERE gc.id IN granteeIds } )
      ){1,50} (nc)
WHERE sc.id IN seedIds
WITH app, appOpensEverything, regionA, declaredScope, belowDeclared, seedIds,
     collect(DISTINCT nc.id) AS belowSeeds

// ===========================================================================
// The union
// ===========================================================================
// The App's own id is included: a partition rooted at the connector needs it,
// and leaving it out 404s the very node the caller just opened.
WITH app, appOpensEverything, regionA, declaredScope, belowDeclared, seedIds, belowSeeds,
     apoc.coll.toSet([app.id] + regionA + declaredScope + belowDeclared + seedIds + belowSeeds) AS visibleIds

RETURN
  app.id                        AS appId,
  appOpensEverything            AS appOpensEverything,
  size(visibleIds)              AS visibleCount,

  // --- where each id came from, so a wrong total can be attributed ---------
  size(regionA)                 AS fromAppWalk,
  size(declaredScope)           AS declaredGroups,
  size(belowDeclared)           AS belowDeclaredGroups,
  size(seedIds)                 AS seeds,
  size(belowSeeds)              AS belowSeedWalk,

  visibleIds                    AS visibleIds;


// ===========================================================================
// Open questions
// ===========================================================================
//
// 1. A SEED UNDER A HIDDEN ANCESTOR IS NOT FILTERED. The walks honour
//    hideChildren hop by hop, but anchor C jumps straight to the seed, so a
//    grant deep inside a hidden subtree still surfaces. The agreed fix is the
//    isHidden flag extended to Record and RecordGroup — which does not exist
//    on those labels yet and nothing propagates it. Vacuous on this store
//    today: 0 of the 608 children under the 43 hidden Slack groups are granted
//    to anyone. Latent, not safe.
//
// 2. CHAIN TOPS ARE NOT COLLAPSED. If a seed sits below another seed, both are
//    in seedIds and both are walked. The ids are identical either way — the
//    second walk is subsumed by the first — so this costs duplicate work, not
//    correctness. It matters for PLACEMENT, which is a separate query: only
//    the topmost seed should be rendered under the connector.
//
// 3. THIS RETURNS A SET, NOT A LISTING. No ordering, no keyset, no exact
//    total, no filters. Deliberate: the set is what browse and flatten both
//    consume, and PG-32's exact-total union is what exhausted the 1.4 GiB
//    transaction cap in v2. Design the listing on top, not inside.
//
// 4. BOTH B AND C SCAN BY PROPERTY. `dg.connectorId = app.id` and
//    `sd.connectorId = app.id` are property filters over RecordGroup and
//    Record. Indexed on this store, but profile before trusting it — an
//    unindexed variant would scan 16,394 records per connector.
// ===========================================================================
