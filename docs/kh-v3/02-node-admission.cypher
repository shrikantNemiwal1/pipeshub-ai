// ===========================================================================
// KH v3 — Step 2: may the user open this node at all?
// ---------------------------------------------------------------------------
// The gate before anything else. If the answer is no, the caller returns 404
// with a constant body and does no further work — it must not reveal the
// node's name, type, or that it exists (SEC-02).
//
// Order of checks, cheapest first:
//   1. the app gate            — nothing outside a reachable app is ever visible
//   2. the app's declaration   — APP_LEVEL opens everything in it
//   3. the group's declaration — a GRANTED RECORD_GROUP_LEVEL group opens
//                                everything in it
//   4. a direct grant on the node
//   5. an unbroken rule-passing chain from the app
//
// 2 and 3 short-circuit: when a declaration applies, the per-node check is
// skipped entirely. Declared groups are common — 310 of 516 on the real store
// — so 3 avoids the walk whenever the user actually holds the group.
//
// The app gate is enforced by the caller before this query runs. It is kept
// here anyway: it costs one list lookup, and it is the boundary that keeps a
// share inside a connector the user does not have invisible (gate-r3), and a
// grant from another org out (orgb-app).
//
// The start node may be an App, a RecordGroup, a folder or a record. All four
// are handled; $parent_type says which, matching the API's own vocabulary.
//
// Parameters
//   $user_key    : User.id
//   $parent_id   : id of the node being opened
//   $parent_type : 'app' | 'recordGroup' | 'folder' | 'record'
//   $org_id      : organisation the request is scoped to
//   $types       : hierarchy edge kinds, ['PARENT_CHILD', 'ATTACHMENT']
// ===========================================================================

MATCH (u:User {id: $user_key})

// --- Who this user counts as ----------------------------------------------
OPTIONAL MATCH (u)-[:PERMISSION {type: 'USER'}]->(principal:Group|Role|Teams)
WITH u, collect(DISTINCT principal.id) AS viaMembership

OPTIONAL MATCH (u)-[:BELONGS_TO]->(userOrg:Organization)
WITH u, viaMembership, collect(DISTINCT userOrg.id) AS viaOrg

WITH u, [u.id] + viaMembership + viaOrg AS granteeIds

// --- The app gate, in one statement ---------------------------------------
// An app is reachable two ways, and both are collected here rather than in
// separate passes: the user is connected to it (USER_APP_RELATION), or any
// grantee holds a PERMISSION on it. Nothing outside a gated app is visible,
// however it was granted — a direct share inside a connector the user does
// not have must stay invisible (case gate-r3).
UNWIND granteeIds AS granteeId
MATCH (grantee:User|Group|Role|Teams|Organization {id: granteeId})
OPTIONAL MATCH (grantee)-[:PERMISSION|USER_APP_RELATION]->(gatedApp:App)
WHERE gatedApp.orgId = $org_id
WITH u, granteeIds, collect(DISTINCT gatedApp.id) AS gatedAppIds

// --- The node being opened ------------------------------------------------
MATCH (target:Record|RecordGroup|App {id: $parent_id})
WHERE NOT coalesce(target.isDeleted, false)
  AND ( ($parent_type = 'app'         AND target:App)
     OR ($parent_type = 'recordGroup' AND target:RecordGroup)
     OR ($parent_type IN ['folder', 'record'] AND target:Record) )

// --- Its app and its record group -----------------------------------------
// By property, not by traversal. Verified on the real store: connectorId,
// recordGroupId and orgId are present on all 16,394 records, and connectorId
// on all 516 groups. connectorId holds the App's own id.
OPTIONAL MATCH (ownerApp:App)
WHERE ownerApp.id = CASE WHEN target:App THEN target.id ELSE target.connectorId END
OPTIONAL MATCH (ownerGroup:RecordGroup)
WHERE ownerGroup.id = CASE WHEN target:RecordGroup THEN target.id
                           WHEN target:Record      THEN target.recordGroupId
                           ELSE null END

WITH u, granteeIds, gatedAppIds, target, ownerApp, ownerGroup,
     coalesce(ownerApp.id, '') IN gatedAppIds AS appIsGated

// --- The group's declaration ----------------------------------------------
// RECORD_GROUP_LEVEL means "everything in this group is open to whoever can
// reach the GROUP" — so the declaration on its own admits nobody. The group
// itself must be granted to one of the grantees first, or a declaration on a
// space the user cannot reach would hand them its contents.
//
// A grant on the group, not a full admission of it — that would recurse. A
// declared group is the unit that gets shared, so the grant is where its
// access actually lives.
//
// APP_LEVEL needs no equivalent grant check: reaching the app IS the access it
// is declared against, and the gate has already established that.
//
// Tested positively against the exact value. permissionModel is ABSENT on 206
// of 516 groups and 4 of 14 apps, and an absent model must mean "no
// declaration, fall through" — never a default. Anything that passed on null
// here would open a whole connector.
OPTIONAL MATCH (groupGrantee)-[:PERMISSION]->(ownerGroup)
WHERE groupGrantee.id IN granteeIds
WITH granteeIds, gatedAppIds, target, ownerApp, ownerGroup, appIsGated,
     count(groupGrantee) AS groupGrants

WITH granteeIds, gatedAppIds, target, ownerApp, ownerGroup, appIsGated, groupGrants,
     ( appIsGated
       AND coalesce(ownerApp.permissionModel, '') = 'APP_LEVEL' ) AS openedByApp,
     ( appIsGated
       AND coalesce(ownerGroup.permissionModel, '') = 'RECORD_GROUP_LEVEL'
       AND groupGrants > 0 ) AS openedByGroup

// --- Direct grant on the node ---------------------------------------------
// Only counts inside a gated app: the gate beats a direct share.
OPTIONAL MATCH (directGrantee)-[:PERMISSION]->(target)
WHERE directGrantee.id IN granteeIds
WITH granteeIds, gatedAppIds, target, ownerApp, ownerGroup, appIsGated,
     groupGrants, openedByApp, openedByGroup,
     count(directGrantee) AS directGrants

WITH granteeIds, gatedAppIds, target, ownerApp, ownerGroup, appIsGated,
     groupGrants, openedByApp, openedByGroup, directGrants,
     ( appIsGated AND directGrants > 0 ) AS isGranted

// --- Otherwise: an unbroken rule-passing chain from the app ---------------
// Skipped when a declaration already settled it. Every hop must pass, so
// strictness comes for free — a STRICT node only survives if the chain above
// it did too.
OPTIONAL MATCH ancestry = (app:App)
      ( (parent)-[hop:NODE_RELATION]->(child)
        WHERE hop.relationshipType IN $types
          AND NOT coalesce(child.isDeleted, false)
          AND NOT coalesce(parent.hideChildren, false)
          AND (
                ( coalesce(child.accessRule, 'OPEN') = 'RESTRICTED'
                  AND EXISTS { (child)-[:INHERIT_PERMISSIONS]->(parent) }
                  AND EXISTS { (cg)-[:PERMISSION]->(child) WHERE cg.id IN granteeIds } )
             OR ( coalesce(child.accessRule, 'OPEN') IN ['STRICT', 'OPEN']
                  AND ( EXISTS { (child)-[:INHERIT_PERMISSIONS]->(parent) }
                        OR EXISTS { (cg)-[:PERMISSION]->(child) WHERE cg.id IN granteeIds } ) )
              )
      )+ (target)
WHERE NOT (openedByApp OR openedByGroup OR isGranted)
  AND app.id IN gatedAppIds

WITH granteeIds, gatedAppIds, target, ownerApp, ownerGroup, appIsGated,
     groupGrants, openedByApp, openedByGroup, directGrants, isGranted,
     count(ancestry) AS ancestryPaths

WITH granteeIds, gatedAppIds, target, ownerApp, ownerGroup, appIsGated,
     groupGrants, openedByApp, openedByGroup, directGrants, isGranted,
     ancestryPaths, ancestryPaths > 0 AS isFromApp

RETURN
  target.id                        AS nodeId,
  ownerApp.id                      AS appId,
  ownerGroup.id                    AS recordGroupId,
  ( (target:App AND appIsGated)
    OR openedByApp OR openedByGroup OR isGranted OR isFromApp ) AS admitted,

  // --- which arm answered ---------------------------------------------------
  appIsGated                       AS appIsGated,
  openedByApp                      AS viaAppDeclaration,
  openedByGroup                    AS viaGroupDeclaration,
  isGranted                        AS viaDirectGrant,
  isFromApp                        AS viaAppChain,

  // --- intermediate values, so a FALSE can be told apart from a vacuous FALSE
  // A refusal is only trustworthy if the inputs behind it were non-empty: a
  // node with 0 grantees or 0 gated apps is refused for want of data, not by
  // the model. These columns make that distinction visible instead of implied.
  size(granteeIds)                             AS granteeCount,
  size(gatedAppIds)                            AS gatedAppCount,
  [l IN labels(target)
     WHERE l IN ['App','RecordGroup','Record']][0] AS nodeLabel,
  coalesce(ownerApp.permissionModel,   '<absent>') AS appModel,
  coalesce(ownerGroup.permissionModel, '<absent>') AS groupModel,
  coalesce(target.accessRule,          '<absent>') AS targetAccessRule,
  coalesce(ownerGroup.hideChildren, false)         AS groupHidesChildren,
  groupGrants                      AS groupGrantEdges,
  directGrants                     AS directGrantEdges,
  ancestryPaths                    AS ancestryPaths;


// ===========================================================================
// Open questions
// ===========================================================================
//
// 1. SETTLED. A declared group now also requires a grant on the GROUP ITSELF,
//    closing the leak where a RECORD_GROUP_LEVEL declaration on an unreachable
//    space handed out its contents to anyone whose app was gated. Reaching the
//    app is no longer sufficient to open a declared group.
//
//    APP_LEVEL is unchanged and still checked here. It needs no equivalent
//    grant: the app is the thing it is declared against, and the gate already
//    establishes that the user reaches it.
//
//    Left open deliberately: "has access to the group" is read here as a GRANT
//    on the group, not a full admission of it. A group reachable only by
//    inheritance, with no grant anywhere on it, will not fire its own
//    declaration and falls through to the walk below. That is a
//    refusal-to-short-circuit, not a refusal — the walk still admits the node
//    if the chain holds — so it costs latency, never access. Worth revisiting
//    only if profiling shows declared groups missing the fast path often.
//
// 2. STRICT AND OPEN SHARE A BRANCH now (`IN ['STRICT','OPEN']`) because their
//    conditions are identical once every hop is checked. RESTRICTED stays
//    separate: it needs inheritance AND a grant, and collapsing it would turn
//    the AND into an OR — the classic over-share, pinned by AC-16.
//
// 3. BOTH ENDS OF THE WALK ARE STILL BOUND. `target` by id, `app` by the gated
//    list. That is the shape that made 5.26 enumerate every path rather than
//    probe for one (39,768–79,536 rows per site in the current code). The
//    declarations short-circuit means it runs far less often, but when it does
//    run it may still be expensive. Profile before trusting it.
//
// 4. SEEDS BELOW A GAP are still not covered — a granted node under a folder
//    you cannot open. That is a denial, not a leak, and belongs in step 3.
//
// 5. BREADCRUMBS ARE **NOT** TAKEN FROM THIS WALK. Decided: keep this walk a
//    plain existence check and give breadcrumbs their own computation later.
//
//    Reusing `ancestry` looked attractive — `nodes(p)` returns App -> target in
//    order and renders as-is (measured, 8 deep on Confluence) — but it fails on
//    both counts that matter:
//
//    (a) It would cost the skip. This walk only runs when nothing cheaper
//        settled admission; measured, 3 of 4 real cases never enter
//        Repeat(Trail) at all. A breadcrumb is needed on every rendered row, so
//        sharing the walk forces it to run always — paying the dearest operator
//        in the query on exactly the rows that had avoided it.
//    (b) It would not work anyway. A declaration-admitted node has NO
//        rule-passing path: measured, a Slack record admitted by
//        RECORD_GROUP_LEVEL returns 0 paths, because its records neither
//        inherit from their group nor hold a grant — the declaration stands in
//        for both. With 310 of 516 groups declared, that is the common row, and
//        for it this walk has nothing to hand back.
//
//    So the trail needs a structural parent-walk of its own, with per-ancestor
//    admission (an unreachable ancestor must be replaced, never named — NV-28,
//    SEC-01). Two facts already measured for whoever builds it: there are ZERO
//    orphans (0 of 16,394 records, 0 of 516 groups), so a parent chain always
//    exists; and 30 nodes carry two hierarchy parents (all Slack), so it needs
//    a deterministic tie-break or the same node renders differently run to run.
// ===========================================================================
