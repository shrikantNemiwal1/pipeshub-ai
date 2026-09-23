// ===========================================================================
// KH v3 — Step 3a: the direct children of an already-admitted node (browse)
// ---------------------------------------------------------------------------
// Step 2 has already decided the user may open $parent_id. This lists what is
// underneath it, one hop down. Flatten is a separate file: it walks the whole
// subtree, and the seed rules that only matter at depth belong with it.
//
// Two modes, chosen by one boolean:
//
//   skipChecks = the app is APP_LEVEL
//                OR the node's own group is RECORD_GROUP_LEVEL *and granted*
//
//   A. skipChecks true  — the declaration already covers everything in this
//      subtree, so children are listed structurally with no per-node test.
//      This is the common path: 310 of 516 groups on the real store are
//      declared. It is also the cheap path — no grant lookup, no inheritance
//      probe, no traversal beyond the one hop.
//
//   B. skipChecks false — every child is tested on its own.
//
// The per-child rule is step 2's rule, applied across exactly one hop, so a
// child is judged against THIS parent and no other:
//
//      RESTRICTED : inherits from this parent AND is granted
//      STRICT     : inherits from this parent OR  is granted
//      OPEN       : inherits from this parent OR  is granted
//
// STRICT and OPEN are identical here on purpose. Strictness is a statement
// about ancestors, and the ancestors were settled by step 2 when it admitted
// the parent — there is nothing left for this hop to add.
//
// WHAT THIS FILE DELIBERATELY DOES NOT DO
// An invisible child is returned with visible = false rather than dropped.
// It must not be rendered, but it also must not prune anything: a folder the
// user cannot open may still contain files they can (the SharePoint case). The
// caller drops the false rows for now; when flatten and seeds land, those rows
// are what the descent continues through. Dropping them here would bake in the
// very pruning that loses seeds below a gap.
//
// Parameters
//   $user_key    : User.id
//   $parent_id   : the node being opened — already admitted by step 2
//   $parent_type : 'app' | 'recordGroup' | 'folder' | 'record'
//   $org_id      : organisation the request is scoped to
// ===========================================================================

MATCH (u:User {id: $user_key})

// --- Who this user counts as (same as steps 1 and 2) -----------------------
OPTIONAL MATCH (u)-[:PERMISSION {type: 'USER'}]->(principal:Group|Role|Teams)
WITH u, collect(DISTINCT principal.id) AS viaMembership

OPTIONAL MATCH (u)-[:BELONGS_TO]->(userOrg:Organization)
WITH u, viaMembership, collect(DISTINCT userOrg.id) AS viaOrg

WITH u, [u.id] + viaMembership + viaOrg AS granteeIds

// --- The app gate ----------------------------------------------------------
UNWIND granteeIds AS granteeId
MATCH (grantee:User|Group|Role|Teams|Organization {id: granteeId})
OPTIONAL MATCH (grantee)-[:PERMISSION|USER_APP_RELATION]->(gatedApp:App)
WHERE gatedApp.orgId = $org_id
WITH granteeIds, collect(DISTINCT gatedApp.id) AS gatedAppIds

// --- The node being listed -------------------------------------------------
MATCH (start:Record|RecordGroup|App {id: $parent_id})
WHERE NOT coalesce(start.isDeleted, false)
  AND ( ($parent_type = 'app'         AND start:App)
     OR ($parent_type = 'recordGroup' AND start:RecordGroup)
     OR ($parent_type IN ['folder', 'record'] AND start:Record) )

// --- Does a declaration switch checking off for this subtree? --------------
// Repeated from step 2 rather than passed in, so this file stands alone.
OPTIONAL MATCH (startApp:App)
WHERE startApp.id = CASE WHEN start:App THEN start.id ELSE start.connectorId END
OPTIONAL MATCH (startGroup:RecordGroup)
WHERE startGroup.id = CASE WHEN start:RecordGroup THEN start.id
                           WHEN start:Record      THEN start.recordGroupId
                           ELSE null END

OPTIONAL MATCH (groupGrantee)-[:PERMISSION]->(startGroup)
WHERE groupGrantee.id IN granteeIds
WITH granteeIds, gatedAppIds, start, startApp, startGroup,
     count(groupGrantee) AS groupGrants

WITH granteeIds, start, startApp, startGroup, groupGrants,
     coalesce(startApp.id, '') IN gatedAppIds AS appIsGated

WITH granteeIds, start, startApp, startGroup, groupGrants, appIsGated,
     ( appIsGated
       AND ( coalesce(startApp.permissionModel, '') = 'APP_LEVEL'
          OR ( coalesce(startGroup.permissionModel, '') = 'RECORD_GROUP_LEVEL'
               AND groupGrants > 0 ) ) ) AS skipChecks

// --- The children, structurally -------------------------------------------
// No relationshipType filter, per your call. Worth knowing what that admits:
// on the real store NODE_RELATION is 16,558 PARENT_CHILD and 382 ATTACHMENT,
// plus 9 BLOCKS, 1 RELATED, 1 CLONES and 2 with the property absent. The last
// four kinds are Jira-style links, not containment, so without a filter a
// "blocks" link makes the blocked issue a child of the blocker. 11 edges
// tenant-wide — negligible now, but it grows with Jira usage.
//
// hideChildren is on the PARENT: a hidden node has no listable children at
// all. 43 groups carry it on the real store, covering 608 children.
MATCH (start)-[rel:NODE_RELATION]->(child)
WHERE NOT coalesce(child.isDeleted, false)
  AND NOT coalesce(start.hideChildren, false)

// --- Is this child granted to the user? ------------------------------------
OPTIONAL MATCH (childGrantee)-[:PERMISSION]->(child)
WHERE childGrantee.id IN granteeIds
WITH granteeIds, start, skipChecks, rel, child,
     count(childGrantee) AS childGrants

// --- Does it inherit through THIS parent? ----------------------------------
// Per (child, parent) pair, never "does it inherit from anything". A node can
// hang off two parents and inherit through only one of them; testing the wrong
// pair is how a second hierarchy parent turns into an unearned grant.
WITH start, skipChecks, rel, child, childGrants,
     EXISTS { (child)-[:INHERIT_PERMISSIONS]->(start) } AS inheritsFromParent

RETURN
  child.id                         AS nodeId,
  [l IN labels(child)
     WHERE l IN ['App','RecordGroup','Record']][0] AS nodeLabel,

  ( skipChecks
    OR CASE coalesce(child.accessRule, 'OPEN')
         WHEN 'RESTRICTED' THEN (inheritsFromParent AND childGrants > 0)
         ELSE                    (inheritsFromParent OR  childGrants > 0)
       END )                      AS visible,

  // --- why, and the raw inputs behind it -----------------------------------
  skipChecks                       AS viaDeclaration,
  inheritsFromParent               AS inheritsFromParent,
  childGrants                      AS grantEdges,
  coalesce(child.accessRule, '<absent>')        AS accessRule,
  coalesce(child.hideChildren, false)           AS hidesOwnChildren,
  coalesce(rel.relationshipType, '<absent>')    AS edgeKind
ORDER BY nodeLabel, nodeId;


// ===========================================================================
// Open questions
// ===========================================================================
//
// 1. AN INVISIBLE CHILD WITH VISIBLE DESCENDANTS. The SharePoint case, and the
//    reason this query returns visible = false instead of dropping the row.
//    Three ways to render it, and they differ in what the user can reach:
//      (a) drop it      — the descendants become unreachable by navigation,
//                         findable only by search or flatten;
//      (b) show a stub  — the node appears un-openable so you can pass through
//                         it. `isPlaceholder` already exists on records and may
//                         be exactly this mechanism;
//      (c) re-place the descendants — surface them under the nearest ancestor
//                         the user CAN see. v2 does this: "own group, then App
//                         fallback" (D78, PG-41).
//    v2 chose (c). It needs deciding before flatten, because flatten's output
//    is what re-placement rearranges.
//
// 2. COLLECTIONS ARE NOT COVERED. KB items hang off their App by BELONGS_TO,
//    not NODE_RELATION, so this MATCH finds none of them. Latent on this
//    tenant, not absent: all 16,394 records here have BOTH routes
//    (onlyBelongs = 0), but all four KB apps hold zero records, so the case is
//    untested rather than impossible. Needs a second arm over BELONGS_TO.
//
// 3. hasChildren IS NOT RETURNED. The UI needs it to decide what expands, and
//    it cannot be derived from this result: a child with zero VISIBLE children
//    may still have invisible ones with visible descendants below them. It is
//    a structural count for most nodes but a BELONGS_TO count for KB Apps.
//
// 4. NO PAGINATION. No ORDER BY beyond a stable id sort, no keyset, no limit,
//    no exact total. PG-32 requires the exact total, which forces collect()
//    before LIMIT — that is what exhausted the 1.4 GiB transaction cap on the
//    12,931-id connector in v2. Worth designing rather than inheriting.
// ===========================================================================
