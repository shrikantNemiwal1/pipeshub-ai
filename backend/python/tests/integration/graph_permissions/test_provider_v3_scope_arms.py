"""v3's scope arms, on a graph shaped the way production actually is.

**Why this module exists.** `get_knowledge_hub_connector_page_v3` builds its
visible set from seven arms, and five of them key on ``connectorId = app.id``:

    arm 2  WHERE dg.connectorId = app.id       -> declaredIds
    arm 3  nested groups, driven by declaredIds
    arm 4  belowDeclared,  driven by declaredScope
    arm 5  WHERE sd.connectorId = app.id       -> seedIds
    arm 6  belowSeeds,     driven by seedIds

The shared acceptance fixture cannot satisfy that predicate. Measured directly
off ``fixture_graph.build_fixture()``:

    RecordGroups: 24, with connectorId: 0      (``rg()`` sets connectorName,
                                                never connectorId)
    Records: 75 of 81 carry a connectorId that is not any App id
             ("drive-conn", "confluence-conn", ...)

So for 12 of its 14 apps those five arms are **structurally dead** -- they
cannot match a single node -- and the suite still passes because arm 1 (the App
walk) happens to cover most of the fixture's cases. On the real store the
invariant does hold: 16,445 of 16,445 Records and 516 of 516 RecordGroups have
``connectorId = App.id``.

This module therefore builds its own small graph with the production convention
so those arms actually run. `test_the_fixture_can_exercise_the_declared_arms`
guards the premise itself, so this module cannot quietly rot into the same hole.

Assertions are against **known values**, not against the other backend: two
engines generated from one set of builders share that set's mistakes, and the
acceptance suite has already been burned by a parity assertion that held while
both sides were wrong.
"""

from __future__ import annotations

import pytest

from .fixture_graph import (
    DRIVE,
    ORG,
    USER_U,
    _principals,
    app,
    bt,
    ip,
    nr,
    perm,
    rec,
    rg,
    user_app,
)
from .loaders import load_into_neo4j

APP = "v3-app"

# Everything in this connector carries connectorId = the App's own id, which is
# what the real store does and what arms 2-6 require.
GROUP = dict(group_type="DRIVE", connector=DRIVE, connectorId=APP)
ITEM = dict(record_type="FILE", connector_id=APP)

# The set the rules say USER_U may see in this connector.
#
#   v3-open         arm 1: inherits from the App
#   v3-dec          arm 1 (granted) and arm 2 (declared + granted)
#   v3-dec-r1/r2    arm 4: BELONGS_TO a declared group
#   v3-dec-nested   arm 3: a group below a declared group
#   v3-nested-r1    arm 4: BELONGS_TO a group in the declared scope
#   v3-seed         arm 5: granted OPEN node below a gap
#   v3-seed-child   arm 6: inherits from a seed
#   v3-hidden       arm 1 (granted); its CONTENTS are hidden, it is not
CORRECT_VISIBLE = {
    "v3-open", "v3-dec", "v3-dec-r1", "v3-dec-r2", "v3-dec-nested",
    "v3-nested-r1", "v3-seed", "v3-seed-child", "v3-hidden",
}

# Currently returned but should not be -- see
# `test_a_node_from_another_org_does_not_leak`. Kept in the expected set so the
# rest of this module asserts today's behaviour exactly; the xfail below is
# what carries the finding.
KNOWN_CROSS_ORG_LEAK = {"v3-otherorg-r1"}

EXPECTED_VISIBLE = CORRECT_VISIBLE | KNOWN_CROSS_ORG_LEAK

EXPECTED_HIDDEN = {
    "v3-gap",                  # the walk stops here: no inheritance, no grant
    "v3-undeclared",           # reachable by an edge, but nothing opens it
    "v3-undec-r1",             # ... so its contents stay out
    "v3-dec-ungranted",        # declared, but granted to nobody
    "v3-dec-ungranted-r1",
    "v3-hidden-r1",            # below hideChildren
    # Deleted, on each arm that could otherwise admit them.
    "v3-dec-deleted",          # arm 4: BELONGS_TO a declared group
    "v3-nested-deleted",       # arm 3: a deleted group below the declaration
    "v3-nested-deleted-r1",    # ... and its contents
    "v3-seed-deleted",         # arm 5: a granted node below the gap
}

# A node carrying this connector's id but ANOTHER org's orgId. The query gates
# the App on `app.orgId = $org_id` and never re-checks orgId on the nodes it
# collects, so this is the shape that would leak if a sync ever wrote a record
# with the wrong org. Asserted separately so its outcome is reported rather
# than folded into the exact-set test.
OTHER_ORG_NODE = "v3-otherorg-r1"


def _graph() -> tuple[list, list]:
    principal_nodes, principal_edges = _principals()
    nodes = principal_nodes + [
        app(APP, "V3 Connector", connector=DRIVE, app_group="Google Workspace"),

        rg("v3-open", "Open group", **GROUP),
        rec("v3-gap", "Gap folder", **ITEM),
        rec("v3-seed", "Granted below gap", **ITEM),
        rec("v3-seed-child", "Child of the seed", **ITEM),

        rg("v3-dec", "Declared group",
           permissionModel="RECORD_GROUP_LEVEL", **GROUP),
        rec("v3-dec-r1", "Declared item 1", **ITEM),
        rec("v3-dec-r2", "Declared item 2", **ITEM),
        rg("v3-dec-nested", "Nested under the declaration", **GROUP),
        rec("v3-nested-r1", "Nested item", **ITEM),

        rg("v3-undeclared", "Undeclared group", **GROUP),
        rec("v3-undec-r1", "Item in an undeclared group", **ITEM),

        rg("v3-dec-ungranted", "Declared, granted to nobody",
           permissionModel="RECORD_GROUP_LEVEL", **GROUP),
        rec("v3-dec-ungranted-r1", "Item in an ungranted declaration", **ITEM),

        rg("v3-hidden", "Declared group that hides its children",
           permissionModel="RECORD_GROUP_LEVEL", hideChildren=True, **GROUP),
        rec("v3-hidden-r1", "Item below hideChildren", **ITEM),

        # Deleted nodes, one per arm that could otherwise admit them.
        rec("v3-dec-deleted", "Deleted item in a declared group",
            isDeleted=True, **ITEM),
        rg("v3-nested-deleted", "Deleted group below the declaration",
           isDeleted=True, **GROUP),
        rec("v3-nested-deleted-r1", "Item in a deleted nested group", **ITEM),
        rec("v3-seed-deleted", "Deleted grant below the gap",
            isDeleted=True, **ITEM),

        # Right connector, wrong org.
        rec(OTHER_ORG_NODE, "Record belonging to another org",
            orgId="org-2", **ITEM),
    ]
    edges = principal_edges + [
        user_app(USER_U, APP),

        # Arm 1: the App walk reaches v3-open because it inherits.
        nr(APP, "v3-open"), ip("v3-open", APP),

        # A gap: v3-gap neither inherits nor is granted, so the walk stops at
        # it -- and everything below it is reachable only as a seed.
        nr("v3-open", "v3-gap"),
        nr("v3-gap", "v3-seed"), perm(USER_U, "v3-seed"),
        nr("v3-seed", "v3-seed-child"), ip("v3-seed-child", "v3-seed"),

        # Arms 2/3/4: a declared group, its BELONGS_TO contents, and a nested
        # group that carries the declaration down.
        nr(APP, "v3-dec"), perm(USER_U, "v3-dec"),
        bt("v3-dec-r1", "v3-dec"), bt("v3-dec-r2", "v3-dec"),
        nr("v3-dec", "v3-dec-nested"), bt("v3-nested-r1", "v3-dec-nested"),

        # Neither declared nor openable: contents must stay out.
        nr(APP, "v3-undeclared"), bt("v3-undec-r1", "v3-undeclared"),

        # Declared but granted to nobody: a declaration opens only a group the
        # user can already open.
        nr(APP, "v3-dec-ungranted"),
        bt("v3-dec-ungranted-r1", "v3-dec-ungranted"),

        # Granted, so the group itself is visible; hideChildren keeps its
        # contents out.
        nr(APP, "v3-hidden"), perm(USER_U, "v3-hidden"),
        bt("v3-hidden-r1", "v3-hidden"),

        # Deleted, attached exactly where a live node would be admitted.
        bt("v3-dec-deleted", "v3-dec"),
        nr("v3-dec", "v3-nested-deleted"),
        bt("v3-nested-deleted-r1", "v3-nested-deleted"),
        nr("v3-gap", "v3-seed-deleted"), perm(USER_U, "v3-seed-deleted"),

        # Wrong org, but granted and carrying this connector's id.
        nr(APP, OTHER_ORG_NODE), perm(USER_U, OTHER_ORG_NODE),
    ]
    return nodes, edges


@pytest.fixture(scope="module")
async def v3_graph(neo4j_provider, neo4j_settings):
    """Neo4j only -- v3 has no Arango implementation."""
    nodes, edges = _graph()
    await load_into_neo4j(neo4j_settings, nodes, edges)
    return {"nodes": nodes, "edges": edges}


async def _visible(provider, **kwargs) -> set[str]:
    access = await provider.get_knowledge_hub_access_v3(
        user_key=USER_U, org_id=ORG,
    )
    page = await provider.get_knowledge_hub_connector_page_v3(
        app_id=APP, org_id=ORG,
        grantee_ids=access["grantee_ids"],
        gated_app_ids=access["gated_app_ids"],
        granted_ids=access["by_connector"].get(APP) or [],
        limit=500, flatten=True, sort_field="name", sort_dir="ASC",
        include_total=True, **kwargs,
    )
    return {row["id"] for row in page["rows"]}


def test_the_fixture_can_exercise_the_declared_arms() -> None:
    """Guard the premise: every node here must satisfy `connectorId = app.id`.

    Without this the module would still pass while testing nothing -- which is
    exactly the state the shared acceptance fixture is in.
    """
    nodes, _ = _graph()
    groups = [n for n in nodes if n["kind"] == "RecordGroup"]
    records = [n for n in nodes if n["kind"] == "Record"]
    assert groups and records
    for node in groups + records:
        assert node["props"].get("connectorId") == APP, node["id"]


async def test_the_whole_visible_set(v3_graph, neo4j_provider) -> None:
    """The exact set, so an extra node is a failure and not just a miss."""
    assert await _visible(neo4j_provider) == EXPECTED_VISIBLE


async def test_nothing_out_of_reach_leaks(v3_graph, neo4j_provider) -> None:
    assert (await _visible(neo4j_provider)) & EXPECTED_HIDDEN == set()


async def test_a_declared_groups_contents_are_reached_by_belongs_to(
    v3_graph, neo4j_provider
) -> None:
    """Arms 2 and 4. The items hold no grant and inherit from nothing; the
    RECORD_GROUP_LEVEL declaration on their group is what admits them."""
    visible = await _visible(neo4j_provider)
    assert {"v3-dec-r1", "v3-dec-r2"} <= visible


async def test_a_nested_group_carries_the_declaration_down(
    v3_graph, neo4j_provider
) -> None:
    """Arm 3, then arm 4: the nested group is admitted by the declaration above
    it, and its own BELONGS_TO contents come with it."""
    visible = await _visible(neo4j_provider)
    assert "v3-dec-nested" in visible
    assert "v3-nested-r1" in visible


async def test_a_granted_node_below_a_gap_is_visible(
    v3_graph, neo4j_provider
) -> None:
    """Arm 5. The App walk cannot pass `v3-gap`, so the only way in is the seed."""
    visible = await _visible(neo4j_provider)
    assert "v3-seed" in visible
    assert "v3-gap" not in visible, "the gap itself is not admissible"


async def test_content_below_a_seed_is_visible(v3_graph, neo4j_provider) -> None:
    """Arm 6: an OPEN node inheriting from a seed comes with it."""
    assert "v3-seed-child" in await _visible(neo4j_provider)


async def test_an_undeclared_groups_contents_stay_out(
    v3_graph, neo4j_provider
) -> None:
    """A hierarchy edge from the App is not access: nothing opens this group."""
    visible = await _visible(neo4j_provider)
    assert "v3-undeclared" not in visible
    assert "v3-undec-r1" not in visible


async def test_a_declaration_granted_to_nobody_opens_nothing(
    v3_graph, neo4j_provider
) -> None:
    """A declaration opens only a group the user can already open."""
    visible = await _visible(neo4j_provider)
    assert "v3-dec-ungranted" not in visible
    assert "v3-dec-ungranted-r1" not in visible


async def test_hidechildren_keeps_contents_out_but_shows_the_group(
    v3_graph, neo4j_provider
) -> None:
    """The group is granted, so it is visible; `hideChildren` stops the descent
    and also excludes it from the declared scope."""
    visible = await _visible(neo4j_provider)
    assert "v3-hidden" in visible
    assert "v3-hidden-r1" not in visible


async def test_deleted_nodes_are_excluded_from_every_arm(
    v3_graph, neo4j_provider
) -> None:
    """Each arm carries its own `isDeleted` guard, and so does the listing.
    One deleted node is planted at each admitting position."""
    visible = await _visible(neo4j_provider)
    for node_id in ("v3-dec-deleted", "v3-nested-deleted",
                    "v3-nested-deleted-r1", "v3-seed-deleted"):
        assert node_id not in visible, node_id


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Cross-org over-share: the v3 scope block gates the App on "
        "`app.orgId = $org_id` and then never re-checks orgId on the nodes it "
        "collects -- arms 2 and 5 key on `connectorId = app.id` alone. A node "
        "carrying this connector's id, a direct grant and ANOTHER org's orgId "
        "is returned. Latent rather than live on the dev store (all 16,445 "
        "records and 516 groups match their app's org, and there is one "
        "Organization), but it is the tenant boundary resting on a single "
        "predicate. Fixing it means adding an orgId predicate to the arms, "
        "which changes what a node with a NULL orgId does -- a permission-model "
        "decision, not a cleanup. strict=True so this fails loudly if it is "
        "ever fixed and the xfail should be removed."
    ),
)
async def test_a_node_from_another_org_does_not_leak(
    v3_graph, neo4j_provider
) -> None:
    """This node carries the right connectorId and a direct grant, and belongs
    to a different org. It should not be in the page."""
    assert OTHER_ORG_NODE not in await _visible(neo4j_provider)


async def test_the_total_matches_the_rows(v3_graph, neo4j_provider) -> None:
    """PG-32: the total counts the whole result, not the page."""
    access = await neo4j_provider.get_knowledge_hub_access_v3(
        user_key=USER_U, org_id=ORG,
    )
    page = await neo4j_provider.get_knowledge_hub_connector_page_v3(
        app_id=APP, org_id=ORG,
        grantee_ids=access["grantee_ids"],
        gated_app_ids=access["gated_app_ids"],
        granted_ids=access["by_connector"].get(APP) or [],
        limit=3, flatten=True, sort_field="name", sort_dir="ASC",
        include_total=True,
    )
    assert page["total"] == len(EXPECTED_VISIBLE)
    assert len(page["rows"]) == 3
    assert sum(page["counts"].values()) == page["total"]


async def test_a_user_with_no_grants_sees_nothing_here(
    v3_graph, neo4j_provider
) -> None:
    """The app is gated to USER_U only, so USER_V gets an empty page rather
    than the connector's contents."""
    from .fixture_graph import USER_V

    access = await neo4j_provider.get_knowledge_hub_access_v3(
        user_key=USER_V, org_id=ORG,
    )
    page = await neo4j_provider.get_knowledge_hub_connector_page_v3(
        app_id=APP, org_id=ORG,
        grantee_ids=access["grantee_ids"],
        gated_app_ids=access["gated_app_ids"],
        granted_ids=access["by_connector"].get(APP) or [],
        limit=500, flatten=True, sort_field="name", sort_dir="ASC",
        include_total=True,
    )
    assert page["rows"] == []
