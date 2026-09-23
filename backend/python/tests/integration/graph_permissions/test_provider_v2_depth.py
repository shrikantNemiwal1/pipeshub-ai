"""BE-04: a chain deeper than v1's bound still opens, lists and traces.

`_KH_V2_MAX_UP_DEPTH` is 50 rather than v1's `_KNOWLEDGE_HUB_INHERIT_MAX_DEPTH`
of 20 precisely so a deep record stays reachable, and nothing exercised it. The
failure mode is the quiet one: a bound set too low raises nothing, the upward
walk simply finds no path, `admitted` comes back false, and a node the user may
legitimately open answers 404. Content appears to stop existing past a depth.

The chain has its own App and inherits from it, so it holds no grant, adds no
seed, and leaves every other App's pinned traversal set alone.
"""

import pytest

from .fixture_graph import DEEP_CHAIN_LENGTH

pytestmark = pytest.mark.integration

USER = "user-u"
ORG = "org-1"
GRANTEES = ["user-u", "group-g", "role-r", "team-t", "orgnode-1"]
GATED_APPS = [
    "ex1-app", "ex2-app", "dec-app", "dec-rgl-app", "ex-app",
    "swm-app", "pl-app", "gp-app", "flag-app", "kb-1", "deep-app",
]
DEEPEST = f"deep-{DEEP_CHAIN_LENGTH}"


@pytest.fixture(params=["neo4j", "arango"])
def provider(request, neo4j_provider, arango_provider):
    return neo4j_provider if request.param == "neo4j" else arango_provider


async def _nodes(prov, parent_id, **kwargs):
    return await prov.get_knowledge_hub_children_v2(
        user_key=USER, org_id=ORG, parent_id=parent_id, limit=200,
        grantee_ids=GRANTEES, gated_app_ids=GATED_APPS, **kwargs,
    )


def _ids(result) -> set[str]:
    assert len(result["partitions"]) == 1, result["partitions"]
    return {row["id"] for row in result["partitions"][0]["rows"]}


async def test_the_deepest_node_is_admitted(loaded_graph, provider) -> None:
    """The 404 signal, at depth. `admitted` false here is indistinguishable to a
    user from the node not existing."""
    result = await _nodes(provider, DEEPEST)
    assert result["scope"]["admitted"] is True, result["scope"]
    assert result["scope"]["nodeId"] == DEEPEST


async def test_the_deepest_nodes_breadcrumbs_span_the_whole_chain(
    loaded_graph, provider
) -> None:
    """The upward walk yields the trail in the same query, so the bound governs
    both admission and the breadcrumbs — a short walk truncates the trail rather
    than failing."""
    scope = (await _nodes(provider, DEEPEST))["scope"]
    trail = [crumb["id"] for crumb in scope["breadcrumbs"]]

    expected = ["deep-app", "deep-rg"] + [
        f"deep-{level}" for level in range(1, DEEP_CHAIN_LENGTH + 1)
    ]
    assert trail == expected, trail
    assert len(trail) == DEEP_CHAIN_LENGTH + 2


async def test_each_link_lists_the_next(loaded_graph, provider) -> None:
    """Depth does not change placement: every level lists exactly its successor."""
    assert _ids(await _nodes(provider, "deep-rg")) == {"deep-1"}
    for level in range(1, DEEP_CHAIN_LENGTH):
        listed = _ids(await _nodes(provider, f"deep-{level}"))
        assert listed == {f"deep-{level + 1}"}, (level, listed)
    assert _ids(await _nodes(provider, DEEPEST)) == set()


async def test_a_flatten_reaches_the_bottom_of_the_chain(
    loaded_graph, provider
) -> None:
    """The downward bound (`_KH_V2_MAX_DEPTH`) is the mirror of the upward one:
    a flatten that stops early loses the tail silently, with a full-looking page.
    """
    placed = {
        row["id"]: row["parentId"]
        for row in (await _nodes(provider, "deep-rg", flatten=True))["partitions"][0]["rows"]
    }
    assert len(placed) == DEEP_CHAIN_LENGTH, sorted(placed)
    assert placed["deep-1"] == "deep-rg"
    assert placed[DEEPEST] == f"deep-{DEEP_CHAIN_LENGTH - 1}"


async def test_both_backends_reach_the_same_depth(
    loaded_graph, neo4j_provider, arango_provider
) -> None:
    """BE-01 for BE-04: a per-dialect bound would diverge here and nowhere else."""
    cypher = (await _nodes(neo4j_provider, DEEPEST))["scope"]
    aql = (await _nodes(arango_provider, DEEPEST))["scope"]
    assert cypher["admitted"] == aql["admitted"] is True
    assert [c["id"] for c in cypher["breadcrumbs"]] == [c["id"] for c in aql["breadcrumbs"]]
