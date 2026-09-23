"""SEC-11: another org's content never appears, by any route.

Orgs share graph state — taxonomy, people, external orgs — so an org-blind
lookup is a live risk here rather than a hypothetical one, and it is the reason
`get_user_apps` having no `orgId` predicate mattered. The fixture's org-B
subgraph is shaped exactly like the others and its own user holds real grants
and an app relation: every path that could leak exists, and only `orgId` differs.

The routes a leak could take are separate code: the gate, partition discovery,
the browse walk and global search. Each is checked, because excluding org B from
one proves nothing about the others.
"""

import pytest

from app.connectors.sources.localKB.handlers.kh_search import search_page

pytestmark = pytest.mark.integration

USER = "user-u"
ORG = "org-1"
SECRET = "kh-integration-secret"
GRANTEES = ["user-u", "group-g", "role-r", "team-t", "orgnode-1"]
GATED_APPS = [
    "ex1-app", "ex2-app", "dec-app", "dec-rgl-app", "ex-app",
    "swm-app", "pl-app", "gp-app", "flag-app", "kb-1",
]
ORG_B_IDS = {"orgb-app", "orgb-rg", "orgb-r1"}
ORG_B_NAMES = ("Org B", "confidential")


@pytest.fixture(params=["neo4j", "arango"])
def provider(request, neo4j_provider, arango_provider):
    return neo4j_provider if request.param == "neo4j" else arango_provider


async def _nodes(prov, parent_id, **kwargs):
    return await prov.get_knowledge_hub_children_v2(
        user_key=USER, org_id=ORG, parent_id=parent_id, limit=200,
        grantee_ids=GRANTEES, gated_app_ids=GATED_APPS, **kwargs,
    )


async def test_the_gate_never_admits_another_orgs_app(loaded_graph, provider) -> None:
    """D43 with an `orgId` predicate: user-b's relation is not user-u's, but the
    gate is what stops the App even being a partition."""
    context = await provider.get_knowledge_hub_access_context_v2(USER, ORG)
    assert not ORG_B_IDS & set(context["gated_app_ids"]), context["gated_app_ids"]


async def test_no_partition_is_built_from_another_orgs_content(
    loaded_graph, provider
) -> None:
    """A partition is a fan-out target, so one built here would be searched."""
    partitions = await provider.get_knowledge_hub_partitions_v2(ORG, GATED_APPS)
    named = {p["partitionId"] for p in partitions} | {
        p["appId"] for p in partitions if p["appId"]
    }
    assert not ORG_B_IDS & named, sorted(ORG_B_IDS & named)


@pytest.mark.parametrize("node_id", ["orgb-app", "orgb-rg", "orgb-r1"])
async def test_browsing_another_orgs_node_by_id_is_refused(
    loaded_graph, provider, node_id
) -> None:
    """SEC-02 applies here too: deep-linking a known id must not confirm it
    exists, so the refusal carries no name and no rows."""
    result = await _nodes(provider, node_id)
    assert result["scope"] == {"admitted": False, "nodeId": node_id}, result["scope"]
    assert result["partitions"][0]["rows"] == []


async def test_a_global_search_returns_no_org_b_node_or_name(
    loaded_graph, provider
) -> None:
    """`search_page` derives its own access context, so this is the whole path."""
    page = await search_page(
        provider, user_key=USER, user_id=USER, org_id=ORG, secret=SECRET, limit=500
    )
    returned = {row["id"] for row in page.rows}
    assert not ORG_B_IDS & returned, sorted(ORG_B_IDS & returned)

    names = " ".join((row.get("name") or "") for row in page.rows).lower()
    for fragment in ORG_B_NAMES:
        assert fragment.lower() not in names, fragment


async def test_searching_for_another_orgs_words_matches_nothing(
    loaded_graph, provider
) -> None:
    """A text match must not be a way in: the filter narrows admitted rows and
    never opens a path (§3.4 step 6)."""
    page = await search_page(
        provider, user_key=USER, user_id=USER, org_id=ORG, secret=SECRET,
        limit=500, filters={"search_query": "confidential"},
    )
    assert page.rows == [], [row["id"] for row in page.rows]
    assert page.total == 0


async def test_both_backends_keep_the_org_boundary(
    loaded_graph, neo4j_provider, arango_provider
) -> None:
    """BE-01 for SEC-11: an org boundary enforced on one store only is a leak."""
    cypher = await neo4j_provider.get_knowledge_hub_access_context_v2(USER, ORG)
    aql = await arango_provider.get_knowledge_hub_access_context_v2(USER, ORG)
    assert set(cypher["gated_app_ids"]) == set(aql["gated_app_ids"])
    assert not ORG_B_IDS & set(cypher["gated_app_ids"])
