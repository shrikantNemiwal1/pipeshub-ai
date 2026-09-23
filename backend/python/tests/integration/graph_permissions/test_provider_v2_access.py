"""`get_knowledge_hub_access_context_v2` against both real engines.

Every other v2 test passes the user's grantees and gated Apps as constants. This
derives them from the membership edges production writes, and pins them to those
same constants, so the rest of the harness is testing the inputs a real request
would actually supply.
"""

import pytest

pytestmark = pytest.mark.integration

ORG = "org-1"
# The constants the rest of the harness passes by hand.
EXPECTED_GRANTEES = {"user-u", "group-g", "role-r", "team-t", "orgnode-1"}
EXPECTED_GATED_APPS = {
    "ex1-app", "ex2-app", "dec-app", "dec-rgl-app", "ex-app",
    "swm-app", "pl-app", "gp-app", "flag-app", "kb-1", "deep-app",
}


@pytest.fixture(params=["neo4j", "arango"])
def provider(request, neo4j_provider, arango_provider):
    return neo4j_provider if request.param == "neo4j" else arango_provider


async def test_grantees_follow_every_membership_path(loaded_graph, provider) -> None:
    """D42: user, group, role, team and org, derived from the membership edges."""
    context = await provider.get_knowledge_hub_access_context_v2("user-u", ORG)
    assert set(context["grantee_ids"]) == EXPECTED_GRANTEES, context


async def test_the_gate_admits_exactly_the_reachable_apps(loaded_graph, provider) -> None:
    """D43, D52, AC-36: gate-app (nothing reaches U) and kb-2 (no grant) stay out."""
    context = await provider.get_knowledge_hub_access_context_v2("user-u", ORG)
    assert set(context["gated_app_ids"]) == EXPECTED_GATED_APPS, sorted(context["gated_app_ids"])


async def test_another_org_reaches_no_apps(loaded_graph, provider) -> None:
    """A user's memberships are not an org boundary; the Apps must belong to the org asked for."""
    context = await provider.get_knowledge_hub_access_context_v2("user-u", "org-elsewhere")
    assert context["gated_app_ids"] == [], context


async def test_an_unknown_user_has_no_access(loaded_graph, provider) -> None:
    context = await provider.get_knowledge_hub_access_context_v2("no-such-user", ORG)
    assert context == {"grantee_ids": [], "gated_app_ids": []}, context


async def test_a_flatten_from_the_derived_context_matches_the_constants(loaded_graph, provider) -> None:
    """End to end: the derived inputs produce the same placement as the harness constants."""
    context = await provider.get_knowledge_hub_access_context_v2("user-u", ORG)

    async def flat(grantees, gated):
        result = await provider.get_knowledge_hub_children_v2(
            user_key="user-u", org_id=ORG, parent_id="pl-app", limit=200,
            grantee_ids=list(grantees), gated_app_ids=list(gated), flatten=True,
        )
        return {row["id"]: row["parentId"] for row in result["partitions"][0]["rows"]}

    derived = await flat(context["grantee_ids"], context["gated_app_ids"])
    assert derived, "pl-app flattened to nothing"
    assert derived == await flat(sorted(EXPECTED_GRANTEES), sorted(EXPECTED_GATED_APPS))


async def test_both_backends_agree_on_the_access_context(
    loaded_graph, neo4j_provider, arango_provider
) -> None:
    """BE-01: the derived grantees and gate must not depend on the store."""
    cypher = await neo4j_provider.get_knowledge_hub_access_context_v2("user-u", ORG)
    aql = await arango_provider.get_knowledge_hub_access_context_v2("user-u", ORG)
    assert set(cypher["grantee_ids"]) == set(aql["grantee_ids"])
    assert set(cypher["gated_app_ids"]) == set(aql["gated_app_ids"])
