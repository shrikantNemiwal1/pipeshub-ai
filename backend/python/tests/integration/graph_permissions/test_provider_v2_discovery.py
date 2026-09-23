"""`get_knowledge_hub_partitions_v2` against both real engines.

The partition list is what a global search fans out over, so a missing partition
is a silently missing slice of results and an extra one is wasted work.
"""

import pytest

from app.utils.kh_partitions import APPS_PARTITION_ID

pytestmark = pytest.mark.integration

ORG = "org-1"
GATED_APPS = [
    "ex1-app", "ex2-app", "dec-app", "dec-rgl-app", "ex-app",
    "swm-app", "pl-app", "gp-app", "flag-app", "kb-1", "deep-app",
]

EXPECTED = {
    ("APPS", APPS_PARTITION_ID, None),
    ("GROUP", "ex1-rg1", "ex1-app"), ("GROUP", "ex1-rg2", "ex1-app"),
    ("GROUP", "ex2-rg1", "ex2-app"), ("GROUP", "ex2-rg2", "ex2-app"),
    ("GROUP", "dec-rg1", "dec-app"),
    ("GROUP", "dec-rg2", "dec-rgl-app"), ("GROUP", "dec-rg4", "dec-rgl-app"),
    ("GROUP", "dec-rg5", "dec-rgl-app"),
    ("GROUP", "ex-rg1", "ex-app"), ("GROUP", "ex-hidden", "ex-app"),
    ("GROUP", "swm-drive", "swm-app"), ("GROUP", "swm-inbox", "swm-app"),
    ("GROUP", "swm-ainbox", "swm-app"),
    ("GROUP", "pl-rg1", "pl-app"), ("GROUP", "pl-rg2", "pl-app"), ("GROUP", "pl-rg3", "pl-app"),
    ("GROUP", "gp-rg1", "gp-app"), ("APP_DIRECT", "gp-app", "gp-app"),
    ("GROUP", "flag-rg", "flag-app"), ("GROUP", "deep-rg", "deep-app"),
    ("COLLECTION", "kb-1", "kb-1"),
}


@pytest.fixture(params=["neo4j", "arango"])
def provider(request, neo4j_provider, arango_provider):
    return neo4j_provider if request.param == "neo4j" else arango_provider


def _triples(partitions):
    return {(p["partitionKind"], p["partitionId"], p["appId"]) for p in partitions}


async def test_every_gated_app_is_partitioned_by_its_top_level_groups(loaded_graph, provider) -> None:
    """§3.9, D7: nested groups (pl-rg4 under pl-rg2) stay inside their top-level partition,
    only gp-app has nodes outside every group, and a closed top-level group (pl-rg2,
    dec-rg4) is still a partition because its grants must be found (PG-04)."""
    partitions = await provider.get_knowledge_hub_partitions_v2(ORG, GATED_APPS)
    assert _triples(partitions) == EXPECTED, sorted(_triples(partitions) ^ EXPECTED)
    assert len(partitions) == len(EXPECTED), "a partition was listed twice"


async def test_an_app_outside_the_gate_contributes_no_partition(loaded_graph, provider) -> None:
    """AC-36: gate-app is not passed in, so none of its groups may be searched."""
    partitions = await provider.get_knowledge_hub_partitions_v2(ORG, GATED_APPS)
    assert "gate-rg1" not in {p["partitionId"] for p in partitions}


async def test_another_org_searches_only_the_apps_partition(loaded_graph, provider) -> None:
    partitions = await provider.get_knowledge_hub_partitions_v2("org-elsewhere", GATED_APPS)
    assert _triples(partitions) == {("APPS", APPS_PARTITION_ID, None)}


async def test_both_backends_list_the_same_partitions(
    loaded_graph, neo4j_provider, arango_provider
) -> None:
    """BE-01: a partition missing on one store is a silently missing slice there."""
    cypher = await neo4j_provider.get_knowledge_hub_partitions_v2(ORG, GATED_APPS)
    aql = await arango_provider.get_knowledge_hub_partitions_v2(ORG, GATED_APPS)
    assert cypher == aql
