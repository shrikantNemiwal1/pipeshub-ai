"""Partition mode (`partition="group" | "app_direct"`) against both real engines.

A global search runs one query per partition and merges them (§3.9). Each
partition must return exactly the nodes a flatten of its App would place inside
it, with the same parents, even when the partition root itself is closed
(PG-04), and the partitions of one App must add back up to that App's flatten.
"""

import pytest

from app.connectors.sources.localKB.handlers.kh_merge import PartitionFeed, merge_pages

pytestmark = pytest.mark.integration

USER = "user-u"
ORG = "org-1"
GRANTEES = ["user-u", "group-g", "role-r", "team-t", "orgnode-1"]
GATED_APPS = [
    "ex1-app", "ex2-app", "dec-app", "dec-rgl-app", "ex-app",
    "swm-app", "pl-app", "gp-app", "flag-app", "kb-1",
]


@pytest.fixture(params=["neo4j", "arango"])
def provider(request, neo4j_provider, arango_provider):
    return neo4j_provider if request.param == "neo4j" else arango_provider


async def _run(prov, parent_id, *, partition=None, flatten=False, **kwargs):
    return await prov.get_knowledge_hub_children_v2(
        user_key=USER, org_id=ORG, parent_id=parent_id, limit=200,
        grantee_ids=GRANTEES, gated_app_ids=GATED_APPS,
        partition=partition, flatten=flatten, **kwargs,
    )


def _placed(result) -> dict[str, str]:
    assert len(result["partitions"]) == 1, result["partitions"]
    return {row["id"]: row["parentId"] for row in result["partitions"][0]["rows"]}


async def _group(prov, group_id, **kwargs) -> dict[str, str]:
    result = await _run(prov, group_id, partition="group", **kwargs)
    assert result["partitions"][0]["partitionKind"] == "GROUP"
    return _placed(result)


async def test_a_closed_group_partition_still_finds_the_grants_inside_it(
    loaded_graph, provider
) -> None:
    """PG-04, PG-06, PG-41: pl-rg2 cannot be opened, so it is never a row, but
    what is granted inside it is found and placed as the App's flatten places it."""
    assert await _group(provider, "pl-rg2") == {
        "pl-r9": "pl-app", "pl-rg4": "pl-app", "pl-r14": "pl-rg4", "pl-r15": "pl-rg4",
    }


@pytest.mark.parametrize(
    "group, expected",
    [
        ("pl-rg1", {"pl-rg1": "pl-app", "pl-r5": "pl-rg1", "pl-r6": "pl-rg1", "pl-r11": "pl-rg1"}),
        ("pl-rg3", {"pl-rg3": "pl-app", "pl-r12": "pl-rg3"}),
        ("dec-rg1", {"dec-rg1": "dec-app", "dec-r1": "dec-rg1"}),
        ("dec-rg2", {"dec-rg2": "dec-rgl-app", "dec-r2": "dec-rg2",
                     "dec-rg3": "dec-rg2", "dec-r3": "dec-rg3"}),
        ("dec-rg4", {}),
        ("dec-rg5", {}),
        ("ex-hidden", {"ex-hidden": "ex-app"}),
    ],
    ids=["open-group", "group-granted-to-a-group", "AC-68-app-level", "AC-69-declared",
         "undeclared-closed", "declared-ungranted", "hidden-channel"],
)
async def test_an_open_group_partition_lists_its_root_under_the_app(
    loaded_graph, provider, group, expected
) -> None:
    """PG-41: the root once, under its App, then its subtree as a flatten places it."""
    assert await _group(provider, group) == expected


async def test_both_shared_with_me_partitions_find_the_record_and_the_merge_keeps_one(
    loaded_graph, provider
) -> None:
    """PG-24: swm-x is in both partitions; the merge keeps it once, under its drive parent."""
    drive = await _run(provider, "swm-drive", partition="group")
    inbox = await _run(provider, "swm-inbox", partition="group")
    assert _placed(drive) == {
        "swm-drive": "swm-app", "swm-f1": "swm-drive", "swm-x": "swm-f1",
        # A chain-top placed by its own group, which in this partition is the root.
        "swm-z": "swm-drive",
    }
    assert _placed(inbox) == {"swm-inbox": "swm-app", "swm-x": "swm-inbox", "swm-y": "swm-inbox"}

    merged = merge_pages(
        [PartitionFeed("swm-drive", "GROUP", iter(drive["partitions"][0]["rows"])),
         PartitionFeed("swm-inbox", "GROUP", iter(inbox["partitions"][0]["rows"]))],
        limit=50,
    )
    placed = {row["id"]: row["parentId"] for row in merged.rows}
    assert [row["id"] for row in merged.rows].count("swm-x") == 1
    assert placed["swm-x"] == "swm-f1", placed


@pytest.mark.parametrize(
    "app, expected",
    [("gp-app", {"gp-direct": "gp-app"}), ("pl-app", {})],
    ids=["one-direct-record", "none"],
)
async def test_the_app_direct_partition_holds_only_what_no_group_holds(
    loaded_graph, provider, app, expected
) -> None:
    """§3.9: nodes with no top-level group get one partition per App."""
    result = await _run(provider, app, partition="app_direct")
    assert result["partitions"][0]["partitionKind"] == "APP_DIRECT"
    assert _placed(result) == expected


@pytest.mark.parametrize(
    "app, groups",
    [
        ("pl-app", ["pl-rg1", "pl-rg2", "pl-rg3"]),
        ("swm-app", ["swm-drive", "swm-inbox", "swm-ainbox"]),
        ("gp-app", ["gp-rg1"]),
    ],
)
async def test_an_apps_partitions_add_up_to_its_flatten(loaded_graph, provider, app, groups) -> None:
    """Partitioning must neither lose nor re-place a node: every node of the App's
    flatten is in some partition with the same parent. A two-parent node may sit in
    two partitions; its flatten parent must be among them."""
    flat = _placed(await _run(provider, app, flatten=True))
    seen: dict[str, set[str]] = {}
    for placed in [await _group(provider, g) for g in groups] + [
        _placed(await _run(provider, app, partition="app_direct"))
    ]:
        for node_id, parent_id in placed.items():
            seen.setdefault(node_id, set()).add(parent_id)
    assert set(seen) == set(flat), f"partitions={sorted(seen)}\nflatten={sorted(flat)}"
    for node_id, parent_id in flat.items():
        assert parent_id in seen[node_id], f"{node_id}: flatten {parent_id}, partitions {seen[node_id]}"


async def test_a_partition_applies_filters(loaded_graph, provider) -> None:
    """PG-41 with node_types: the closed root never appears, its granted group does."""
    assert set(await _group(provider, "pl-rg2", node_types=["recordGroup"])) == {"pl-rg4"}


async def test_an_unknown_partition_kind_is_refused(provider) -> None:
    with pytest.raises(ValueError, match="unknown partition kind"):
        await _run(provider, "pl-rg1", partition="app")


@pytest.mark.parametrize("group", ["pl-rg1", "pl-rg2", "swm-inbox", "swm-drive", "dec-rg2"])
async def test_both_backends_partition_alike(
    loaded_graph, neo4j_provider, arango_provider, group
) -> None:
    """BE-01: partition mode is where a dialect divergence would hide longest."""
    cypher = (await _run(neo4j_provider, group, partition="group"))["partitions"][0]["rows"]
    aql = (await _run(arango_provider, group, partition="group"))["partitions"][0]["rows"]
    assert cypher, f"{group} partition returned nothing"
    assert [r["id"] for r in cypher] == [r["id"] for r in aql]
    for left, right in zip(cypher, aql):
        for field in ("name", "parentId", "parentName", "parentIsInternal", "nodeType",
                      "hasChildren", "userRole", "sortKey", "nullRank"):
            assert left[field] == right[field], f"{left['id']}.{field}: {left[field]!r} vs {right[field]!r}"
