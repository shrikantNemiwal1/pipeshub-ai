"""Contract details the other v2 modules prove only by agreement, or not at all.

Three things sat uncovered until the frontend was about to depend on them:

* **`parentType` had no value assertion.** The cross-engine parity tests compare
  it field-for-field, which proves the two engines *agree* — not that either is
  right. Both emitting null passes parity, and that is exactly how every browse
  row came back nameless on both backends earlier in this work.
* **`only_containers` and `record_group_ids`** were proven only at builder level
  (the generated text contains the right clause), never against a store. The
  sidebar sends `only_containers` on every expand.
* **Available filters** were unit-tested against a mocked provider, so nothing
  showed the service's gate composing with the real listing query (PG-34).
"""

import logging

import pytest

from app.connectors.sources.localKB.handlers.knowledge_hub_service import (
    KnowledgeHubService,
)

pytestmark = pytest.mark.integration

USER = "user-u"
ORG = "org-1"
GRANTEES = ["user-u", "group-g", "role-r", "team-t", "orgnode-1"]
GATED_APPS = [
    "ex1-app", "ex2-app", "dec-app", "dec-rgl-app", "ex-app",
    "swm-app", "pl-app", "gp-app", "flag-app", "kb-1", "deep-app",
]


@pytest.fixture(params=["neo4j", "arango"])
def provider(request, neo4j_provider, arango_provider):
    return neo4j_provider if request.param == "neo4j" else arango_provider


async def _browse(prov, parent_id, **kwargs):
    return await prov.get_knowledge_hub_children_v2(
        user_key=USER, org_id=ORG, parent_id=parent_id, limit=100,
        grantee_ids=GRANTEES, gated_app_ids=GATED_APPS, **kwargs,
    )


def _rows(result) -> dict[str, dict]:
    assert len(result["partitions"]) == 1, result["partitions"]
    return {row["id"]: row for row in result["partitions"][0]["rows"]}


# ------------------------------------------------------------- the parent triple

@pytest.mark.parametrize(
    "parent, child, expected_type",
    [
        ("kb-1", "kb-f1", "app"),
        ("kb-f1", "kb-f2", "record"),
        ("pl-rg1", "pl-r5", "recordGroup"),
    ],
    ids=["app-parent", "record-parent", "group-parent"],
)
async def test_rows_name_their_parents_type(
    loaded_graph, provider, parent, child, expected_type
) -> None:
    """D69, and the value — not just cross-engine agreement on it.

    A client navigates to the parent as `/nodes/{parentType}/{parentId}`, so an
    id without a type is not addressable. All three types appear here because a
    folder is a Record in this model, a collection item's parent is the App
    itself, and a connector record's parent is its record group.
    """
    rows = _rows(await _browse(provider, parent))
    assert child in rows, sorted(rows)
    assert rows[child]["parentId"] == parent, rows[child]
    assert rows[child]["parentType"] == expected_type, rows[child]


async def test_the_parent_triple_is_complete_on_every_row(loaded_graph, provider) -> None:
    """Whatever a row's placement, the id and type travel together.

    `_doc_to_node_item` builds `NodeItem.parent` only when both are present, so
    a row with an id and no type silently loses the triple in the API response.
    """
    for parent in ("pl-app", "kb-1", "swm-app"):
        for row_id, row in _rows(await _browse(provider, parent)).items():
            assert row["parentId"], f"{row_id} has no parentId under {parent}"
            assert row["parentType"] in ("app", "recordGroup", "record"), row


# ------------------------------------------------------------------- filters

async def test_only_containers_keeps_what_the_sidebar_can_expand(
    loaded_graph, provider
) -> None:
    """The sidebar sends this on every expand, and nothing proved it end to end.

    `kb-f1` holds `kb-f2`, so it expands; `kb-r4` is a leaf. Dropping the flag's
    effect would flood the tree with files, and applying it when unasked would
    hide them from the data table.
    """
    everything = set(_rows(await _browse(provider, "kb-1")))
    assert {"kb-f1", "kb-r4"} <= everything, sorted(everything)

    containers = _rows(await _browse(provider, "kb-1", only_containers=True))
    assert set(containers) == {"kb-f1"}, sorted(containers)
    assert containers["kb-f1"]["hasChildren"] is True


async def test_record_group_ids_does_not_narrow_anything_else(
    loaded_graph, provider
) -> None:
    """BE-11: the agent's KB boundary. It restricts COLLECTION-origin record
    groups and nothing else.

    The fixture has no such group — a collection holds Records, not RecordGroups
    — so the restricting half stays builder-level. What is provable here is the
    direction that loses data silently: a list naming an unrelated id must leave
    connector groups and records exactly as they were. This is also where the
    dialects could diverge, since Cypher yields null for `null <> 'COLLECTION'`
    where AQL yields true.
    """
    baseline = set(_rows(await _browse(provider, "pl-app")))
    assert baseline, "the placement App should list several rows"

    filtered = set(_rows(await _browse(provider, "pl-app", record_group_ids=["kb-1"])))
    assert filtered == baseline, sorted(baseline ^ filtered)


# ---------------------------------------------------------- available filters

async def test_available_filters_list_exactly_the_openable_sources(
    loaded_graph, provider
) -> None:
    """PG-34 end to end: the gate, the listing and the filter list agree.

    `gate-app` (nothing reaches this user) and `kb-2` (a collection with no
    grant) are what the old `get_user_apps` source could not exclude on its own,
    and a reachable collection is what it dropped unconditionally.
    """
    service = KnowledgeHubService(
        logger=logging.getLogger("kh_contract"), graph_provider=provider
    )
    filters = await service._get_available_filters(USER, ORG)

    listed = {option.id: option for option in filters.connectors}
    assert set(listed) == set(GATED_APPS), sorted(set(listed) ^ set(GATED_APPS))
    assert "gate-app" not in listed and "kb-2" not in listed
    assert listed["kb-1"].label == "My Collection", listed["kb-1"]
    assert listed["pl-app"].label, "a source with no label is unusable in a filter"
