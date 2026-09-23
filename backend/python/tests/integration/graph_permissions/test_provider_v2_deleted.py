"""PG-53, and PG-11's deleted half: a deleted record leaves *every* part of the response.

PG-11's other half -- that a *placeholder* is a normal item -- is not asserted
here and cannot be: both v2 filter builders exclude placeholders outright, while
decisions 44 and 71 and section 3.3 all say stubs are shown. That contradiction
is open, so citing PG-11 whole would claim coverage of a case the code does not
currently satisfy.

The flatten module already proves a deleted node is absent from `items`. Nothing
proved the rest of the row: the `total` beside those items, the per-type counts,
and the available filters. Those are computed on separate paths — the total and
counts from the whole-result id union, the filters from the gate — so a deleted
node can be filtered out of the list and still be counted, which reads as a page
that is simply missing a row.

`del-in-group` inherits inside an open connector group and `del-in-kb` sits in
the collection whose grant opens it, so both would be returned if they were live.
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
DELETED = {"del-in-group", "del-in-kb"}


@pytest.fixture(params=["neo4j", "arango"])
def provider(request, neo4j_provider, arango_provider):
    return neo4j_provider if request.param == "neo4j" else arango_provider


async def _nodes(prov, parent_id, **kwargs):
    return await prov.get_knowledge_hub_children_v2(
        user_key=USER, org_id=ORG, parent_id=parent_id, limit=200,
        grantee_ids=GRANTEES, gated_app_ids=GATED_APPS, **kwargs,
    )


def _partition(result) -> dict:
    assert len(result["partitions"]) == 1, result["partitions"]
    return result["partitions"][0]


def _ids(result) -> set[str]:
    return {row["id"] for row in _partition(result)["rows"]}


@pytest.mark.parametrize(
    "parent, deleted",
    [("pl-rg1", "del-in-group"), ("kb-1", "del-in-kb")],
    ids=["connector-group", "collection"],
)
async def test_a_deleted_record_is_absent_from_browse(
    loaded_graph, provider, parent, deleted
) -> None:
    """D54: the two admission paths reach these nodes differently — one by the
    per-hop rule, one by the collection opening its subtree — so each needs its
    own exclusion."""
    assert deleted not in _ids(await _nodes(provider, parent))


@pytest.mark.parametrize("scope", ["pl-app", "kb-1"])
async def test_a_deleted_record_is_absent_from_a_flatten(
    loaded_graph, provider, scope
) -> None:
    assert not DELETED & _ids(await _nodes(provider, scope, flatten=True))


@pytest.mark.parametrize(
    "parent", ["pl-rg1", "kb-1"],
)
async def test_the_total_counts_only_what_it_returned(
    loaded_graph, provider, parent
) -> None:
    """The count beside a listing is computed separately from the listing.

    A deleted node filtered out of `rows` but still counted gives a total that
    disagrees with the page — the same class of defect as counting the page
    while the total counts everything.
    """
    partition = _partition(await _nodes(provider, parent))
    assert partition["total"] == len(partition["rows"]), partition["rows"]


async def test_the_whole_result_total_and_counts_exclude_deleted_records(
    loaded_graph, provider
) -> None:
    """PG-33 over the id union: the counts are built from the same surviving set.

    `search_page` takes its own access context, so this is the whole read path —
    partition discovery, the per-partition queries and the merge — agreeing that
    a deleted record is not there to be counted.
    """
    page = await search_page(
        provider, user_key=USER, user_id=USER, org_id=ORG, secret=SECRET, limit=500
    )
    returned = {row["id"] for row in page.rows}

    assert not DELETED & returned, sorted(DELETED & returned)
    assert page.total == len(returned)
    assert sum(page.counts_by_type.values()) == page.total


async def test_a_deleted_record_never_reaches_the_available_filters(
    loaded_graph, provider
) -> None:
    """The last part of the row: the filter list is built from openable sources,
    so a deleted node must not add one, and must not remove the source it lived
    in — `pl-app` and `kb-1` both still hold live content."""
    ids = await provider.get_knowledge_hub_partitions_v2(ORG, GATED_APPS)
    named = {p["partitionId"] for p in ids}
    assert not DELETED & named, sorted(DELETED & named)
    assert {"pl-rg1", "kb-1"} <= named, sorted(named)


async def test_both_backends_exclude_the_same_deleted_records(
    loaded_graph, neo4j_provider, arango_provider
) -> None:
    """BE-01 for PG-53: both exclusion paths must agree across the two stores."""
    for scope in ("pl-app", "kb-1"):
        cypher = _ids(await _nodes(neo4j_provider, scope, flatten=True))
        aql = _ids(await _nodes(arango_provider, scope, flatten=True))
        assert cypher == aql, sorted(cypher ^ aql)
