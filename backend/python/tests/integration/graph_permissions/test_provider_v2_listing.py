"""`get_knowledge_hub_root_nodes_v2` against both real engines.

The unit tests proved the builders self-consistent. This is the first thing that
proves they *compose* — into a query each server accepts, that pages correctly,
and that returns the row shape the cursor and merge expect.

That distinction is not theoretical. Writing these two methods surfaced six
composition defects the passing unit suite could not see: the projection could
not carry a sort key that did not exist yet; it assumed storage-shaped input
where an App's `origin` and `connector` are computed; Cypher's `WITH` silently
dropped the carried total; the Cypher keyset had to sit inside the sort's
`WITH`; Arango's tiebreak read `_key`, which is null on a projected row; and the
Arango keyset had to sit between the sort's `LET`s and its `SORT`. So nothing
here asserts "it parsed" without also asserting what it means.

Every test runs against **both** providers. Cross-backend agreement is therefore
structural rather than a single comparison bolted on at the end: the two must
satisfy identical expectations, and `test_both_backends_return_the_same_page`
additionally pins the rows against each other.
"""

import pytest

pytestmark = pytest.mark.integration

USER = "user-u"
ORG = "org-1"

# Every App in the fixture. The gate is the caller's list, so passing them all
# exercises the query rather than the caller's filtering.
APP_IDS = [
    "ex1-app", "ex2-app", "dec-app", "dec-rgl-app", "kb-1", "kb-2",
    "pl-app", "gp-app", "ex-app", "swm-app", "gate-app", "flag-app",
]


@pytest.fixture(params=["neo4j", "arango"])
def provider(request, neo4j_provider, arango_provider):
    """Each test twice, once per backend.

    Both fixtures are requested either way, so the graph is loaded into both
    stores before any parameter runs — the alternative would let the first
    backend's tests pass against a store the second had not yet populated.
    """
    return neo4j_provider if request.param == "neo4j" else arango_provider


async def _page(prov, limit, after=None, app_ids=None, **kwargs):
    return await prov.get_knowledge_hub_root_nodes_v2(
        user_key=USER,
        org_id=ORG,
        user_app_ids=APP_IDS if app_ids is None else app_ids,
        limit=limit,
        after=after,
        **kwargs,
    )


def _rows(result) -> list[dict]:
    assert len(result["partitions"]) == 1, result["partitions"]
    return result["partitions"][0]["rows"]


def _boundary(row: dict) -> dict:
    return {"nullRank": row["nullRank"], "sortKey": row["sortKey"], "id": row["id"]}


async def test_returns_the_partitioned_envelope(loaded_graph, provider) -> None:
    """One partition, because a root listing traverses nothing.

    It still returns the envelope so the service handles all three v2 methods
    the same way instead of special-casing this one.
    """
    result = await _page(provider, limit=100)
    assert result["scope"] is None
    partition = result["partitions"][0]
    assert partition["partitionKind"] == "ROOT"
    assert partition["appId"] is None
    # `ids` is empty unless include_ids asks for every matching id, which a
    # global search needs on its first page for an exact total (PG-24).
    assert set(partition) == {
        "partitionId", "partitionKind", "appId", "rows",
        "hasMore", "exhausted", "total", "countsByType", "ids",
    }, sorted(partition)
    assert partition["ids"] == []


async def test_rows_carry_bare_ids_and_the_comparator_output(
    loaded_graph, provider
) -> None:
    """API-07 plus the cursor/merge contract, on real rows.

    A prefixed id would round-trip wrong for a client, and a row without
    sortKey/nullRank cannot be resumed from or merged against another partition.
    """
    rows = _rows(await _page(provider, limit=100))
    assert rows, "the fixture should expose several apps"
    for row in rows:
        assert "/" not in row["id"], f"collection prefix leaked: {row['id']}"
        assert row["nodeType"] == "app"
        assert "sortKey" in row and "nullRank" in row, sorted(row)
        assert row["parentId"] is None, "an App has no parent"


async def test_collections_and_connectors_are_distinguished(
    loaded_graph, provider
) -> None:
    """`origin` is computed from `type`; an App stores no origin field."""
    by_id = {row["id"]: row for row in _rows(await _page(provider, limit=100))}
    assert by_id["kb-1"]["origin"] == "COLLECTION"
    assert by_id["kb-1"]["connector"] == "KB"
    assert by_id["ex1-app"]["origin"] == "CONNECTOR"
    assert by_id["ex1-app"]["webUrl"] == "/app/ex1-app"


@pytest.mark.parametrize("sort_dir", ["ASC", "DESC"])
@pytest.mark.parametrize("sort_field", ["name", "createdAt"])
async def test_paging_reproduces_the_single_page_order(
    loaded_graph, provider, sort_field: str, sort_dir: str
) -> None:
    """Keyset paging in threes must equal one large page, exactly.

    `createdAt` is the sharper case: every fixture node carries the same fixed
    timestamp, so the sort key ties on every row and the id tiebreak alone
    decides the order. Mutation testing confirmed it is the case doing the work
    — breaking the keyset's direction handling failed *only* this parameter,
    because the name-sorted rows have distinct keys and never reach the
    tiebreak.

    It is also where a wrong tiebreak stays invisible: on Arango a projected row
    has no `_key`, so reading it yields null rather than an error, every row
    ties, and paging silently skips and repeats rows while every page still
    looks full.
    """
    whole = _rows(await _page(
        provider, limit=100, sort_field=sort_field, sort_dir=sort_dir,
    ))
    expected = [row["id"] for row in whole]
    assert len(expected) >= 5, expected

    seen: list[str] = []
    after = None
    for _ in range(20):
        result = await _page(
            provider, limit=3, after=after,
            sort_field=sort_field, sort_dir=sort_dir,
        )
        partition = result["partitions"][0]
        seen.extend(row["id"] for row in partition["rows"])
        if not partition["hasMore"]:
            break
        after = _boundary(partition["rows"][-1])
    else:
        pytest.fail(f"paging never exhausted: {seen}")

    assert seen == expected, f"paged={seen}\nwhole={expected}"
    assert len(seen) == len(set(seen)), f"a row was returned twice: {seen}"


async def test_has_more_and_exhausted_agree(loaded_graph, provider) -> None:
    small = (await _page(provider, limit=2))["partitions"][0]
    assert small["hasMore"] is True
    assert small["exhausted"] is False
    assert len(small["rows"]) == 2, "the probe row must be trimmed off"

    everything = (await _page(provider, limit=100))["partitions"][0]
    assert everything["hasMore"] is False
    assert everything["exhausted"] is True


async def test_total_is_counted_once_and_not_repeated(
    loaded_graph, provider
) -> None:
    """The cursor caches the total; recounting per page pays for it twice.

    A later page returns None rather than a number, because a *wrong* total is
    worse than an absent one — the caller would render it.
    """
    first = (await _page(provider, limit=2))["partitions"][0]
    assert first["total"] == len(_rows(await _page(provider, limit=100)))

    second = (await _page(
        provider, limit=2, after=_boundary(first["rows"][-1]),
    ))["partitions"][0]
    assert second["total"] is None


async def test_an_app_the_caller_did_not_list_is_not_returned(
    loaded_graph, provider
) -> None:
    """The connector gate is the caller's list, and nothing else widens it."""
    result = await _page(provider, limit=100, app_ids=["ex1-app"])
    assert [row["id"] for row in _rows(result)] == ["ex1-app"]


async def test_the_arango_listing_breaks_ties_on_the_projected_id(
    loaded_graph, arango_provider
) -> None:
    """White-box on purpose: the behavioural test provably cannot see this.

    Mutation testing established it. Nulling the Arango tiebreak — reverting
    `id_expr` so the sort falls back to `node._key`, which is null on a
    projected row — failed **none** of the eight paging parameters.

    The reason is a coincidence, not correctness. The projected `id` *is* the
    document `_key`, and `FOR app IN apps` returns rows in primary-index order,
    so the broken tie order matches the correct one exactly and the round-trip
    still reproduces the single-page order.

    That coincidence holds only for a full collection scan. It disappears as
    soon as rows arrive from a traversal, which is every remaining v2 method, so
    the contract is pinned here rather than left for B5 to rediscover as a
    paging bug that skips and repeats rows while every page still looks full.
    """
    import unittest.mock as mock

    original = arango_provider._kh_v2_sort_aql
    captured: dict = {}

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    with mock.patch.object(arango_provider, "_kh_v2_sort_aql", spy):
        await _page(arango_provider, limit=3)

    assert captured, "the sort builder was never called"
    assert captured.get("id_expr") == "node.id", (
        f"the sort must break ties on the projected id, not the document key: "
        f"{captured}"
    )


async def test_both_backends_return_the_same_page(
    loaded_graph, neo4j_provider, arango_provider
) -> None:
    """BE-01. The same request, the same rows, in the same order, field for field.

    The parametrised tests above already hold both backends to identical
    expectations; this pins them to each other, which catches a field that
    differs in *value* rather than in presence — a computed `origin` or
    `sharingStatus` that one dialect derives differently.
    """
    cypher = _rows(await _page(neo4j_provider, limit=100))
    aql = _rows(await _page(arango_provider, limit=100))

    assert [r["id"] for r in cypher] == [r["id"] for r in aql], (
        f"order diverges:\nneo4j={[r['id'] for r in cypher]}\n"
        f"arango={[r['id'] for r in aql]}"
    )
    for left, right in zip(cypher, aql):
        assert set(left) == set(right), (
            f"{left['id']}: neo4j-only={sorted(set(left) - set(right))}, "
            f"arango-only={sorted(set(right) - set(left))}"
        )
        for field in ("nodeType", "origin", "connector", "webUrl",
                      "sharingStatus", "hasChildren", "parentId"):
            assert left[field] == right[field], (
                f"{left['id']}.{field}: neo4j={left[field]!r} arango={right[field]!r}"
            )
