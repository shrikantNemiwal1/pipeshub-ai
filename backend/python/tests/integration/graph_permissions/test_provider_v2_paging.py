"""Backward paging and whole-result ids against both real engines.

Global search needs two things from every partition query: a previous page
that is exactly the page the user came from (PG-13), and, on the first page,
every matching id so a node found in two partitions counts once (PG-24).
"""

import pytest

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


def _flatten(prov):
    async def call(**kwargs):
        return await prov.get_knowledge_hub_children_v2(
            user_key=USER, org_id=ORG, parent_id="pl-app",
            grantee_ids=GRANTEES, gated_app_ids=GATED_APPS, flatten=True, **kwargs,
        )
    return call


def _root(prov):
    async def call(**kwargs):
        return await prov.get_knowledge_hub_root_nodes_v2(
            user_key=USER, org_id=ORG, user_app_ids=GATED_APPS, **kwargs,
        )
    return call


def _boundary(row: dict) -> dict:
    return {"nullRank": row["nullRank"], "sortKey": row["sortKey"], "id": row["id"]}


async def _pages_forward(call, limit, **kwargs) -> list[list[dict]]:
    pages, after = [], None
    for _ in range(100):
        part = (await call(limit=limit, after=after, **kwargs))["partitions"][0]
        pages.append(part["rows"])
        if not part["hasMore"]:
            return pages
        after = _boundary(part["rows"][-1])
    raise AssertionError("forward paging did not terminate")


def _gp_flatten(prov):
    """A flatten of gp-app, the one scenario whose records carry sizes.

    gp-rg1 itself has no inheritance edge and no grant, so it is not admitted
    and cannot be the scope; its granted records are chain-tops that place
    under the App.
    """
    async def call(**kwargs):
        return await prov.get_knowledge_hub_children_v2(
            user_key=USER, org_id=ORG, parent_id="gp-app",
            grantee_ids=GRANTEES, gated_app_ids=GATED_APPS, flatten=True, **kwargs,
        )
    return call


@pytest.mark.parametrize(
    "sort_dir, expected",
    [
        ("ASC", ["gp-group", "gp-team", "gp-role", "gp-user"]),
        ("DESC", ["gp-user", "gp-role", "gp-team", "gp-group"]),
    ],
)
async def test_size_sorts_as_a_number(
    loaded_graph, provider, sort_dir, expected
) -> None:
    """PG-15: sizes 9, 10, 100 and 2000 order numerically, not as text.

    Lexicographically those sort "10", "100", "2000", "9" -- a completely
    different sequence -- so a comparison on the rendered value cannot pass
    here by accident.

    The assertion reads `nullRank` rather than a fixed row set: the projection
    does not carry `sizeInBytes`, and pinning membership would make this fail
    for a change in what gp-app contains rather than for the ordering it
    exists to check. Requiring every remaining row to be a null also pins
    nulls-last in **both** directions (PG-51).
    """
    rows = (await _gp_flatten(provider)(
        limit=100, sort_field="sizeInBytes", sort_dir=sort_dir,
    ))["partitions"][0]["rows"]

    sized = [row["id"] for row in rows if row["nullRank"] == 0]
    assert sized == expected, [(r["id"], r["nullRank"], r["sortKey"]) for r in rows]
    assert all(row["nullRank"] == 1 for row in rows[len(sized):]), (
        "a node with no size must sort last whichever way the page runs"
    )


@pytest.mark.parametrize("sort_field, sort_dir", [("name", "ASC"), ("name", "DESC"), ("createdAt", "ASC")])
@pytest.mark.parametrize("listing", ["flatten", "root"])
async def test_a_previous_page_is_exactly_the_page_before(
    loaded_graph, provider, listing, sort_field, sort_dir
) -> None:
    """PG-13: walk forward, then back from each page's first row; every previous
    page must equal the forward page before it, and the walk back must end at the
    first page with nothing before it. createdAt ties on every fixture node, so the
    id tiebreak alone decides that parameter — where a half-reversed sort slips."""
    call = (_flatten if listing == "flatten" else _root)(provider)
    limit = 2
    pages = await _pages_forward(call, limit, sort_field=sort_field, sort_dir=sort_dir)
    assert len(pages) >= 3, f"need several pages to test paging back, got {len(pages)}"

    for index in range(len(pages) - 1, 0, -1):
        part = (await call(limit=limit, after=_boundary(pages[index][0]), direction="prev",
                           sort_field=sort_field, sort_dir=sort_dir))["partitions"][0]
        assert [r["id"] for r in part["rows"]] == [r["id"] for r in pages[index - 1]], index
        assert part["hasMore"] == (index - 1 > 0), index


@pytest.mark.parametrize("listing", ["flatten", "root"])
async def test_include_ids_returns_every_matching_id(loaded_graph, provider, listing) -> None:
    """PG-24's exact total: ids cover the whole result, not the page.

    Each entry carries its `nodeType` as well, because the whole-result counts
    (PG-33) are taken over the union across partitions — summing each
    partition's own counts would count a two-parent node twice.
    """
    call = (_flatten if listing == "flatten" else _root)(provider)
    pages = await _pages_forward(call, 50)
    whole = {r["id"]: r["nodeType"] for page in pages for r in page}
    part = (await call(limit=2, include_ids=True))["partitions"][0]
    assert len(part["rows"]) == 2
    returned = {entry["id"]: entry["nodeType"] for entry in part["ids"]}
    assert returned == whole, sorted(set(returned) ^ set(whole))
    assert len(part["ids"]) == len(returned), "an id was returned twice"
    assert (await call(limit=2))["partitions"][0]["ids"] == [], "ids must be opt-in"


async def test_the_root_listing_applies_search_filters(loaded_graph, provider) -> None:
    """The Apps partition of a global search takes the same filters as the others."""
    part = (await _root(provider)(limit=50, search_query="PLACEMENT"))["partitions"][0]
    assert {r["id"] for r in part["rows"]} == {"pl-app"}


async def test_an_unknown_direction_is_refused(loaded_graph, provider) -> None:
    with pytest.raises(ValueError, match="page direction"):
        await _flatten(provider)(limit=2, direction="sideways")
