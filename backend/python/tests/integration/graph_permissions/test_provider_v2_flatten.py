"""Flatten (`flatten=True`) against both real engines.

A flatten of X returns every node whose placement chain passes through X, once,
under its preferred placement parent inside X (decision 28, NV-36, NV-39). The
expectations are written as ``{id: parentId}`` so a row in the wrong place
fails as loudly as a missing one. Browse's placement tests show where a node
lives one level at a time; these show the same answer for a whole subtree.
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


async def _nodes(prov, parent_id, *, flatten, limit=200, **kwargs):
    return await prov.get_knowledge_hub_children_v2(
        user_key=USER,
        org_id=ORG,
        parent_id=parent_id,
        limit=limit,
        grantee_ids=GRANTEES,
        gated_app_ids=GATED_APPS,
        flatten=flatten,
        **kwargs,
    )


def _placed(result) -> dict[str, str]:
    assert len(result["partitions"]) == 1, result["partitions"]
    return {row["id"]: row["parentId"] for row in result["partitions"][0]["rows"]}


async def _flat(prov, parent_id, **kwargs) -> dict[str, str]:
    result = await _nodes(prov, parent_id, flatten=True, **kwargs)
    assert result["scope"]["admitted"] is True, result["scope"]
    return _placed(result)


PL_APP = {
    "pl-rg1": "pl-app", "pl-r5": "pl-rg1", "pl-r6": "pl-rg1", "pl-r11": "pl-rg1",
    "pl-rg3": "pl-app", "pl-r12": "pl-rg3", "pl-r9": "pl-app",
    "pl-rg4": "pl-app", "pl-r14": "pl-rg4", "pl-r15": "pl-rg4",
}
SWM_APP = {
    "swm-drive": "swm-app", "swm-inbox": "swm-app", "swm-f1": "swm-drive",
    "swm-x": "swm-f1", "swm-y": "swm-inbox",
    # Both hierarchy parents are unreachable, so swm-z is placed by its own
    # group. It has two; swm-ainbox is not in scope, so swm-drive is the answer.
    "swm-z": "swm-drive",
}


async def test_a_flatten_is_one_subtree_partition(loaded_graph, provider) -> None:
    """A subtree-scoped request is one query, not a partitioned fan-out (PG-09)."""
    result = await _nodes(provider, "pl-rg1", flatten=True)
    assert len(result["partitions"]) == 1
    assert result["partitions"][0]["partitionKind"] == "SUBTREE"
    assert result["scope"]["admitted"] is True


async def test_a_flatten_reaches_below_a_gap_inside_the_scope(loaded_graph, provider) -> None:
    """NV-32: the chain-top lists under its group, its descendants under it."""
    assert await _flat(provider, "ex2-rg1") == {
        "ex2-r4": "ex2-rg1", "ex2-r6": "ex2-rg1", "ex2-r7": "ex2-r6", "ex2-r8": "ex2-r6",
    }


async def test_a_flatten_can_start_at_a_chain_top(loaded_graph, provider) -> None:
    """NV-34: the start node is admitted through its own grant."""
    assert await _flat(provider, "ex2-r6") == {"ex2-r7": "ex2-r6", "ex2-r8": "ex2-r6"}


async def test_a_grant_placed_above_the_scope_is_out_of_reach(loaded_graph, provider) -> None:
    """AC-59, NV-45: pl-r11 lies below pl-r5 but lists under pl-rg1, outside this scope."""
    assert await _flat(provider, "pl-r5") == {}


async def test_a_group_flatten_places_chain_tops_and_refuses_strict_below_them(
    loaded_graph, provider
) -> None:
    """The gaps never appear, and pl-r13 (STRICT under the chain-top pl-r6) stays out."""
    assert await _flat(provider, "pl-rg1") == {
        "pl-r5": "pl-rg1", "pl-r6": "pl-rg1", "pl-r11": "pl-rg1",
    }


async def test_an_app_flatten_places_every_node_it_can_open(loaded_graph, provider) -> None:
    """Every placement rule at once: the App fallback (pl-r9, pl-rg4), a
    group opened only by its own grant (pl-r14, pl-r15), a group granted to a
    group (pl-r12), and no gap anywhere."""
    assert await _flat(provider, "pl-app") == PL_APP


async def test_a_two_parent_record_appears_once_under_its_drive_parent(
    loaded_graph, provider
) -> None:
    """NV-17, decision 67: swm-x reaches the flatten through both parents."""
    assert await _flat(provider, "swm-app") == SWM_APP


async def test_a_flatten_of_the_other_parent_includes_the_record_under_it(
    loaded_graph, provider
) -> None:
    """The user's decision: a scope includes a node reached through any of its
    placement parents inside the scope, under that parent."""
    assert await _flat(provider, "swm-inbox") == {"swm-x": "swm-inbox", "swm-y": "swm-inbox"}


@pytest.mark.parametrize(
    "scope, expected",
    [
        ("kb-1", {"kb-f1": "kb-1", "kb-f2": "kb-f1", "kb-r3": "kb-f2", "kb-r4": "kb-1"}),
        ("kb-f1", {"kb-f2": "kb-f1", "kb-r3": "kb-f2"}),
    ],
    ids=["AC-44-collection", "NV-35-folder"],
)
async def test_a_collection_flattens_with_its_role(loaded_graph, provider, scope, expected) -> None:
    """AC-44, NV-26, NV-35: every item, at its parentId, with the collection's role."""
    result = await _nodes(provider, scope, flatten=True)
    assert _placed(result) == expected
    assert all(row["userRole"] == "WRITER" for row in result["partitions"][0]["rows"])


@pytest.mark.parametrize(
    "scope, expected",
    [
        ("dec-app", {"dec-rg1": "dec-app", "dec-r1": "dec-rg1"}),
        ("dec-rgl-app", {"dec-rg2": "dec-rgl-app", "dec-r2": "dec-rg2",
                         "dec-rg3": "dec-rg2", "dec-r3": "dec-rg3"}),
    ],
    ids=["AC-68", "AC-69"],
)
async def test_a_declaration_flattens_everything_below_it(
    loaded_graph, provider, scope, expected
) -> None:
    """§3.8: dec-rg4 (undeclared) and dec-rg5 (declared, ungranted) stay out."""
    assert await _flat(provider, scope) == expected


async def test_hidden_and_deleted_content_stays_out_of_a_flatten(loaded_graph, provider) -> None:
    """D54, D40, D44: the hidden channel is listed, its content is not."""
    assert set(await _flat(provider, "ex-app")) == {
        "ex-rg1", "ex-hidden", "ex-stub", "ex-under-stub",
    }


@pytest.mark.parametrize("scope", ["pl-r3", "ex1-r4"])
async def test_a_flatten_of_an_inaccessible_node_is_refused(loaded_graph, provider, scope) -> None:
    """AC-59, AC-58: a scope the user cannot open returns nothing at all."""
    result = await _nodes(provider, scope, flatten=True)
    assert result["scope"] == {"admitted": False, "nodeId": scope}
    assert result["partitions"][0]["rows"] == []


@pytest.mark.parametrize("scope", ["pl-app", "swm-app", "kb-1", "dec-rgl-app", "ex2-app"])
async def test_every_flattened_row_is_listed_at_its_parent(loaded_graph, provider, scope) -> None:
    """NV-36: browsing a row's parentId lists that row."""
    placed = await _flat(provider, scope)
    assert placed, f"{scope} flattened to nothing; the invariant would hold vacuously"
    for node_id, parent_id in placed.items():
        listed = _placed(await _nodes(provider, parent_id, flatten=False))
        assert node_id in listed, f"{node_id} has parentId {parent_id}, which lists {sorted(listed)}"


@pytest.mark.parametrize("scope", ["pl-app", "swm-app", "kb-1"])
async def test_the_parent_chain_equals_the_breadcrumbs(loaded_graph, provider, scope) -> None:
    """NV-39: following parentId up to the scope gives each row's breadcrumbs."""
    placed = await _flat(provider, scope)
    for node_id in placed:
        chain = [node_id]
        while chain[-1] != scope:
            chain.append(placed[chain[-1]])
        crumbs = (await _nodes(provider, node_id, flatten=False))["scope"]["breadcrumbs"]
        assert [c["id"] for c in crumbs] == list(reversed(chain)), node_id


async def test_paging_a_flatten_reproduces_the_single_page(loaded_graph, provider) -> None:
    whole = [r["id"] for r in (await _nodes(provider, "pl-app", flatten=True))["partitions"][0]["rows"]]
    seen, after = [], None
    for _ in range(len(whole) + 1):
        part = (await _nodes(provider, "pl-app", flatten=True, limit=3, after=after))["partitions"][0]
        seen += [r["id"] for r in part["rows"]]
        if not part["hasMore"]:
            break
        last = part["rows"][-1]
        after = {"nullRank": last["nullRank"], "sortKey": last["sortKey"], "id": last["id"]}
    assert seen == whole, f"paged={seen}\nwhole={whole}"


FIXTURE_TS = 1_700_000_000_000  # fixture_graph.TS, the createdAt of every node


@pytest.mark.parametrize(
    "scope, filters, expected",
    [
        ("pl-app", {"search_query": "GRANTED"},
         {"pl-r6", "pl-r9", "pl-r11", "pl-r12", "pl-r14", "pl-r15", "pl-rg3", "pl-rg4"}),
        ("pl-app", {"search_query": "gap"}, {"pl-r6", "pl-r11", "pl-rg4", "pl-r14"}),
        ("pl-app", {"node_types": ["recordGroup"]}, {"pl-rg1", "pl-rg3", "pl-rg4"}),
        ("swm-app", {"record_types": ["FILE"]}, {"swm-f1", "swm-x", "swm-y", "swm-z"}),
        ("kb-1", {"origins": ["UPLOAD"]}, {"kb-f1", "kb-f2", "kb-r3", "kb-r4"}),
        ("kb-1", {"origins": ["CONNECTOR"]}, set()),
        ("pl-rg1", {"created_at": {"gte": FIXTURE_TS, "lte": FIXTURE_TS}},
         {"pl-r5", "pl-r6", "pl-r11"}),
        ("pl-rg1", {"created_at": {"gte": FIXTURE_TS + 1}}, set()),
        ("pl-rg1", {"size": {"gte": 0}}, set()),
        # BE-03: the match is two groups down -- dec-rg2 -> dec-rg3 -> dec-r3 --
        # and "Nested item" names only the record, not its group ("Nested group")
        # nor its sibling ("Work item"), so a hit proves the search descended
        # into the nested group rather than merely finding something.
        ("dec-rg2", {"search_query": "Nested item"}, {"dec-r3"}),
    ],
    ids=["PG-38-case-insensitive-q", "PG-37-no-inaccessible-match", "PG-41-node-types",
         "record-types", "origins-upload", "origins-connector", "PG-46-inclusive-window",
         "window-excludes", "PG-47-no-size-excluded", "BE-03-nested-group"],
)
async def test_filters_apply_after_permission(
    loaded_graph, provider, scope, filters, expected
) -> None:
    """§3.4 step 6: filters narrow the admitted rows and never open a path.

    `q=gap` is the sharp case (PG-37): `pl-r3` "Gap folder" and `pl-r10`
    "Second gap" match the text but cannot be opened, so they stay out, while
    `pl-r11` matches and is found below the unmatched `pl-r5` (PG-36). The size
    window excludes every node here because none has a size: an unknown size is
    not zero (PG-47).
    """
    assert set(await _flat(provider, scope, **filters)) == expected


async def test_rows_name_their_parent_and_whether_it_is_internal(loaded_graph, provider) -> None:
    """D69: every row carries its parent's name, and the merge's internal flag (D67)."""
    rows = {r["id"]: r for r in (await _nodes(provider, "swm-app", flatten=True))["partitions"][0]["rows"]}
    assert rows["swm-x"]["parentName"] == "Folder", rows["swm-x"]
    assert rows["swm-x"]["parentIsInternal"] is False, rows["swm-x"]
    assert rows["swm-y"]["parentName"] == "U's Shared with Me", rows["swm-y"]
    assert rows["swm-y"]["parentIsInternal"] is True, rows["swm-y"]
    assert rows["swm-drive"]["parentName"] == "Drive", rows["swm-drive"]


@pytest.mark.parametrize("scope", ["pl-app", "swm-app", "kb-1", "dec-rgl-app"])
async def test_both_backends_flatten_alike(
    loaded_graph, neo4j_provider, arango_provider, scope
) -> None:
    """BE-01: one flatten, one placement map, whichever store answers it."""
    cypher = (await _nodes(neo4j_provider, scope, flatten=True))["partitions"][0]["rows"]
    aql = (await _nodes(arango_provider, scope, flatten=True))["partitions"][0]["rows"]
    assert cypher, f"{scope} flattened to nothing"
    assert [r["id"] for r in cypher] == [r["id"] for r in aql]
    for left, right in zip(cypher, aql):
        for field in ("name", "nodeType", "parentId", "parentType", "hasChildren",
                      "userRole", "sortKey", "nullRank"):
            assert left[field] == right[field], f"{left['id']}.{field}: {left[field]!r} vs {right[field]!r}"
