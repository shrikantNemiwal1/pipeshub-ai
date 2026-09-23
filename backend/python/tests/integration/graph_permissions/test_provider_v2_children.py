"""`get_knowledge_hub_children_v2` against both real engines.

Browse is where placement matters. The root listing had no traversal at all, so
it could not exercise the rule that gives this model its shape: a node the user
can reach whose *parent* they cannot is the top of a continuous-permission
chain, and it lists under **its own record group** — not under the nearest
accessible ancestor (§3.3, decision 13).

`pl-r11` is the sharpest case in the fixture: a grant two levels below an
accessible record (`pl-rg1 -> pl-r5 ✓ -> pl-r10 ✗ -> pl-r11 grant`). It must
appear under `pl-rg1`, not under `pl-r5`. An implementation that walked up to
the nearest accessible ancestor would place it under `pl-r5` and look entirely
reasonable doing so.

Every test runs against both providers, so cross-backend agreement is
structural rather than a single comparison bolted on at the end. That matters
here more than anywhere: the root listing's equivalent test is what caught the
`sharingStatus` divergence, where Cypher makes `null = null` null and AQL makes
it true.
"""

import pytest

pytestmark = pytest.mark.integration

USER = "user-u"
ORG = "org-1"
GRANTEES = ["user-u", "group-g", "role-r", "team-t", "orgnode-1"]

# The apps the connector gate admits for USER_U, as test_seeds_and_grants pins
# them. gate-app and kb-2 are deliberately absent.
GATED_APPS = [
    "ex1-app", "ex2-app", "dec-app", "dec-rgl-app", "ex-app",
    "swm-app", "pl-app", "gp-app", "flag-app", "kb-1",
]


@pytest.fixture(params=["neo4j", "arango"])
def provider(request, neo4j_provider, arango_provider):
    """Each test twice, once per backend.

    Both fixtures are requested either way, so the graph is loaded into both
    stores before any parameter runs — otherwise the first backend's tests
    could pass against a store the second had not yet populated.
    """
    return neo4j_provider if request.param == "neo4j" else arango_provider


async def _browse(prov, parent_id, limit=100, **kwargs):
    return await prov.get_knowledge_hub_children_v2(
        user_key=USER,
        org_id=ORG,
        parent_id=parent_id,
        limit=limit,
        grantee_ids=GRANTEES,
        gated_app_ids=GATED_APPS,
        **kwargs,
    )


def _ids(result) -> set[str]:
    assert len(result["partitions"]) == 1, result["partitions"]
    return {row["id"] for row in result["partitions"][0]["rows"]}


def _boundary(row: dict) -> dict:
    return {"nullRank": row["nullRank"], "sortKey": row["sortKey"], "id": row["id"]}


async def test_the_envelope_is_a_single_browse_partition(
    loaded_graph, provider
) -> None:
    """Browse is one query, not partitioned (§3.9).

    Partitions exist for global search, filter and flatten; a subtree-scoped
    request has nothing to merge across. The envelope still matches the other
    v2 methods so the service treats them alike.
    """
    result = await _browse(provider, "ex1-rg1")
    partition = result["partitions"][0]
    assert partition["partitionKind"] == "BROWSE"
    assert partition["partitionId"] == "ex1-rg1"
    assert result["scope"]["admitted"] is True
    assert result["scope"]["nodeId"] == "ex1-rg1"


@pytest.mark.xfail(
    strict=True,
    reason="PG-48/D61: hasChildren is structural, not rule-aware. pl-r5 reports "
           "hasChildren=true on both engines while nothing under it is reachable",
)
async def test_has_children_counts_only_children_the_user_can_reach(
    loaded_graph, provider
) -> None:
    """PG-48's hasChildren half: the flag should be rule-aware, not structural (D61).

    `pl-r5` has exactly one child, `pl-r10`, which carries no grant and no
    inheritance edge -- `test_a_grant_placed_above_the_scope_is_out_of_reach`
    pins that flattening `pl-r5` returns nothing. Measured: both engines report
    `hasChildren=true` for it anyway, so the flag counts structural children
    rather than reachable ones. That renders as an expander opening onto an
    empty folder, and it is what PG-48 forbids ("F1.hasChildren = false, since
    R9 is not accessible").

    Both backends agree, so the four cross-engine parity comparisons cannot see
    it: they pin that the field matches, not what it says.

    `swm-f1` is the control rather than decoration: asserting only the false
    case would pass just as well with the flag hardcoded off.

    Marked xfail rather than fixed -- making the flag rule-aware means applying
    the per-hop rule to each candidate child, which has a cost worth deciding
    deliberately rather than inside a test-coverage change.
    """
    group = {row["id"]: row
             for row in (await _browse(provider, "pl-rg1"))["partitions"][0]["rows"]}
    assert group["pl-r5"]["hasChildren"] is False, group["pl-r5"]

    drive = {row["id"]: row
             for row in (await _browse(provider, "swm-drive"))["partitions"][0]["rows"]}
    assert drive["swm-f1"]["hasChildren"] is True, drive["swm-f1"]


async def test_browsing_a_space_hides_the_restricted_page(
    loaded_graph, provider
) -> None:
    """Example 1: RG1 -> R3 (strict, inherits) and R4 (restricted, no grant)."""
    found = _ids(await _browse(provider, "ex1-rg1"))
    assert "ex1-r3" in found, sorted(found)
    assert "ex1-r4" not in found, "a restricted page with no grant must stay hidden"


async def test_a_grant_below_a_gap_lists_under_its_own_group(
    loaded_graph, provider
) -> None:
    """§3.3, decision 13 — the rule the root listing could not reach.

    Under pl-rg1: pl-r5 arrives by the root pass (it inherits); pl-r6 and pl-r11
    arrive by the seeds arm, because each holds a grant, each belongs to pl-rg1,
    and the hierarchy parent of each is inaccessible.

    pl-r11 is the case worth the test: its chain is pl-rg1 -> pl-r5 (accessible)
    -> pl-r10 (gap) -> pl-r11. Placement is by *own group*, so it lists under
    pl-rg1 — an implementation that climbed to the nearest accessible ancestor
    would put it under pl-r5 and look perfectly sensible.
    """
    found = _ids(await _browse(provider, "pl-rg1"))
    assert found == {"pl-r5", "pl-r6", "pl-r11"}, sorted(found)
    assert "pl-r3" not in found, "the gap itself is never listed"
    assert "pl-r10" not in found, "the second gap is never listed"


async def test_an_inaccessible_node_is_not_admitted(loaded_graph, provider) -> None:
    """A downward walk cannot decide this; it needs the upward walk (AC-57).

    `admitted=False` is what the router turns into a 404 with a constant body —
    the response must not confirm the node exists (SEC-02).
    """
    result = await _browse(provider, "ex1-r4")
    assert result["scope"]["admitted"] is False
    assert result["scope"] == {"admitted": False, "nodeId": "ex1-r4"}, (
        "SEC-02: an inadmissible node carries no name, type or trail"
    )
    assert result["partitions"][0]["rows"] == []


async def test_a_subtree_behind_the_connector_gate_is_not_admitted(
    loaded_graph, provider
) -> None:
    """AC-36: grants inside an app the user cannot reach admit nothing.

    gate-rg1 carries a direct grant, so only the gate keeps it out.
    """
    result = await _browse(provider, "gate-rg1")
    assert result["scope"]["admitted"] is False
    assert result["partitions"][0]["rows"] == []


async def test_rows_carry_bare_ids_and_the_comparator_output(
    loaded_graph, provider
) -> None:
    rows = (await _browse(provider, "pl-rg1"))["partitions"][0]["rows"]
    assert rows
    for row in rows:
        assert "/" not in row["id"], f"collection prefix leaked: {row['id']}"
        assert "sortKey" in row and "nullRank" in row, sorted(row)
        assert row["parentId"] == "pl-rg1", row


async def test_records_carry_the_names_the_fixture_gave_them(
    loaded_graph, provider
) -> None:
    """Assert against fixture ground truth, not against the other backend.

    This exists because cross-backend parity has a blind spot. The projection
    read `name`, a property only an App carries — a record stores `recordName`
    — so every browse row was nameless on *both* backends, and they agreed with
    each other perfectly. Eight passing tests, a clean 108-test suite and three
    mutation rounds went over it. It surfaced only as a side effect of AQL
    coercing `LOWER(null)` to `""` while Cypher keeps null: an accident of
    dialect, not of design.

    Parity catches divergence; it cannot catch a mistake both backends inherit
    from a shared builder. Only ground truth can.
    """
    rows = {r["id"]: r
            for r in (await _browse(provider, "pl-rg1"))["partitions"][0]["rows"]}
    expected = {
        "pl-r5": "Reachable child",
        "pl-r6": "Granted below gap",
        "pl-r11": "Granted below second gap",
    }
    for node_id, name in expected.items():
        assert rows[node_id]["name"] == name, rows[node_id]


async def test_record_groups_carry_their_group_names(
    loaded_graph, provider
) -> None:
    """The `groupName` branch of the fallback, exercised on its own.

    A record group stores `groupName`, not `recordName`, so a fallback that
    handled only records would still leave every space nameless — and the
    record test above would not notice.
    """
    rows = {r["id"]: r
            for r in (await _browse(provider, "ex1-app"))["partitions"][0]["rows"]}
    assert rows["ex1-rg1"]["name"] == "Space 1", rows["ex1-rg1"]
    assert rows["ex1-rg2"]["name"] == "Space 2", rows["ex1-rg2"]


async def test_a_grant_in_an_unreachable_group_lists_under_the_app(
    loaded_graph, provider
) -> None:
    """§3.3 — the App fallback (decision 13, NV-07).

    `pl-r9` holds a direct grant and belongs only to `pl-rg2`, which is attached
    to the App but neither inherits from it nor holds a grant — so the group is
    hidden. The record must still appear, directly under the App.

    The exact-set assertion is what gives this teeth: `pl-rg2` must **not** be
    listed. An implementation that surfaced a hidden group's contents by
    listing the group itself would otherwise pass.
    """
    partition = (await _browse(provider, "pl-app"))["partitions"][0]
    found = {row["id"]: row for row in partition["rows"]}
    assert set(found) == {"pl-rg1", "pl-rg3", "pl-r9", "pl-rg4"}, sorted(found)
    assert found["pl-r9"]["parentId"] == "pl-app", found["pl-r9"]


async def test_a_grant_in_an_openable_group_below_a_gap_lists_under_that_group(
    loaded_graph, provider
) -> None:
    """§3.3: a record lists under its own group whenever the user can open that group.

    `pl-rg4` sits below the unreachable `pl-rg2` but is granted, so it opens on
    its own and lists under the App. `pl-r14` is granted inside it. Its own
    group is openable, so it lists under `pl-rg4` and never also under the App.
    The App is where "own group unreachable by the rule from the App" would put
    it, and that would disagree with its breadcrumbs (NV-36, NV-39).
    """
    under_app = _ids(await _browse(provider, "pl-app"))
    assert "pl-rg4" in under_app, sorted(under_app)
    assert "pl-r14" not in under_app, sorted(under_app)
    # pl-r15 is a chain-top below the closed folder pl-f5. Its own group opens
    # only through its own grant, which the rule walked from the App cannot see.
    assert "pl-r15" not in under_app, sorted(under_app)
    assert _ids(await _browse(provider, "pl-rg4")) == {"pl-r14", "pl-r15"}


async def test_a_granted_record_under_an_open_folder_is_not_listed_again_under_its_group(
    loaded_graph, provider
) -> None:
    """NV-37: placement by own group is only for chain-tops.

    `swm-x` is granted and belongs to `swm-drive`, but its folder `swm-f1` is
    open, so it lists under the folder. Browsing the drive must not list it a
    second time by its own group.
    """
    # swm-z is here for a different reason and belongs: it is a chain-top whose
    # only openable own group is this one, which is also what its trail says.
    assert _ids(await _browse(provider, "swm-drive")) == {"swm-f1", "swm-z"}


async def test_a_chain_top_places_under_the_own_group_it_can_open(
    loaded_graph, provider
) -> None:
    """A chain-top with two own groups uses the one the user can open (§3.3).

    `swm-z` belongs to `swm-ainbox`, another user's inbox that U holds no grant
    on, and to `swm-drive`, which U can open. Both its hierarchy parents are
    unreachable, so placement falls to the own group. The crumb query reduces
    the two groups to the lowest id before anyone asks whether it is openable,
    so the unopenable inbox wins and the trail drops to the App -- discarding a
    group the user can open, which §3.3 only permits when there is none.
    """
    result = await _browse(provider, "swm-z")
    assert _trail(result) == ["swm-app", "swm-drive", "swm-z"], result["scope"]


async def test_a_group_reachable_through_another_grantee_is_not_a_fallback(
    loaded_graph, provider
) -> None:
    """The fallback asks whether the group is reachable for *any* grantee.

    `pl-r12` is granted to the user; its group `pl-rg3` only to `group-g`, of
    which the user is a member. The group is reachable, so the record lists
    under it and never under the App.

    This is the behavioural pin for a scoping bug: the rule builder once named
    its grantee `g`, and the fallback arm has its own `g` in scope. Neo4j binds
    an EXISTS subquery's `g` to the outer one, so the check silently became
    "granted to the grantee who granted this record" — `pl-rg3` looked
    unreachable and `pl-r12` surfaced under the App. No error on that engine.
    """
    under_app = _ids(await _browse(provider, "pl-app"))
    assert "pl-r12" not in under_app, sorted(under_app)

    rows = {r["id"]: r for r in (await _browse(provider, "pl-rg3"))["partitions"][0]["rows"]}
    assert set(rows) == {"pl-r12"}, sorted(rows)
    assert rows["pl-r12"]["parentId"] == "pl-rg3", rows["pl-r12"]


async def test_a_grant_beneath_a_hidden_group_is_not_a_fallback(
    loaded_graph, provider
) -> None:
    """The narrow reading of §3.3: hideChildren stays hidden.

    `ex-hmsg` is granted and OPEN, and its own group `ex-hrg` is unreachable —
    exactly the fallback's shape — except that both sit beneath `ex-hidden`.
    Falling back would surface a hidden channel's message one level up.
    """
    found = _ids(await _browse(provider, "ex-app"))
    assert "ex-hmsg" not in found, sorted(found)
    assert "ex-hrg" not in found, sorted(found)
    # Guards the guard: the App does list something, so the absence above is
    # not an empty listing passing for the right answer.
    assert "ex-rg1" in found, sorted(found)


async def test_a_collection_lists_its_items_with_the_collections_role(
    loaded_graph, provider
) -> None:
    """BE-02, decision 52 — a collection is visible by its own grant, not by
    inheritance.

    `kb-1` carries a WRITER grant for this user and its items carry **no**
    inheritance edges at all. Under the ordinary rule every one of them is
    hidden, so a collection would browse as empty — which is why the role and
    the visibility are one decision rather than two: attaching a role to a list
    that is always empty would look like it worked.
    """
    partition = (await _browse(provider, "kb-1"))["partitions"][0]
    found = {row["id"] for row in partition["rows"]}
    assert found == {"kb-f1", "kb-r4"}, sorted(found)
    for row in partition["rows"]:
        assert row["userRole"] == "WRITER", row


async def test_connector_items_carry_no_role(loaded_graph, provider) -> None:
    """Decision 34 — a connector item has no permission of its own.

    The contract is a null `userRole`, not an absent field: the response model
    keeps the key and the caller renders nothing for it.
    """
    rows = (await _browse(provider, "pl-rg1"))["partitions"][0]["rows"]
    assert rows
    for row in rows:
        assert "userRole" in row, sorted(row)
        assert row["userRole"] is None, row


async def test_paging_reproduces_the_single_page_order(
    loaded_graph, provider
) -> None:
    """Keyset paging over a browse result, same contract as the root listing."""
    whole = [r["id"] for r in (await _browse(provider, "pl-rg1"))["partitions"][0]["rows"]]
    assert len(whole) >= 3, whole

    seen: list[str] = []
    after = None
    for _ in range(10):
        partition = (await _browse(provider, "pl-rg1", limit=1, after=after))["partitions"][0]
        seen.extend(r["id"] for r in partition["rows"])
        if not partition["hasMore"]:
            break
        after = _boundary(partition["rows"][-1])
    else:
        pytest.fail(f"paging never exhausted: {seen}")

    assert seen == whole, f"paged={seen}\nwhole={whole}"


async def test_an_admissible_node_with_no_children_is_still_admitted(
    loaded_graph, provider
) -> None:
    """An empty listing is not a 404.

    `ex1-r3` is a STRICT page inheriting from `ex1-rg1`, so the user may open
    it, and it has no children at all. The distinction is not cosmetic:
    `admitted=False` becomes a 404 whose body must not confirm the node exists,
    so returning it for a node the user can legitimately open denies access to
    something they hold.

    On Neo4j this was a real defect — `admitted` was read with
    `head(collect(admitted))` after an UNWIND, and an aggregation over zero rows
    yields null, which `bool()` turns into False. AQL binds it with `LET` before
    any rows exist and never had the flaw. The test runs on both because the
    *contract* is shared even where the failure surface is not.
    """
    result = await _browse(provider, "ex1-r3")
    assert result["scope"]["admitted"] is True, result["scope"]
    assert result["partitions"][0]["rows"] == []
    assert result["partitions"][0]["exhausted"] is True


@pytest.mark.parametrize(
    "node_id, children",
    [("pl-r6", set()), ("ex2-r6", {"ex2-r7", "ex2-r8"}), ("ex2-r7", set())],
    ids=["chain-top", "chain-top-with-children", "below-chain-top"],
)
async def test_a_node_below_a_gap_is_admitted(
    loaded_graph, provider, node_id, children
) -> None:
    """A granted OPEN node under an inaccessible ancestor can be opened (NV-28, SEC-01).

    Such a node is accessible on its own (decision 3, §3.6 arm 2), and so is
    anything admitted downward from it — `ex2-r7` inherits from `ex2-r6`, whose
    parent `ex2-r3` the user cannot see. An upward walk demanding one unbroken
    rule-passing path from the App breaks at that gap, so it would answer 404
    for nodes the listings themselves show the user.
    """
    result = await _browse(provider, node_id)
    assert result["scope"]["admitted"] is True, result["scope"]
    assert _ids(result) == children, sorted(_ids(result))


async def test_a_strict_node_below_a_seed_is_neither_listed_nor_admitted(
    loaded_graph, provider
) -> None:
    """§3.6 arm 2 — below a seed, nothing strict is admissible.

    `pl-r13` is STRICT and inherits from `pl-r6`, a seed under the gap `pl-r3`.
    Reading (b) needs every ancestor accessible and `pl-r3` is not, so the seed
    path that opens `pl-r6` must not carry its strict child — neither as a row
    when browsing `pl-r6`, nor as an admissible start node.
    """
    listed = _ids(await _browse(provider, "pl-r6"))
    assert "pl-r13" not in listed, sorted(listed)
    result = await _browse(provider, "pl-r13")
    assert result["scope"]["admitted"] is False, result["scope"]


@pytest.mark.parametrize(
    "node_id, children",
    [("kb-f1", {"kb-f2"}), ("kb-f2", {"kb-r3"})],
    ids=["folder", "nested-folder"],
)
async def test_a_folder_inside_a_collection_is_admitted(
    loaded_graph, provider, node_id, children
) -> None:
    """Decision 52 — inside a collection, access comes from the collection's grant.

    `kb-f1` and `kb-f2` neither inherit nor hold grants of their own, so the
    per-hop rule admits no path to them and neither is a seed. Browsing `kb-1`
    lists them; opening one must not answer 404 (NV-30).
    """
    result = await _browse(provider, node_id)
    assert result["scope"]["admitted"] is True, result["scope"]
    rows = result["partitions"][0]["rows"]
    assert {r["id"] for r in rows} == children, sorted(r["id"] for r in rows)
    assert all(r["userRole"] == "WRITER" for r in rows), rows


async def test_an_item_in_an_ungranted_collection_is_not_admitted(
    loaded_graph, provider
) -> None:
    """Decision 52's other half: no explicit grant on the collection, no way in (AC-49)."""
    result = await _browse(provider, "kb2-r1")
    assert result["scope"]["admitted"] is False, result["scope"]


async def test_an_ungranted_collection_lists_nothing(loaded_graph, provider) -> None:
    """AC-49 from the other end: browse the collection itself, not its item.

    Also PG-07's browse side. PG-07's own shape is sharper -- its `e2` is
    *granted to the user* inside a KB they cannot open, so the gate has to beat
    a direct grant -- and the fixture has no such node: `kb2-r1` carries no
    permission edge. That half stays uncovered, from search as well as browse.

    A collection's items are reached by BELONGS_TO, which every item carries
    whether or not the user may open the App. Only the gate stops them, so
    without it an ungranted collection would list its whole contents.

    Added because a mutation proved the gap: removing that gate failed no test,
    since `kb-2` appeared in the suite only as an id asserted *absent* from
    other results and nothing ever browsed it as a start node.
    """
    result = await _browse(provider, "kb-2")
    assert result["scope"]["admitted"] is False, result["scope"]
    assert result["partitions"][0]["rows"] == [], result["partitions"][0]["rows"]


@pytest.mark.parametrize(
    "node_id, children",
    [
        ("dec-app", {"dec-rg1"}),
        ("dec-rg1", {"dec-r1"}),
        ("dec-rg2", {"dec-r2", "dec-rg3"}),
        ("dec-rg3", {"dec-r3"}),
    ],
    ids=["AC-68-app-level", "AC-68-below-app-level", "AC-69-declared-group", "AC-69-nested-group"],
)
async def test_a_declaration_opens_everything_below_it(
    loaded_graph, provider, node_id, children
) -> None:
    """§3.8, decisions 31 and 39: nothing below a declaration is checked.

    `dec-app` is APP_LEVEL, and `dec-rg1`/`dec-r1` beneath it are RESTRICTED
    with no grant and no inheritance. `dec-rg2` is RECORD_GROUP_LEVEL and
    granted, and nothing below it inherits or holds a grant. Under the ordinary
    rule every one of these children is hidden.
    """
    result = await _browse(provider, node_id)
    assert result["scope"]["admitted"] is True, result["scope"]
    assert _ids(result) == children, sorted(_ids(result))


async def test_a_declaration_does_not_reach_an_undeclared_sibling(
    loaded_graph, provider
) -> None:
    """AC-69's controls: a declaration reaches neither a sibling nor an unopenable group.

    `dec-rg4` shares the App with the declared group but not its declaration.
    `dec-rg5` is declared too, but granted to nobody, so its declaration opens
    nothing: RECORD_GROUP_LEVEL applies only to a group the user can open.
    """
    assert _ids(await _browse(provider, "dec-rgl-app")) == {"dec-rg2"}
    assert (await _browse(provider, "dec-rg4"))["scope"]["admitted"] is False
    assert (await _browse(provider, "dec-r5"))["scope"]["admitted"] is False


def _trail(result) -> list[str]:
    return [crumb["id"] for crumb in result["scope"]["breadcrumbs"]]


@pytest.mark.parametrize(
    "node_id, trail",
    [
        ("pl-r6", ["pl-app", "pl-rg1", "pl-r6"]),
        ("ex2-r7", ["ex2-app", "ex2-rg1", "ex2-r6", "ex2-r7"]),
        ("pl-r9", ["pl-app", "pl-r9"]),
        ("kb-f2", ["kb-1", "kb-f1", "kb-f2"]),
        # A collection's *root* item, which has no hierarchy parent at all. The
        # nested case above never exercises that: kb-f2 has kb-f1, so the trail
        # is built from ordinary hierarchy edges and only the last step needs
        # the collection. Added after a mutation showed the crumb edge could be
        # disabled without failing anything.
        ("kb-f1", ["kb-1", "kb-f1"]),
        ("pl-app", ["pl-app"]),
        ("pl-r14", ["pl-app", "pl-rg4", "pl-r14"]),
    ],
    ids=["NV-28-chain-top", "SEC-01-below-chain-top", "NV-07-unreachable-group",
         "NV-30-collection", "NV-30-collection-root", "app", "granted-group-below-gap"],
)
async def test_breadcrumbs_follow_placement(
    loaded_graph, provider, node_id, trail
) -> None:
    """§3.3, decision 5: each level is where its child is listed, not its raw parent."""
    scope = (await _browse(provider, node_id))["scope"]
    assert [c["id"] for c in scope["breadcrumbs"]] == trail, scope
    assert scope["currentNode"]["id"] == node_id, scope
    expected_parent = trail[-2] if len(trail) > 1 else None
    assert (scope["parentNode"] or {}).get("id") == expected_parent, scope


@pytest.mark.parametrize(
    "node_id, hidden",
    [("pl-r6", ("pl-r3", "Gap folder")), ("ex2-r7", ("ex2-r3", "Folder 3"))],
    ids=["NV-28", "SEC-01"],
)
async def test_breadcrumbs_never_name_the_inaccessible_ancestor(
    loaded_graph, provider, node_id, hidden
) -> None:
    """The gap above a chain-top is absent from the whole response, id and name."""
    body = repr(await _browse(provider, node_id))
    for leak in hidden:
        assert leak not in body, f"{leak!r} leaked into the response for {node_id}"


@pytest.mark.parametrize(
    "via, trail",
    [
        (None, ["swm-app", "swm-drive", "swm-f1", "swm-x"]),
        ("swm-f1", ["swm-app", "swm-drive", "swm-f1", "swm-x"]),
        ("swm-inbox", ["swm-app", "swm-inbox", "swm-x"]),
        ("pl-rg1", ["swm-app", "swm-drive", "swm-f1", "swm-x"]),
    ],
    ids=["default-drive-wins", "via-folder", "via-shared-with-me", "via-not-a-parent"],
)
async def test_breadcrumbs_follow_the_navigated_parent(
    loaded_graph, provider, via, trail
) -> None:
    """NV-47, decision 79: a record with two parents shows the trail the user took.

    Without `via_parent_id`, or with one that is not a parent the record lists
    under, the drive location wins over Shared with Me (decision 67).
    """
    result = await _browse(provider, "swm-x", via_parent_id=via)
    assert _trail(result) == trail, result["scope"]


@pytest.mark.parametrize("parent_id", ["pl-rg1", "pl-app", "kb-1"])
async def test_both_backends_return_the_same_children(
    loaded_graph, neo4j_provider, arango_provider, parent_id
) -> None:
    """BE-01. Same request, same rows, same order, field for field.

    The parametrised tests above hold both backends to identical expectations;
    this pins them to each other, which catches a field that differs in *value*
    rather than in presence. The root listing's equivalent found exactly that —
    `sharingStatus` diverging because `null = null` is null in Cypher and true
    in AQL.

    One parent per arm: a record group (rule + seeds), an App (the fallback)
    and a collection (its role and visibility). Browsing only the group left
    the other two arms outside parity entirely — mutating the fallback on one
    engine failed nothing here.
    """
    cypher_result = await _browse(neo4j_provider, parent_id)
    aql_result = await _browse(arango_provider, parent_id)
    assert cypher_result["scope"] == aql_result["scope"], (
        f"scope diverges:\nneo4j={cypher_result['scope']}\narango={aql_result['scope']}"
    )
    cypher = cypher_result["partitions"][0]["rows"]
    aql = aql_result["partitions"][0]["rows"]
    assert cypher, f"{parent_id} listed nothing — the comparison would be vacuous"

    assert [r["id"] for r in cypher] == [r["id"] for r in aql], (
        f"order diverges:\nneo4j={[r['id'] for r in cypher]}\n"
        f"arango={[r['id'] for r in aql]}"
    )
    for left, right in zip(cypher, aql):
        assert set(left) == set(right), (
            f"{left['id']}: neo4j-only={sorted(set(left) - set(right))}, "
            f"arango-only={sorted(set(right) - set(left))}"
        )
        for field in ("name", "nodeType", "parentId", "parentType", "origin",
                      "connector", "hasChildren", "userRole", "sortKey", "nullRank"):
            assert left[field] == right[field], (
                f"{left['id']}.{field}: neo4j={left[field]!r} arango={right[field]!r}"
            )
