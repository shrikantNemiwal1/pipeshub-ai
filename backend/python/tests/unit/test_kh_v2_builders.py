"""The v2 query builders, checked as strings without a database.

These are the properties that do not need an engine to verify and that an
integration test would not localise well: that the two backends agree on what
may be sorted, that a sort direction is validated before it is interpolated,
and that the rule text keeps the shape two hard-won constraints require.

The rule's *meaning* is proven elsewhere, against real engines, and pinned to
the harness oracle in ``test_rule_equivalence.py``. Nothing here duplicates
that — these assertions are about text, not about access.
"""

from unittest.mock import MagicMock

import pytest

from app.services.graph_db.arango.arango_http_provider import ArangoHTTPProvider
from app.services.graph_db.neo4j.neo4j_provider import Neo4jProvider


@pytest.fixture
def neo4j() -> Neo4jProvider:
    return Neo4jProvider(logger=MagicMock(), config_service=MagicMock())


@pytest.fixture
def arango() -> ArangoHTTPProvider:
    return ArangoHTTPProvider(MagicMock(), MagicMock())


def test_both_providers_offer_the_same_sort_fields() -> None:
    """A field sortable on one backend and not the other is a silent 500.

    The same instance runs on whichever store it was installed with, so the
    allowlists are part of one contract rather than two.
    """
    assert Neo4jProvider._KH_V2_SORT_FIELDS == ArangoHTTPProvider._KH_V2_SORT_FIELDS


@pytest.mark.parametrize("provider", [Neo4jProvider, ArangoHTTPProvider])
@pytest.mark.parametrize(
    ("given", "expected"),
    [("asc", "ASC"), ("ASC", "ASC"), ("desc", "DESC"), ("Desc", "DESC"),
     (" desc ", "DESC")],
)
def test_sort_direction_is_normalised(provider, given: str, expected: str) -> None:
    assert provider._kh_v2_sort_direction(given) == expected


@pytest.mark.parametrize("provider", [Neo4jProvider, ArangoHTTPProvider])
@pytest.mark.parametrize(
    "given",
    ["", None, "ascending", "ASC; DROP", "RANDOM()", "asc desc", "1"],
)
def test_an_invalid_sort_direction_is_refused(provider, given) -> None:
    """The direction is interpolated, not bound.

    AQL takes ASC/DESC as a keyword rather than a value, so it cannot be a bind
    parameter -- which means the only thing standing between a caller's string
    and the query text is this check. Rejecting beats defaulting: a request
    that silently sorts by something else is harder to notice than a 400.
    """
    with pytest.raises(ValueError, match="sort order"):
        provider._kh_v2_sort_direction(given)


def test_the_cypher_rule_never_uses_a_case(neo4j: Neo4jProvider) -> None:
    """5.26 crashes its planner on an EXISTS over two QPP variables inside a CASE.

    The crash poisons the session rather than failing the single query, so this
    is worth asserting on the text: the rule must stay boolean algebra.
    """
    assert "CASE" not in neo4j._kh_v2_rule_cypher().upper()


@pytest.mark.parametrize("rule_text", ["cypher", "aql"])
def test_every_flag_is_read_through_a_null_guard(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider, rule_text: str
) -> None:
    """An absent property is not false, and an unguarded read would admit it."""
    if rule_text == "cypher":
        text = neo4j._kh_v2_rule_cypher()
        assert "coalesce(c.accessRule, 'OPEN')" in text
        assert "coalesce(c.isDeleted, false)" in text
        assert "coalesce(p.hideChildren, false)" in text
    else:
        text = arango._kh_v2_rule_aql("c", "p", "r")
        assert 'NOT_NULL(c.accessRule, "OPEN")' in text
        # Arango's form is an inequality against true, which is null-safe.
        assert "c.isDeleted != true" in text
        assert "p.hideChildren != true" in text


def test_the_cypher_rule_declares_the_parameters_it_needs(neo4j: Neo4jProvider) -> None:
    text = neo4j._kh_v2_rule_cypher()
    for param in ("$types", "$grantees", "$allowStrict", "$skipChecks"):
        assert param in text, f"{param} missing from the Cypher rule"


def test_the_aql_rule_declares_the_parameters_it_needs(arango: ArangoHTTPProvider) -> None:
    """Arango rejects a query that declares a bind parameter it never uses.

    The inverse -- using one that is never declared -- fails at execution, so
    both directions have to line up exactly.
    """
    text = arango._kh_v2_rule_aql("c", "p", "r")
    for param in ("@types", "@grantees", "@allowStrict", "@skipChecks"):
        assert param in text, f"{param} missing from the AQL rule"


def test_the_rule_namespaces_its_internal_variables(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """The rule is spliced into callers' queries, so its own names must not collide.

    The two engines fail very differently, which is why this is asserted on the
    text rather than left to an integration run. Arango rejects a variable
    declared again in a nested scope: "variable 'pm' is assigned multiple
    times". Neo4j raises nothing. An EXISTS subquery that names a variable
    already bound outside it binds to that variable, so a caller with its own
    `g` in scope turned "granted to any of the user's grantees" into "granted
    to this one grantee" — a different permission decision, silently.

    The browse App-fallback arm was the first caller with `g`/`pm` in scope
    around the rule. Arango's loud error is what exposed Neo4j's silent one,
    and the fixture could not have: the group involved had no grant at all, so
    both readings agreed.
    """
    cypher = neo4j._kh_v2_rule_cypher()
    assert "(g)" not in cypher and "g.id" not in cypher, cypher
    assert "kh_grantee" in cypher

    aql = arango._kh_v2_rule_aql("c", "p", "r")
    assert "FOR pm " not in aql and "FOR ip " not in aql, aql
    assert "kh_pm" in aql and "kh_ip" in aql


def test_the_rule_can_refuse_strict_nodes_inline(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """One browse query allows strictness from the App and refuses it below a seed.

    A single bound `allowStrict` cannot say both, so the builders take it as an
    expression. The default keeps every existing caller on the parameter.
    """
    assert neo4j._kh_v2_rule_cypher().count("$allowStrict") == 2
    refusing = neo4j._kh_v2_rule_cypher(allow_strict="false")
    assert "$allowStrict" not in refusing
    # Refusal is asserted as "no strict branch survives", not as a count of
    # `AND false`. Both engines now elide the two unreachable arms, because
    # 5.26 plans every branch it is given -- measured 296ms against 18ms for the
    # same probe without them, on a rule spliced in ten times per partition
    # query. Counting the dead text pinned the spelling; this pins the rule.
    # An unrecognised accessRule still matches no arm, which is the fail-closed
    # property decision 77 exists to protect.
    assert "'RESTRICTED'" not in refusing and "'STRICT'" not in refusing
    assert "'OPEN'" in refusing

    assert arango._kh_v2_rule_aql("c", "p", "r").count("@allowStrict") == 2
    refusing = arango._kh_v2_rule_aql("c", "p", "r", allow_strict="false")
    assert "@allowStrict" not in refusing
    assert '"RESTRICTED"' not in refusing and '"STRICT"' not in refusing
    assert '"OPEN"' in refusing


def test_all_three_access_rules_are_branched_on(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """Three explicit branches, so an undeclared value matches none of them."""
    cypher = neo4j._kh_v2_rule_cypher()
    aql = arango._kh_v2_rule_aql("c", "p", "r")
    for value in ("RESTRICTED", "STRICT", "OPEN"):
        assert value in cypher, f"{value} branch missing from the Cypher rule"
        assert value in aql, f"{value} branch missing from the AQL rule"


def test_the_rule_is_parameterised_on_its_variable_names(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """The same text serves the last hop and every earlier hop on the path.

    In AQL the rule is asserted over the whole path with positional
    expressions, so a builder that hardcoded ``c``/``p`` would filter the last
    hop only -- which leaks, measured at 7 nodes where Cypher returns 4.
    """
    cypher = neo4j._kh_v2_rule_cypher(child="kid", parent="mum", edge="edge")
    assert "kid.accessRule" in cypher and "mum.hideChildren" in cypher
    assert "edge.relationshipType" in cypher
    assert "c.accessRule" not in cypher, "default variable names leaked through"

    aql = arango._kh_v2_rule_aql(
        "path.vertices[i+1]", "path.vertices[i]", "path.edges[i]"
    )
    assert "path.vertices[i+1].accessRule" in aql
    assert "path.edges[i].relationshipType" in aql


# ---------------------------------------------------------------- sort builders


@pytest.mark.parametrize("provider", [Neo4jProvider, ArangoHTTPProvider])
@pytest.mark.parametrize("field", ["", "id", "password", "name; DROP", "createdat"])
def test_an_unknown_sort_field_is_refused(provider, field: str) -> None:
    """The field is interpolated into the query text, so it must be vetted.

    `createdat` is in the list on purpose: a near-miss of a real field is the
    realistic mistake, and defaulting it to name would sort by something the
    caller never asked for while looking like it worked.
    """
    with pytest.raises(ValueError, match="sort field"):
        provider._kh_v2_sort_property(field)


@pytest.mark.parametrize("provider", [Neo4jProvider, ArangoHTTPProvider])
def test_every_allowed_sort_field_resolves(provider) -> None:
    for field in provider._KH_V2_SORT_FIELDS:
        assert provider._kh_v2_sort_property(field)


def test_the_sort_projects_the_comparator_fields(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """The cursor stores sortKey/nullRank, so the query has to return them."""
    cypher = neo4j._kh_v2_sort_cypher("name", "asc")
    assert "AS sortKey" in cypher and "AS nullRank" in cypher

    aql = arango._kh_v2_sort_aql("name", "asc")
    assert "LET sortKey" in aql and "LET nullRank" in aql


@pytest.mark.parametrize("order", ["asc", "desc"])
def test_nulls_and_ties_do_not_flip_with_direction(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider, order: str
) -> None:
    """nullRank first and id last, both ascending, whichever way the value sorts.

    This is the property that makes `prev` return exactly the page the user came
    from, and it has to hold identically in kh_merge -- the merge orders rows
    across partitions using the very keys these clauses emit.
    """
    direction = order.upper()
    cypher = neo4j._kh_v2_sort_cypher("name", order)
    assert f"ORDER BY nullRank ASC, sortKey {direction}, node.id ASC" in cypher

    aql = arango._kh_v2_sort_aql("name", order)
    assert f"SORT nullRank ASC, sortKey {direction}, node._key ASC" in aql


@pytest.mark.parametrize("order, flipped", [("asc", "DESC"), ("desc", "ASC")])
def test_a_reversed_sort_flips_every_term(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider, order, flipped
) -> None:
    """A previous page takes the rows just before the boundary, nearest first.

    That needs all three terms reversed. Flipping only the value would walk the
    null bucket and the id tiebreak the wrong way, so `prev` would skip or
    repeat rows at exactly the boundaries where nulls or ties sit (PG-13).
    """
    cypher = neo4j._kh_v2_sort_cypher("name", order, reverse=True)
    assert f"ORDER BY nullRank DESC, sortKey {flipped}, node.id DESC" in cypher

    aql = arango._kh_v2_sort_aql("name", order, reverse=True)
    assert f"SORT nullRank DESC, sortKey {flipped}, node._key DESC" in aql


def test_the_name_sort_is_case_insensitive(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """§3.9 requires case-insensitive ordering by name.

    Sorting the raw property puts every capital ahead of every lowercase letter
    on both engines, so a listing looks alphabetised right up until someone
    capitalises a folder name — and then looks arbitrary.

    Lowering in the sort expression rather than at comparison time matters for
    paging: this expression's output *is* the sortKey the cursor stores and the
    keyset compares, so the ordering and the resume point cannot disagree.
    """
    assert "toLower(node.name)" in neo4j._kh_v2_sort_cypher("name", "asc")
    assert "LOWER(node.name)" in arango._kh_v2_sort_aql("name", "asc")


def test_the_name_falls_back_to_record_and_group_names(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """Only an App stores `name`.

    A record stores `recordName`, a record group `groupName`. Reading `name`
    alone leaves every browse row nameless — and the root listing cannot reveal
    it, because Apps are the one node type that does store `name`.
    """
    cypher = neo4j._kh_v2_projection_cypher()
    assert "coalesce(node.name, node.recordName, node.groupName)" in cypher

    aql = arango._kh_v2_projection_aql()
    assert "NOT_NULL(node.name, node.recordName, node.groupName)" in aql


def test_the_arango_name_sort_does_not_turn_null_into_empty(
    arango: ArangoHTTPProvider,
) -> None:
    """`LOWER(null)` is `""` in AQL and null in Cypher's `toLower()`.

    Unguarded, a nameless node gets nullRank 0 on Arango and 1 on Neo4j, so
    nulls sort first on one backend and last on the other — against §3.9's
    "nulls last in both directions", and invisible to any single-backend test.
    """
    text = arango._kh_v2_sort_aql("name", "asc")
    assert "== null ? null : LOWER(" in text, text


def test_only_the_name_sort_is_lowered(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """Timestamps and sizes are compared as stored; lowering them is nonsense."""
    assert "toLower" not in neo4j._kh_v2_sort_cypher("createdAt", "asc")
    assert "toLower" not in neo4j._kh_v2_sort_cypher("sizeInBytes", "desc")
    assert "LOWER" not in arango._kh_v2_sort_aql("createdAt", "asc")
    assert "LOWER" not in arango._kh_v2_sort_aql("sizeInBytes", "desc")


def test_the_sort_field_is_interpolated_never_bound(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """v1 binds both field and direction; AQL cannot bind a SORT keyword."""
    cypher = neo4j._kh_v2_sort_cypher("createdAt", "desc")
    assert "node.createdAt" in cypher
    assert "$sort_field" not in cypher and "$sort_dir" not in cypher

    aql = arango._kh_v2_sort_aql("createdAt", "desc")
    assert "node.createdAt" in aql
    assert "@sort_field" not in aql and "@sort_dir" not in aql


def test_an_invalid_order_is_refused_by_the_sort_builders(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    with pytest.raises(ValueError, match="sort order"):
        neo4j._kh_v2_sort_cypher("name", "sideways")
    with pytest.raises(ValueError, match="sort order"):
        arango._kh_v2_sort_aql("name", "sideways")


def test_the_cypher_sort_carries_named_variables(neo4j: Neo4jProvider) -> None:
    """Cypher's WITH drops whatever it does not name.

    A listing threads a total through the sort, and later a partition id. If the
    builder's WITH omitted them the query would fail to compile — but only once
    someone actually composed it, which is why this is asserted on the text
    rather than discovered against a live server.
    """
    text = neo4j._kh_v2_sort_cypher("name", "asc", carry=("total", "partitionId"))
    # The carried prefix, not the whole rendered line. An earlier version quoted
    # the line verbatim -- sort key expression included -- so it broke the moment
    # the name sort legitimately became case-insensitive, failing a test whose
    # subject is the carry mechanism for a reason unrelated to it.
    with_lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("WITH ")]
    assert len(with_lines) == 2, text
    for line in with_lines:
        assert line.startswith("WITH node, total, partitionId,"), line


def test_the_cypher_sort_takes_the_keyset_filter(neo4j: Neo4jProvider) -> None:
    """The keyset WHERE has to sit between the final WITH and the ORDER BY.

    Cypher hangs WHERE off a WITH, not off an ORDER BY, so appending the
    predicate after the sort block would filter rows that are already ordered
    and re-project them.
    """
    text = neo4j._kh_v2_sort_cypher("name", "asc", where="nullRank > $ks_null_rank")
    assert "WHERE nullRank > $ks_null_rank" in text
    assert text.index("WHERE") < text.index("ORDER BY"), (
        "the keyset filter must precede the ordering"
    )


def test_the_cypher_sort_emits_no_filter_when_none_is_given(
    neo4j: Neo4jProvider,
) -> None:
    assert "WHERE" not in neo4j._kh_v2_sort_cypher("name", "asc")


def test_the_cypher_sort_carries_nothing_by_default(neo4j: Neo4jProvider) -> None:
    text = neo4j._kh_v2_sort_cypher("name", "asc")
    with_lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("WITH ")]
    assert len(with_lines) == 2, text
    assert all(ln.startswith("WITH node,") for ln in with_lines), with_lines
    assert "total" not in text and "partitionId" not in text
    assert "WITH node, sortKey," in text


def test_the_sort_honours_the_node_variable(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    cypher = neo4j._kh_v2_sort_cypher("name", "asc", node_var="row")
    assert "row.name" in cypher and "row.id ASC" in cypher
    aql = arango._kh_v2_sort_aql("name", "asc", node_var="row")
    assert "row.name" in aql and "row._key ASC" in aql


# -------------------------------------------------------------- keyset builders


@pytest.mark.parametrize(
    ("order", "direction", "rank_op", "value_op", "id_op"),
    [
        ("asc", "next", ">", ">", ">"),
        ("desc", "next", ">", "<", ">"),
        ("asc", "prev", "<", "<", "<"),
        ("desc", "prev", "<", ">", "<"),
    ],
)
def test_the_keyset_compares_the_way_the_page_is_ordered(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider,
    order: str, direction: str, rank_op: str, value_op: str, id_op: str,
) -> None:
    """Only the *value* comparison follows the sort direction.

    nullRank and the id tiebreak track the page direction instead, because both
    are always ascending within a page: reversing them with the sort would put
    nulls at the wrong end and flip tied rows, and `prev` would then return
    something other than the page the user came from.
    """
    cypher = neo4j._kh_v2_keyset_cypher(order, direction)
    assert f"nullRank {rank_op} $ks_null_rank" in cypher
    assert f"sortKey {value_op} $ks_sort_key" in cypher
    assert f"node.id {id_op} $ks_id" in cypher

    aql = arango._kh_v2_keyset_aql(order, direction)
    assert f"nullRank {rank_op} @ks_null_rank" in aql
    assert f"sortKey {value_op} @ks_sort_key" in aql
    assert f"node._key {id_op} @ks_id" in aql


def test_the_null_bucket_is_resumed_by_id_alone(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """Every row in the null bucket has a null sort key, so only the id orders it.

    In Cypher `null = null` is null, not true, so a "values equal, fall through
    to the id" term matches nothing there: without this arm the walk skips the
    entire null bucket and those records appear on no page at all.
    """
    cypher = neo4j._kh_v2_keyset_cypher("asc", "next")
    bucket = cypher.split("$ks_null_rank = 1", 1)[1].split("OR (", 1)[0]
    assert "node.id" in bucket, bucket
    assert "sortKey" not in bucket, f"the null bucket must not compare values: {bucket}"

    aql = arango._kh_v2_keyset_aql("asc", "next")
    bucket = aql.split("@ks_null_rank == 1", 1)[1].split("OR (", 1)[0]
    assert "node._key" in bucket, bucket
    assert "sortKey" not in bucket, f"the null bucket must not compare values: {bucket}"


@pytest.mark.parametrize("provider_name", ["neo4j", "arango"])
@pytest.mark.parametrize("direction", ["", "forward", "NEXT", "back", None])
def test_an_unknown_cursor_direction_is_refused(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider,
    provider_name: str, direction,
) -> None:
    build = (
        neo4j._kh_v2_keyset_cypher if provider_name == "neo4j"
        else arango._kh_v2_keyset_aql
    )
    with pytest.raises(ValueError, match="cursor direction"):
        build("asc", direction)


def test_the_keyset_validates_the_sort_order_too(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    with pytest.raises(ValueError, match="sort order"):
        neo4j._kh_v2_keyset_cypher("sideways", "next")
    with pytest.raises(ValueError, match="sort order"):
        arango._kh_v2_keyset_aql("sideways", "next")


def test_the_keyset_binds_its_boundary_rather_than_interpolating_it(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """The boundary is user-supplied data, unlike the sort field and direction."""
    cypher = neo4j._kh_v2_keyset_cypher("asc", "next")
    for param in ("$ks_null_rank", "$ks_sort_key", "$ks_id"):
        assert param in cypher
    aql = arango._kh_v2_keyset_aql("asc", "next")
    for param in ("@ks_null_rank", "@ks_sort_key", "@ks_id"):
        assert param in aql


def test_the_keyset_honours_the_node_variable(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    assert "row.id" in neo4j._kh_v2_keyset_cypher("asc", "next", node_var="row")
    assert "row._key" in arango._kh_v2_keyset_aql("asc", "next", node_var="row")


def test_the_arango_tiebreak_can_target_a_projected_row(
    arango: ArangoHTTPProvider,
) -> None:
    """A projected row carries `id` and has no `_key`.

    Sorting happens after projection — several sort fields are computed, not
    stored — so the tiebreak has to read the domain id off the projected row.
    Defaulting to `_key` keeps raw-document callers working; Cypher needs no
    equivalent because the domain id is `id` in both shapes.

    If this were wrong the query would not error: `row._key` on a projected
    object is simply null, so every row would tie and the order would come out
    arbitrary — and paging would silently skip and repeat rows.
    """
    sort = arango._kh_v2_sort_aql("name", "asc", id_expr="node.id")
    assert "SORT nullRank ASC, sortKey ASC, node.id ASC" in sort
    assert "._key" not in sort

    keyset = arango._kh_v2_keyset_aql("asc", "next", id_expr="node.id")
    assert "node.id >" in keyset
    assert "._key" not in keyset


def test_the_arango_tiebreak_still_defaults_to_the_document_key(
    arango: ArangoHTTPProvider,
) -> None:
    assert "node._key ASC" in arango._kh_v2_sort_aql("name", "asc")
    assert "node._key >" in arango._kh_v2_keyset_aql("asc", "next")


def test_the_arango_sort_takes_the_keyset_filter(arango: ArangoHTTPProvider) -> None:
    """The FILTER must fall between the LETs and the SORT.

    AQL's FILTER is standalone, so it looks as though it could precede the sort
    block entirely — but the keyset predicate reads sortKey and nullRank, which
    those LETs define. Placing it before the block would reference unbound
    variables; placing it after the SORT would filter rows already ordered.
    """
    text = arango._kh_v2_sort_aql("name", "asc", where="nullRank > @ks_null_rank")
    assert "FILTER nullRank > @ks_null_rank" in text
    assert text.index("LET nullRank") < text.index("FILTER") < text.index("SORT"), text


def test_the_arango_sort_emits_no_filter_when_none_is_given(
    arango: ArangoHTTPProvider,
) -> None:
    assert "FILTER" not in arango._kh_v2_sort_aql("name", "asc")


# ------------------------------------------------------------- filter builders

FULL_FILTERS = dict(
    search_query="report",
    node_types=["record"],
    record_types=["FILE"],
    indexing_status=["COMPLETED"],
    created_at={"gte": 0, "lte": 5},
    updated_at={"gte": 1},
    size={"gte": 0, "lte": 10},
    origins=["CONNECTOR"],
    connector_ids=["conn-1"],
    only_containers=True,
    record_group_ids=["rg-1"],
)


def _builders(neo4j: Neo4jProvider, arango: ArangoHTTPProvider):
    return (("neo4j", "$", neo4j._kh_v2_filters_cypher),
            ("arango", "@", arango._kh_v2_filters_aql))


def test_a_zero_bound_produces_a_condition(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """v1 tests bounds for truth, so `0` silently drops the whole filter.

    A zero is a legitimate bound — "at least zero bytes", and the epoch for a
    date — and dropping it does not narrow the result, it *widens* it, which is
    why nobody notices. v1 gets this right on Arango and wrong on Neo4j, so the
    same request returns different sets depending on the store.
    """
    for name, sigil, build in _builders(neo4j, arango):
        conditions, params = build(size={"gte": 0}, created_at={"gte": 0})
        text = " ".join(conditions)
        assert f"{sigil}kh_size_gte" in text, f"{name}: zero size bound dropped"
        assert f"{sigil}kh_created_gte" in text, f"{name}: epoch date bound dropped"
        assert params["kh_size_gte"] == 0
        assert params["kh_created_gte"] == 0


def test_the_search_term_is_matched_literally(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """PG-40: no LIKE, so `%` and `_` in a search term are not wildcards.

    v1's AQL builds `LIKE CONCAT('%', ..., '%')`, where `a_b` matches `axb` and
    `50% off` matches nearly everything — while Cypher's CONTAINS is literal.
    """
    for name, _sigil, build in _builders(neo4j, arango):
        conditions, params = build(search_query="50% off_A")
        text = " ".join(conditions)
        assert "LIKE" not in text.upper(), f"{name}: search reintroduced LIKE"
        assert "CONTAINS" in text.upper(), f"{name}: expected a substring match"
        assert params["kh_search"] == "50% off_a", "the term should be lowercased as-is"


def test_only_containers_is_applied_by_both(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """v1 accepts this in the filter builder and ignores it on Neo4j."""
    for name, _sigil, build in _builders(neo4j, arango):
        with_flag = " ".join(build(only_containers=True)[0])
        without = " ".join(build(only_containers=False)[0])
        assert "hasChildren" in with_flag, f"{name}: only_containers ignored"
        assert "hasChildren" not in without, f"{name}: applied when not asked"


def test_a_record_without_a_size_is_outside_every_size_window(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """Unknown size is not zero; it must not fall into a `lte` window."""
    for name, _sigil, build in _builders(neo4j, arango):
        text = " ".join(build(size={"lte": 100})[0])
        assert ("sizeInBytes IS NOT NULL" in text
                or "sizeInBytes != null" in text), f"{name}: null sizes not excluded"


def test_every_declared_parameter_is_referenced(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """Arango rejects a query declaring a bind parameter it never uses.

    The mirror case — referencing one that was never declared — fails at
    execution instead, so both directions are asserted here rather than
    discovered against a live server.
    """
    import re

    for name, sigil, build in _builders(neo4j, arango):
        conditions, params = build(**FULL_FILTERS)
        text = " ".join(conditions)
        referenced = set(re.findall(rf"{re.escape(sigil)}(kh_\w+)", text))
        assert referenced == set(params), (
            f"{name}: declared-but-unused={sorted(set(params) - referenced)}, "
            f"used-but-undeclared={sorted(referenced - set(params))}"
        )


def test_parameters_are_namespaced(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """These conditions are spliced beside the traversal's own bind variables."""
    for name, _sigil, build in _builders(neo4j, arango):
        _, params = build(**FULL_FILTERS)
        assert params, f"{name}: expected parameters"
        assert all(k.startswith("kh_") for k in params), sorted(params)


def test_with_no_filters_only_placeholders_are_excluded(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    for name, _sigil, build in _builders(neo4j, arango):
        conditions, params = build()
        assert params == {}, f"{name}: {params}"
        assert len(conditions) == 1, f"{name}: {conditions}"
        assert "isPlaceholder" in conditions[0]


def test_both_backends_emit_the_same_filters(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """BE-01 at builder level: same request, same set of restrictions — differing
    only in dialect."""
    cypher_conditions, cypher_params = neo4j._kh_v2_filters_cypher(**FULL_FILTERS)
    aql_conditions, aql_params = arango._kh_v2_filters_aql(**FULL_FILTERS)
    assert len(cypher_conditions) == len(aql_conditions), (
        f"neo4j={cypher_conditions}\narango={aql_conditions}"
    )
    assert cypher_params == aql_params


# --------------------------------------------------------- projection builders


def _projection_fields(text: str) -> set[str]:
    import re

    return set(re.findall(r"^\s*(\w+):", text, re.MULTILINE))


def test_both_projections_emit_the_same_fields(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """A field on one backend and absent on the other is a silent contract break.

    The same instance runs on whichever store it was installed with, so the row
    shape is one contract rather than two. Nothing in a single-backend test
    would notice a field quietly missing from the other.
    """
    cypher = _projection_fields(neo4j._kh_v2_projection_cypher())
    aql = _projection_fields(arango._kh_v2_projection_aql())
    assert cypher == aql, (
        f"neo4j-only={sorted(cypher - aql)}, arango-only={sorted(aql - cypher)}"
    )
    assert "id" in cypher and "name" in cypher


def test_ids_are_returned_bare(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """API-07: no collection prefix, on either backend.

    v1 emits `'apps/' + id` at six sites, leaking Arango's collection/key
    addressing into responses the Neo4j backend also serves, so a client cannot
    round-trip a parentId it was handed.
    """
    cypher = neo4j._kh_v2_projection_cypher()
    for prefix in ("'apps/'", "'recordGroups/'", "'records/'"):
        assert prefix not in cypher, f"collection prefix leaked: {prefix}"

    aql = arango._kh_v2_projection_aql()
    assert "._key" in aql, "Arango ids must come from _key"
    assert "._id" not in aql, "the collection/key form must not reach the response"


def test_the_parent_triple_travels_with_the_row(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """D69 — id, type and name, so a search hit renders 'in <folder>'."""
    for text in (neo4j._kh_v2_projection_cypher(), arango._kh_v2_projection_aql()):
        fields = _projection_fields(text)
        assert {"parentId", "parentType", "parentName"} <= fields, sorted(fields)


def test_the_comparator_output_is_returned(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """The cursor resumes from these and the merge orders partitions by them."""
    for text in (neo4j._kh_v2_projection_cypher(), arango._kh_v2_projection_aql()):
        fields = _projection_fields(text)
        assert {"sortKey", "nullRank"} <= fields, sorted(fields)


def test_overrides_replace_the_default_expressions(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """Several fields are computed per branch, not read from storage.

    An App has no `origin` property — v1 derives it from `type` — so the root
    listing has to supply its own expression for it.
    """
    cypher = neo4j._kh_v2_projection_cypher(overrides={
        "nodeType": "'app'",
        "origin": "CASE WHEN app.type = 'KB' THEN 'COLLECTION' ELSE 'CONNECTOR' END",
        "hasChildren": "has_children",
    })
    assert "nodeType: 'app'" in cypher
    assert "hasChildren: has_children" in cypher
    assert "'COLLECTION'" in cypher
    assert "nodeType: 'record'" not in cypher, "the default leaked through"

    aql = arango._kh_v2_projection_aql(overrides={
        "nodeType": '"app"', "hasChildren": "has_children"
    })
    assert 'nodeType: "app"' in aql
    assert "hasChildren: has_children" in aql


def test_an_unknown_override_is_refused(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """A mistyped key would add a field to one backend and pass the parity test.

    Parity compares the two backends against each other, so a field added to
    only one side is caught — but a field added to *both* by the same typo, or
    to one branch of one backend, is not. Refusing unknown keys closes that at
    the source rather than hoping a later assertion notices.
    """
    for build in (neo4j._kh_v2_projection_cypher, arango._kh_v2_projection_aql):
        with pytest.raises(ValueError, match="Unknown projection field"):
            build(overrides={"nodetype": "'app'"})


def test_overrides_never_change_the_field_set(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    for build in (neo4j._kh_v2_projection_cypher, arango._kh_v2_projection_aql):
        base = _projection_fields(build())
        overridden = _projection_fields(build(overrides={"hasChildren": "true"}))
        assert base == overridden, sorted(base ^ overridden)


def test_the_projection_honours_the_node_and_parent_variables(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    cypher = neo4j._kh_v2_projection_cypher(node_var="row", parent_var="mum")
    assert "row.name" in cypher and "mum.name" in cypher
    aql = arango._kh_v2_projection_aql(node_var="row", parent_var="mum")
    assert "row.name" in aql and "mum._key" in aql


def test_timestamps_and_flags_are_null_guarded(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """An absent property is not zero and not false."""
    cypher = neo4j._kh_v2_projection_cypher()
    assert "coalesce(node.createdAtTimestamp, 0)" in cypher
    assert "coalesce(node.isInternal, false)" in cypher

    aql = arango._kh_v2_projection_aql()
    assert "NOT_NULL(node.createdAtTimestamp, 0)" in aql
    assert "NOT_NULL(node.isInternal, false)" in aql


def test_the_comparator_fields_can_be_omitted(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    """The base row exists before its sort key does.

    A listing must project the row first and sort second, because nodeType,
    origin and connector are computed rather than stored and the sort reads them
    off the projected map. sortKey and nullRank therefore cannot be part of that
    first projection — they are merged in by the final RETURN.
    """
    for name, build in (("neo4j", neo4j._kh_v2_projection_cypher),
                        ("arango", arango._kh_v2_projection_aql)):
        text = build(include_comparator=False)
        fields = _projection_fields(text)
        assert "sortKey" not in fields, f"{name}: {sorted(fields)}"
        assert "nullRank" not in fields, f"{name}: {sorted(fields)}"
        # Everything else must survive, or the two forms describe different rows.
        assert {"id", "name", "parentId", "userRole", "hasChildren"} <= fields
        # A dangling comma before the closing brace is a syntax error in both
        # dialects and would only surface when the query is executed.
        assert not text.rstrip().rstrip("}").rstrip().endswith(","), (
            f"{name}: trailing comma left by the omission"
        )


def test_omitting_the_comparator_changes_nothing_else(
    neo4j: Neo4jProvider, arango: ArangoHTTPProvider
) -> None:
    for build in (neo4j._kh_v2_projection_cypher, arango._kh_v2_projection_aql):
        full = _projection_fields(build())
        base = _projection_fields(build(include_comparator=False))
        assert full - base == {"sortKey", "nullRank"}, sorted(full - base)
