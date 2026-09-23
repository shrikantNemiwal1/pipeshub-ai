"""`KnowledgeHubService.get_nodes` on the v2 read path.

The v2 queries decide permission, placement and breadcrumbs themselves, so the
service's job shrank to three things these tests pin: pick the right query for
the request, turn a cursor into the next page (and refuse one that is not the
caller's), and build the response — including the 404 that must not say what it
is hiding.

The provider is a mock here on purpose: the queries themselves are proven
against real Neo4j and Arango in `tests/integration/graph_permissions`, and
repeating that here would test the fixture, not the wiring.
"""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from app.connectors.sources.localKB.handlers.kh_search import SearchPage
from app.connectors.sources.localKB.handlers.knowledge_hub_service import (
    _MAX_FILTER_SOURCES,
    KnowledgeHubService,
)


@pytest.fixture
def logger():
    log = logging.getLogger("test_kh_service_v2")
    log.setLevel(logging.CRITICAL)
    return log


@pytest.fixture
def config_service():
    config = AsyncMock()
    config.get_config.return_value = {"scopedJwtSecret": "test-secret"}
    return config


@pytest.fixture
def provider():
    graph = AsyncMock()
    graph.get_user_by_user_id.return_value = {"_key": "uk1"}
    graph.get_knowledge_hub_access_context_v2.return_value = {
        "grantee_ids": ["uk1", "group-g"],
        "gated_app_ids": ["app1", "kb-1"],
    }
    graph.get_knowledge_hub_access_v3.return_value = {
        "grantee_ids": ["uk1", "group-g"],
        "gated_app_ids": ["app1", "kb-1"],
        "by_connector": {"app1": [], "kb-1": []},
    }
    return graph


@pytest.fixture
def service(logger, provider, config_service):
    return KnowledgeHubService(
        logger=logger, graph_provider=provider, config_service=config_service
    )


def row(row_id: str, name: str, *, node_type: str = "record", parent: str | None = None,
        parent_type: str | None = None) -> dict:
    return {
        "id": row_id, "name": name, "nodeType": node_type,
        "sortKey": name.lower(), "nullRank": 0,
        "parentId": parent, "parentType": parent_type, "parentName": "Parent",
        "origin": "COLLECTION", "createdAt": 1, "updatedAt": 2, "hasChildren": False,
    }


def envelope(rows: list[dict], *, total: int | None = None, has_more: bool = False,
             ids: list[dict] | None = None, scope: dict | None = None) -> dict:
    return {
        "partitions": [{
            "partitionId": "p1", "partitionKind": "BROWSE", "appId": None,
            "rows": rows, "hasMore": has_more, "exhausted": not has_more,
            "total": len(rows) if total is None else total,
            "ids": ids or [], "countsByType": None,
        }],
        "scope": scope,
    }


def page_of(rows: list[dict], *, total: int | None = None, has_more: bool = False,
            scope: dict | None = None, counts: dict | None = None) -> dict:
    return {
        "rows": rows,
        "hasMore": has_more,
        "total": len(rows) if total is None else total,
        "counts": counts,
        "scope": scope,
    }


def admitted_scope(node_id: str = "p1", node_type: str = "recordGroup") -> dict:
    return {
        "admitted": True,
        "nodeId": node_id,
        "currentNode": {"id": node_id, "name": "Current", "nodeType": node_type},
        "parentNode": {"id": "app1", "name": "App One", "nodeType": "app"},
        "breadcrumbs": [
            {"id": "app1", "name": "App One", "nodeType": "app"},
            {"id": node_id, "name": "Current", "nodeType": node_type},
        ],
    }


# --------------------------------------------------------------- routing

@pytest.mark.asyncio
async def test_a_root_listing_uses_the_root_query(service, provider) -> None:
    """BE-08: the request mode picks the v2 query, with no runtime switch."""
    provider.get_knowledge_hub_root_nodes_v2.return_value = envelope(
        [row("app1", "App One", node_type="app")]
    )
    result = await service.get_nodes(user_id="u1", org_id="o1")

    assert result.success is True
    assert [item.id for item in result.items] == ["app1"]
    provider.get_knowledge_hub_children_v2.assert_not_called()
    # The gate is the access context's, not the old owned-plus-shared list.
    assert provider.get_knowledge_hub_root_nodes_v2.call_args.kwargs["user_app_ids"] == [
        "app1", "kb-1",
    ]


@pytest.mark.asyncio
async def test_browsing_a_node_uses_the_children_query_unflattened(service, provider) -> None:
    """BE-08: browse routes to the children query, and asks it not to flatten."""
    provider.get_knowledge_hub_connector_page_v3.return_value = page_of(
        [row("r1", "Record")], scope=admitted_scope()
    )
    result = await service.get_nodes(
        user_id="u1", org_id="o1", parent_id="p1", parent_type="recordGroup"
    )

    assert result.success is True
    kwargs = provider.get_knowledge_hub_connector_page_v3.call_args.kwargs
    assert kwargs["start_id"] == "p1" and kwargs["flatten"] is False
    assert kwargs["include_scope"] is True
    assert kwargs["grantee_ids"] == ["uk1", "group-g"]


@pytest.mark.asyncio
async def test_a_filtered_scoped_request_flattens(service, provider) -> None:
    """BE-08. A filter means "search below here", which is the same query with depth."""
    provider.get_knowledge_hub_connector_page_v3.return_value = page_of(
        [], scope=admitted_scope()
    )
    await service.get_nodes(
        user_id="u1", org_id="o1", parent_id="p1", parent_type="recordGroup", q="report"
    )
    kwargs = provider.get_knowledge_hub_connector_page_v3.call_args.kwargs
    assert kwargs["flatten"] is True
    assert kwargs["filters"]["search_query"] == "report"


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason="BE-12: depth is accepted by get_nodes and never read; "
           "the connector page query has no depth parameter to receive it",
)
async def test_depth_bounds_a_flattened_request(service, provider) -> None:
    """BE-12, decision 33: the agent navigator asks for a bounded flatten.

    `navigator.py` clamps depth to its own maximum, passes it, and then computes
    each row's nesting level from it — so it believes the bound was applied. v2
    drops it and flattens the whole subtree up to `_KH_V2_MAX_DEPTH` (50), so an
    agent asking for two levels can receive fifty. No error is raised anywhere,
    which is why nothing caught it.

    Marked xfail rather than fixed: the remedy is a product decision — bound the
    traversal in the provider, clamp by level in the service after fetching, or
    retire the parameter and update the navigator.
    """
    provider.get_knowledge_hub_connector_page_v3.return_value = page_of(
        [row("r1", "Record")], scope=admitted_scope()
    )
    await service.get_nodes(
        user_id="u1", org_id="o1", parent_id="p1", parent_type="recordGroup",
        flattened=True, depth=2,
    )
    kwargs = provider.get_knowledge_hub_connector_page_v3.call_args.kwargs
    assert kwargs.get("depth") == 2, sorted(kwargs)


@pytest.mark.asyncio
async def test_the_user_override_point_still_decides_the_user(
    logger, provider, config_service
) -> None:
    """BE-13: `edition_config.knowledge_hub_service_factory` swaps the whole
    service (`connectors_main.py:887`), and an enterprise build overrides
    `_resolve_user` for an org-scoped lookup. If the base stopped calling it,
    that deployment would silently resolve users the community way.

    The assertion reads the call rather than its keyword names, so it pins that
    the override was consulted without pinning an argument spelling.
    """
    class _EEService(KnowledgeHubService):
        async def _resolve_user(self, user_id: str, org_id: str):
            return {"_key": "ee-user-key"}

    service = _EEService(
        logger=logger, graph_provider=provider, config_service=config_service
    )
    provider.get_knowledge_hub_root_nodes_v2.return_value = envelope([])
    await service.get_nodes(user_id="u1", org_id="o1")

    call = provider.get_knowledge_hub_access_context_v2.call_args
    assert "ee-user-key" in list(call.args) + list(call.kwargs.values()), call


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason="BE-13: _get_user_app_ids is defined as an override point but has no "
           "caller in the service package; v2 gates on the access context instead",
)
async def test_the_app_gate_override_point_is_still_consulted(
    logger, provider, config_service
) -> None:
    """BE-13's other half: an enterprise build overriding only `_get_user_app_ids`
    should still change which apps the request may reach.

    It does not. The method survives the v2 rewrite, so the override compiles and
    looks effective, but nothing calls it — the gate comes from
    `get_knowledge_hub_access_context_v2`. An EE deployment relying on it would
    widen or narrow nothing, with no error to notice.
    """
    consulted: list[tuple[str, str]] = []

    class _EEService(KnowledgeHubService):
        async def _get_user_app_ids(self, user_key: str, org_id: str) -> list[str]:
            consulted.append((user_key, org_id))
            return ["ee-app"]

    service = _EEService(
        logger=logger, graph_provider=provider, config_service=config_service
    )
    provider.get_knowledge_hub_root_nodes_v2.return_value = envelope([])
    await service.get_nodes(user_id="u1", org_id="o1")

    assert consulted, "the override point was never consulted"


@pytest.mark.asyncio
async def test_a_global_search_is_partitioned(service, provider) -> None:
    """BE-08: the one mode that fans out across partitions and merges (§3.9)."""
    page = SearchPage(rows=[row("r1", "Report")], total=1, counts_by_type={"record": 1},
                      start_index=1, end_index=1, next_cursor=None, prev_cursor=None)
    with patch(
        "app.connectors.sources.localKB.handlers.knowledge_hub_service.search_page",
        AsyncMock(return_value=page),
    ) as searched:
        result = await service.get_nodes(user_id="u1", org_id="o1", q="report")

    assert [item.id for item in result.items] == ["r1"]
    assert searched.call_args.kwargs["filters"]["search_query"] == "report"
    # The rows come from the partitioned search. The root listing may still run
    # — a global search returns availableFilters, and those are read back
    # through it — but never as the source of the page.
    for call in provider.get_knowledge_hub_root_nodes_v2.call_args_list:
        assert call.kwargs["limit"] == _MAX_FILTER_SOURCES
    provider.get_knowledge_hub_children_v2.assert_not_called()


@pytest.mark.asyncio
async def test_sorting_by_size_reaches_the_root_listing(service, provider) -> None:
    """The three sort maps that disagreed are one: a root listing sorted by
    size used to fall back to name without telling anyone."""
    provider.get_knowledge_hub_root_nodes_v2.return_value = envelope([])
    await service.get_nodes(user_id="u1", org_id="o1", sort_by="size", sort_order="asc")
    kwargs = provider.get_knowledge_hub_root_nodes_v2.call_args.kwargs
    assert (kwargs["sort_field"], kwargs["sort_dir"]) == ("sizeInBytes", "ASC")


@pytest.mark.asyncio
async def test_the_access_context_is_resolved_once_per_request(service, provider) -> None:
    """The listing, the filter options and a global search gate on one answer.

    Asking separately would let them disagree — filters offering a source the
    listing refuses to open, which is the shape of PG-34.
    """
    provider.get_knowledge_hub_root_nodes_v2.return_value = envelope([])
    await service.get_nodes(user_id="u1", org_id="o1", include=["availableFilters"])
    assert provider.get_knowledge_hub_access_context_v2.await_count == 1


@pytest.mark.asyncio
async def test_available_filters_come_from_the_gate(service, provider) -> None:
    """PG-34: an App the user cannot open cannot appear among the filters."""
    provider.get_knowledge_hub_root_nodes_v2.return_value = envelope(
        [row("app1", "App One", node_type="app")]
    )
    result = await service.get_nodes(
        user_id="u1", org_id="o1", include=["availableFilters"]
    )
    assert result.filters.available is not None
    # Both listing and filters ask for exactly the gated set.
    for call in provider.get_knowledge_hub_root_nodes_v2.call_args_list:
        assert call.kwargs["user_app_ids"] == ["app1", "kb-1"]
    provider.get_knowledge_hub_filter_options.assert_not_called()


# --------------------------------------------------------------- scope

@pytest.mark.asyncio
async def test_current_and_parent_nodes_come_from_the_listing_query(
    service, provider
) -> None:
    """NV-29: browsing used to cost three extra round trips for these."""
    provider.get_knowledge_hub_connector_page_v3.return_value = page_of(
        [], scope=admitted_scope()
    )
    result = await service.get_nodes(
        user_id="u1", org_id="o1", parent_id="p1", parent_type="recordGroup"
    )

    assert result.currentNode.id == "p1"
    assert result.parentNode.id == "app1"
    provider.get_knowledge_hub_node_info.assert_not_called()
    provider.get_knowledge_hub_parent_node.assert_not_called()


@pytest.mark.asyncio
async def test_breadcrumbs_come_from_the_same_query(service, provider) -> None:
    provider.get_knowledge_hub_connector_page_v3.return_value = page_of(
        [], scope=admitted_scope()
    )
    result = await service.get_nodes(
        user_id="u1", org_id="o1", parent_id="p1", parent_type="recordGroup",
        include=["breadcrumbs"],
    )
    assert [crumb.id for crumb in result.breadcrumbs] == ["app1", "p1"]
    provider.get_knowledge_hub_breadcrumbs.assert_not_called()


@pytest.mark.asyncio
async def test_an_inadmissible_node_is_a_404_that_names_nothing(service, provider) -> None:
    """SEC-02: the body must not confirm the node exists, or what it is."""
    provider.get_knowledge_hub_connector_page_v3.return_value = page_of(
        [], scope={"admitted": False, "nodeId": "secret-1"}
    )
    result = await service.get_nodes(
        user_id="u1", org_id="o1", parent_id="secret-1", parent_type="recordGroup"
    )

    assert result.success is False
    assert result.error == "Node not found"
    assert "secret-1" not in result.error
    assert result.currentNode is None and result.items == []


@pytest.mark.asyncio
async def test_the_wrong_type_in_the_url_is_still_a_400(service, provider) -> None:
    provider.get_knowledge_hub_connector_page_v3.return_value = page_of(
        [], scope=admitted_scope(node_type="app")
    )
    result = await service.get_nodes(
        user_id="u1", org_id="o1", parent_id="p1", parent_type="recordGroup"
    )
    assert result.success is False
    assert "type mismatch" in result.error.lower()


@pytest.mark.asyncio
async def test_a_folder_browse_is_not_a_type_mismatch(service, provider) -> None:
    """A folder is a record in the graph; comparing the raw strings would
    reject every folder browse."""
    provider.get_knowledge_hub_connector_page_v3.return_value = page_of(
        [], scope=admitted_scope(node_type="record")
    )
    result = await service.get_nodes(
        user_id="u1", org_id="o1", parent_id="p1", parent_type="folder"
    )
    assert result.success is True


# --------------------------------------------------------------- paging

@pytest.mark.asyncio
async def test_a_page_carries_cursors_and_indices(service, provider) -> None:
    provider.get_knowledge_hub_root_nodes_v2.return_value = envelope(
        [row("app1", "App One", node_type="app")], total=5, has_more=True
    )
    result = await service.get_nodes(user_id="u1", org_id="o1", limit=1)

    pagination = result.pagination
    assert pagination.nextCursor and pagination.prevCursor is None
    assert (pagination.startIndex, pagination.endIndex) == (1, 1)
    assert pagination.currentPageItems == 1
    assert pagination.hasNext is True and pagination.hasPrev is False
    assert pagination.totalItems == 5
    # Legacy fields stay until the frontend moves off them (D22).
    assert pagination.page == 1


@pytest.mark.asyncio
async def test_the_next_cursor_resumes_after_the_last_row(service, provider) -> None:
    provider.get_knowledge_hub_root_nodes_v2.return_value = envelope(
        [row("app1", "App One", node_type="app")], total=5, has_more=True
    )
    first = await service.get_nodes(user_id="u1", org_id="o1", limit=1)

    await service.get_nodes(
        user_id="u1", org_id="o1", limit=1, cursor=first.pagination.nextCursor
    )
    kwargs = provider.get_knowledge_hub_root_nodes_v2.call_args.kwargs
    assert kwargs["after"] == {"nullRank": 0, "sortKey": "app one", "id": "app1"}
    assert kwargs["direction"] == "next"


@pytest.mark.asyncio
async def test_the_cursor_decides_sort_and_filters(service, provider) -> None:
    """PG-27: the cursor wins over conflicting parameters, `include` does not.

    Scoped, because a `q` with no parent is a *global* search and would fan out
    across partitions instead of reaching a single listing query.
    """
    provider.get_knowledge_hub_connector_page_v3.return_value = page_of(
        [row("r1", "Record")], total=5, has_more=True, scope=admitted_scope()
    )
    first = await service.get_nodes(
        user_id="u1", org_id="o1", parent_id="p1", parent_type="recordGroup",
        limit=1, sort_by="createdAt", sort_order="desc", q="alpha",
    )

    await service.get_nodes(
        user_id="u1", org_id="o1", parent_id="p1", parent_type="recordGroup",
        limit=1, cursor=first.pagination.nextCursor,
        sort_by="name", sort_order="asc", q="beta",
    )
    kwargs = provider.get_knowledge_hub_connector_page_v3.call_args.kwargs
    assert (kwargs["sort_field"], kwargs["sort_dir"]) == ("createdAt", "DESC")
    assert kwargs["filters"]["search_query"] == "alpha"


@pytest.mark.asyncio
async def test_a_cursor_from_another_user_is_refused(service, provider, logger,
                                                     config_service) -> None:
    """PG-28: it is a 400, never a quietly-served first page."""
    provider.get_knowledge_hub_root_nodes_v2.return_value = envelope(
        [row("app1", "App One", node_type="app")], total=5, has_more=True
    )
    first = await service.get_nodes(user_id="u1", org_id="o1", limit=1)

    other = KnowledgeHubService(
        logger=logger, graph_provider=provider, config_service=config_service
    )
    result = await other.get_nodes(
        user_id="u2", org_id="o1", limit=1, cursor=first.pagination.nextCursor
    )
    assert result.success is False
    assert "invalid cursor" in result.error.lower()


@pytest.mark.asyncio
async def test_deep_page_numbers_are_refused(service, provider) -> None:
    """A page number re-traverses, so an unbounded one is free server load."""
    provider.get_knowledge_hub_root_nodes_v2.return_value = envelope([])
    result = await service.get_nodes(user_id="u1", org_id="o1", page=500, limit=200)
    assert result.success is False
    assert "invalid page" in result.error.lower()


@pytest.mark.asyncio
async def test_a_page_number_walks_forward(service, provider) -> None:
    """`page` survives transitionally by paging for the caller (D22)."""
    provider.get_knowledge_hub_root_nodes_v2.side_effect = [
        envelope([row("app1", "App One", node_type="app")], total=3, has_more=True),
        envelope([row("app2", "App Two", node_type="app")], total=3, has_more=True),
    ]
    result = await service.get_nodes(user_id="u1", org_id="o1", page=2, limit=1)

    assert [item.id for item in result.items] == ["app2"]
    assert provider.get_knowledge_hub_root_nodes_v2.await_count == 2


# --------------------------------------------------------------- counts

@pytest.mark.asyncio
async def test_counts_describe_the_whole_result(service, provider) -> None:
    """PG-33. The old breakdown counted the current page while the total beside
    it counted everything, so the two disagreed on every page but the last."""
    provider.get_knowledge_hub_root_nodes_v2.return_value = envelope(
        [row("app1", "App One", node_type="app")],
        total=3, has_more=True,
        ids=[{"id": "app1", "nodeType": "app"},
             {"id": "r1", "nodeType": "record"},
             {"id": "f1", "nodeType": "folder"}],
    )
    result = await service.get_nodes(
        user_id="u1", org_id="o1", limit=1, include=["counts"]
    )

    assert provider.get_knowledge_hub_root_nodes_v2.call_args.kwargs["include_ids"] is True
    assert result.counts.total == 3
    assert {item.label: item.count for item in result.counts.items} == {
        "apps": 1, "records": 1, "folders": 1,
    }


@pytest.mark.asyncio
async def test_ids_are_not_fetched_when_counts_were_not_asked_for(
    service, provider
) -> None:
    provider.get_knowledge_hub_root_nodes_v2.return_value = envelope([])
    await service.get_nodes(user_id="u1", org_id="o1")
    assert provider.get_knowledge_hub_root_nodes_v2.call_args.kwargs["include_ids"] is False


# --------------------------------------------------------------- rows

@pytest.mark.asyncio
async def test_rows_carry_the_parent_triple(service, provider) -> None:
    """D69: a search hit renders "in <folder>" without a second lookup."""
    provider.get_knowledge_hub_connector_page_v3.return_value = page_of(
        [row("r1", "Record", parent="f1", parent_type="folder")],
        scope=admitted_scope(),
    )
    result = await service.get_nodes(
        user_id="u1", org_id="o1", parent_id="p1", parent_type="recordGroup"
    )
    item = result.items[0]
    assert item.parentId == "f1"
    assert (item.parent.id, item.parent.nodeType, item.parent.name) == (
        "f1", "folder", "Parent",
    )


@pytest.mark.asyncio
async def test_an_unknown_user_is_reported_without_querying(service, provider) -> None:
    provider.get_user_by_user_id.return_value = None
    result = await service.get_nodes(user_id="nobody", org_id="o1")
    assert result.success is False and result.error == "User not found"
    provider.get_knowledge_hub_root_nodes_v2.assert_not_called()


@pytest.mark.asyncio
async def test_without_a_signer_pages_work_but_hand_out_no_cursor(
    logger, provider
) -> None:
    """An unsigned cursor is an editable one, so none is handed out at all.

    The agent tools build this service without a config service and page by
    number; the HTTP router always passes one, so the API keeps its cursors.
    """
    unconfigured = KnowledgeHubService(logger=logger, graph_provider=provider)
    provider.get_knowledge_hub_root_nodes_v2.return_value = envelope(
        [row("app1", "App One", node_type="app")], total=5, has_more=True
    )
    result = await unconfigured.get_nodes(user_id="u1", org_id="o1", limit=1)

    assert result.success is True
    assert [item.id for item in result.items] == ["app1"]
    assert result.pagination.nextCursor is None
    assert result.pagination.prevCursor is None


@pytest.mark.asyncio
async def test_a_cursor_is_refused_when_nothing_can_verify_it(
    logger, provider
) -> None:
    unconfigured = KnowledgeHubService(logger=logger, graph_provider=provider)
    provider.get_knowledge_hub_root_nodes_v2.return_value = envelope([])
    result = await unconfigured.get_nodes(user_id="u1", org_id="o1", cursor="anything")
    assert result.success is False
    assert "invalid cursor" in result.error.lower()


@pytest.mark.asyncio
async def test_page_numbers_still_walk_without_a_signer(logger, provider) -> None:
    """Walking uses cursors internally; they just never leave the process."""
    unconfigured = KnowledgeHubService(logger=logger, graph_provider=provider)
    provider.get_knowledge_hub_root_nodes_v2.side_effect = [
        envelope([row("app1", "App One", node_type="app")], total=3, has_more=True),
        envelope([row("app2", "App Two", node_type="app")], total=3, has_more=True),
    ]
    result = await unconfigured.get_nodes(user_id="u1", org_id="o1", page=2, limit=1)
    assert [item.id for item in result.items] == ["app2"]
