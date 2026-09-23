"""Knowledge Hub Unified Browse Service"""

import logging
import re
import traceback
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from app.config.configuration_service import ConfigurationService
from app.config.constants.arangodb import ProgressStatus
from app.config.constants.service import config_node_constants
from app.connectors.sources.localKB.api.knowledge_hub_models import (
    AppliedFilters,
    AvailableFilters,
    BreadcrumbItem,
    CountItem,
    CountsInfo,
    CurrentNode,
    FilterOption,
    FiltersInfo,
    ItemPermission,
    KnowledgeHubNodesResponse,
    NodeItem,
    NodeType,
    OriginType,
    PaginationInfo,
    ParentRef,
    PermissionsInfo,
    SortField,
    SortOrder,
)
from app.connectors.sources.localKB.handlers.kh_search import SearchPage, search_page
from app.models.entities import RecordType
from app.services.graph_db.interface.graph_db_provider import IGraphDBProvider
from app.utils.kh_cursor import (
    Boundary,
    CursorError,
    KnowledgeHubCursor,
    decode,
    derive_cursor_secret,
    encode,
)
from app.utils.user_messages import action_failed, not_found


class BrowseRequestError(Exception):
    """A browse request this service can explain to the person who made it.

    Only messages written here for a reader travel in one of these. Everything
    else that goes wrong — including a ``ValueError`` the graph client raises for
    its own reasons, such as a transaction it can no longer find — is internal,
    and the caller answers it with a generic message instead.
    """

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# A page number re-traverses and skips, so deep paging is both slow and a
# trivial way to make the server work hard. Cursors have no such cost; the
# bound exists because `page` survives only until the frontend moves (D22).
_MAX_PAGE_WALK_ITEMS = 10_000

# Filter options list the user's sources, and a tenant with more Apps and
# collections than this has a filter dropdown nobody can use anyway.
_MAX_FILTER_SOURCES = 500


FOLDER_MIME_TYPES = [
    'application/vnd.folder',
    'application/vnd.google-apps.folder',
    'text/directory'
]

class KnowledgeHubService:
    """Service for unified Knowledge Hub browse API"""

    # One map, keyed by the API's names and valued with the provider's. It
    # replaced three that disagreed: the root listing's lacked `size` and
    # `type`, so sorting a root listing by either silently fell back to name.
    _SORT_FIELDS = {
        "name": "name",
        "createdAt": "createdAt",
        "updatedAt": "updatedAt",
        "size": "sizeInBytes",
        "type": "nodeType",
    }

    def __init__(
        self,
        logger: logging.Logger,
        graph_provider: IGraphDBProvider,
        config_service: ConfigurationService | None = None,
    ) -> None:
        self.logger = logger
        self.graph_provider = graph_provider
        self.config_service = config_service
        self._cursor_secret: bytes | None = None

    async def _get_cursor_secret(self) -> bytes | None:
        """The cursor-signing key, derived once, or None when nothing can sign.

        An unsigned cursor is an editable one, and what a cursor carries decides
        what the next page looks at (PG-31) — so a caller with no config service
        (the agent tools, which page by number) gets pages **without** cursors
        rather than unsigned ones, and any cursor it is handed is refused.
        """
        if self._cursor_secret is None:
            if self.config_service is None:
                return None
            secret_keys = await self.config_service.get_config(
                config_node_constants.SECRET_KEYS.value
            )
            self._cursor_secret = derive_cursor_secret(
                (secret_keys or {}).get("scopedJwtSecret")
            )
        return self._cursor_secret

    def _sort_field(self, sort_by: str) -> str:
        return self._SORT_FIELDS.get(sort_by, "name")

    def _decode_cursor(
        self, token: str | None, secret: bytes | None, user_id: str, org_id: str
    ) -> KnowledgeHubCursor | None:
        """A cursor that does not verify is a 400, never a silent first page.

        A caller's cursor is refused up front when nothing can verify it, so a
        token reaching here unsigned is one this service made to walk to a page
        number and never handed out.
        """
        if not token:
            return None
        try:
            return decode(token, secret, expected_user_id=user_id, expected_org_id=org_id)
        except CursorError as exc:
            raise BrowseRequestError(f"Invalid cursor: {exc}", 400) from exc

    async def _walk_pages(
        self,
        fetch_page: Callable[[str | None], Awaitable[SearchPage]],
        cursor_token: str | None,
        page: int,
        limit: int,
    ) -> SearchPage:
        """One page, by cursor if given, otherwise by walking forward to `page`."""
        if cursor_token:
            return await fetch_page(cursor_token)
        if page > 1 and page * limit > _MAX_PAGE_WALK_ITEMS:
            raise BrowseRequestError(
                f"Invalid page: page * limit must not exceed {_MAX_PAGE_WALK_ITEMS}. "
                f"Use the cursor from a previous response.",
                400,
            )
        result = await fetch_page(None)
        for _ in range(page - 1):
            if not result.next_cursor:
                break
            result = await fetch_page(result.next_cursor)
        return result

    def _page_from_partition(
        self,
        part: dict[str, Any],
        cursor: KnowledgeHubCursor | None,
        secret: bytes,
        *,
        filters: dict[str, Any],
        sort_field: str,
        sort_dir: str,
        parent_id: str | None,
        parent_type: str | None,
        user_id: str,
        org_id: str,
        want_counts: bool,
    ) -> SearchPage:
        """A single-partition listing (root or scoped) as a page, with its cursors.

        Browse and a scoped flatten are one query, so there is nothing to merge
        — but the page shape, the boundary and the carried total are the same
        as a global search's, which is what lets one response builder serve both.
        """
        rows = part["rows"]
        direction = cursor.direction if cursor else "next"

        if cursor is not None and cursor.total is not None:
            total, counts = cursor.total, cursor.counts_by_type
        else:
            total = part["total"] if part["total"] is not None else len(rows)
            if want_counts and part.get("counts") is not None:
                counts = part["counts"]
            elif want_counts:
                counts = dict(Counter(entry["nodeType"] for entry in part["ids"]))
            else:
                counts = None

        seen_before = 0
        if cursor is not None:
            seen_before = (
                cursor.items_seen if direction == "next"
                else max(0, cursor.items_seen - len(rows))
            )
        # Going back, `hasMore` means more rows lie *before* this page; the page
        # ahead is the one the caller just came from and always exists.
        has_next = part["hasMore"] if direction == "next" else True
        has_prev = (cursor is not None) if direction == "next" else part["hasMore"]

        def issue(row: dict[str, Any], next_direction: str, items_seen: int) -> str:
            return encode(
                KnowledgeHubCursor(
                    boundary=Boundary.of(row),
                    direction=next_direction,
                    items_seen=items_seen,
                    total=total,
                    counts_by_type=counts,
                    filters=filters,
                    sort_by=sort_field,
                    sort_order=sort_dir,
                    parent_id=parent_id,
                    parent_type=parent_type,
                    via_parent_id=cursor.via_parent_id if cursor else None,
                    user_id=user_id,
                    org_id=org_id,
                ),
                secret,
            )

        return SearchPage(
            rows=rows,
            total=total,
            counts_by_type=counts,
            start_index=seen_before + 1 if rows else 0,
            end_index=seen_before + len(rows),
            next_cursor=issue(rows[-1], "next", seen_before + len(rows))
            if rows and has_next else None,
            prev_cursor=issue(rows[0], "prev", seen_before)
            if rows and has_prev else None,
        )

    async def _v2_nodes(
        self,
        *,
        user_id: str,
        user_key: str,
        org_id: str,
        parent_id: str | None,
        parent_type: str | None,
        limit: int,
        page: int,
        cursor: str | None,
        sort_by: str,
        sort_order: str,
        filters: dict[str, Any],
        flatten: bool,
        want_counts: bool,
        access: dict[str, list[str]],
    ) -> tuple[SearchPage, dict[str, Any] | None]:
        """One page of the v2 read path, plus the `scope` a scoped request carries.

        Three modes, one shape: a global search is partitioned and merged
        (§3.9), while a root listing and a scoped browse or flatten are each a
        single query that also returns `currentNode`, `parentNode` and the
        breadcrumb trail — which is why browsing no longer costs three extra
        round trips (NV-29, PERF-01).
        """
        secret = await self._get_cursor_secret()
        # Nothing can verify it, so it cannot be trusted to say where to resume.
        if cursor and secret is None:
            raise BrowseRequestError("Invalid cursor: cursor paging is not configured", 400)
        sort_field = self._sort_field(sort_by)
        sort_dir = "ASC" if sort_order.lower() == "asc" else "DESC"

        if parent_id is None and flatten:
            async def fetch_global(token: str | None) -> SearchPage:
                return await search_page(
                    self.graph_provider,
                    user_key=user_key, user_id=user_id, org_id=org_id,
                    limit=limit, sort_field=sort_field, sort_dir=sort_dir,
                    filters=filters, cursor_token=token, secret=secret,
                    access=access,
                )
            try:
                global_page = await self._walk_pages(fetch_global, cursor, page, limit)
            except CursorError as exc:
                raise BrowseRequestError(f"Invalid cursor: {exc}", 400) from exc
            return self._only_signed_cursors(global_page, secret), None

        scope: dict[str, Any] | None = None

        async def fetch_single(token: str | None) -> SearchPage:
            nonlocal scope
            page_cursor = self._decode_cursor(token, secret, user_id, org_id)
            # PG-27: the cursor decides sort and filters, so a page cannot
            # resume a keyset from an order that no longer applies.
            active = dict(page_cursor.filters or {}) if page_cursor else dict(filters)
            field = (page_cursor.sort_by if page_cursor else None) or sort_field
            order = (page_cursor.sort_order if page_cursor else None) or sort_dir
            reuse_total = page_cursor is not None and page_cursor.total is not None

            common = {
                "user_key": user_key, "org_id": org_id, "limit": limit,
                "sort_field": field, "sort_dir": order,
                "after": page_cursor.boundary.as_after() if page_cursor else None,
                "direction": page_cursor.direction if page_cursor else "next",
                "include_ids": want_counts and not reuse_total,
            }
            if parent_id is None:
                envelope = await self.graph_provider.get_knowledge_hub_root_nodes_v2(
                    user_app_ids=access["gated_app_ids"],
                    # Apps are in no record group, and this filter only narrows
                    # collection-origin groups.
                    **{k: v for k, v in active.items() if k != "record_group_ids"},
                    **common,
                )
            else:
                granted = await self.graph_provider.get_knowledge_hub_access_v3(
                    user_key=user_key, org_id=org_id,
                )
                start_type = {
                    "app": "app", "kb": "app", "recordGroup": "recordGroup",
                    "folder": "record", "record": "record",
                }.get(parent_type or "", "record")
                page = await self.graph_provider.get_knowledge_hub_connector_page_v3(
                    app_id=parent_id if start_type == "app" else "",
                    org_id=org_id,
                    grantee_ids=granted["grantee_ids"],
                    gated_app_ids=granted["gated_app_ids"],
                    granted_ids=None,
                    limit=limit,
                    flatten=flatten,
                    sort_field=field,
                    sort_dir=order,
                    after=page_cursor.boundary.as_after() if page_cursor else None,
                    direction=page_cursor.direction if page_cursor else "next",
                    filters=active,
                    include_total=not reuse_total,
                    start_id=parent_id,
                    start_type=start_type,
                    grants_by_connector=granted["by_connector"],
                    include_scope=True,
                    via_parent_id=page_cursor.via_parent_id if page_cursor else None,
                )
                envelope = {
                    "partitions": [{
                        "rows": page["rows"],
                        "hasMore": page["hasMore"],
                        "total": page["total"],
                        "ids": [],
                        "counts": page.get("counts"),
                    }],
                    "scope": page.get("scope"),
                }
            scope = envelope.get("scope")
            return self._page_from_partition(
                envelope["partitions"][0], page_cursor, secret,
                filters=active, sort_field=field, sort_dir=order,
                parent_id=parent_id, parent_type=parent_type,
                user_id=user_id, org_id=org_id, want_counts=want_counts,
            )

        scoped_page = await self._walk_pages(fetch_single, cursor, page, limit)
        return self._only_signed_cursors(scoped_page, secret), scope

    def _only_signed_cursors(self, page: SearchPage, secret: bytes | None) -> SearchPage:
        """Hand out cursors only when they are signed.

        Walking to a `page` number builds cursors internally to step forward, so
        they exist either way — but an unsigned one must never leave the
        process, where it becomes an editable instruction about what to read
        next (PG-31).
        """
        if secret is not None:
            return page
        return replace(page, next_cursor=None, prev_cursor=None)

    def _to_current_node(self, crumb: dict[str, Any] | None) -> CurrentNode | None:
        if not crumb or not crumb.get('id'):
            return None
        return CurrentNode(
            id=crumb['id'],
            name=crumb.get('name') or '',
            nodeType=crumb.get('nodeType') or '',
            subType=crumb.get('subType'),
        )

    async def _resolve_user(self, user_id: str, org_id: str) -> Any | None:
        """Resolve graph user node from external userId. EE overrides for org-scoped lookup."""
        return await self.graph_provider.get_user_by_user_id(user_id=user_id)

    async def _get_user_app_ids(
        self, user_key: str, org_id: str
    ) -> list[str]:
        """Return app IDs accessible to this user. EE overrides for org-scoped lookup."""
        owned_app_ids = await self.graph_provider.get_user_app_ids(user_key)
        shared_app_ids = await self.graph_provider.get_user_permission_app_ids(user_key, org_id)
        return list(dict.fromkeys([*owned_app_ids, *shared_app_ids]))

    def _has_flattening_filters(self, q: str | None, node_types: list[str] | None,
                                 record_types: list[str] | None, origins: list[str] | None,
                                 connector_ids: list[str] | None,
                                 indexing_status: list[str] | None,
                                 created_at: dict | None, updated_at: dict | None,
                                 size: dict | None) -> bool:
        """Check if any filters that should trigger flattened/recursive search are provided.

        These filters should return flattened results (all nested children):
        - q, nodeTypes, recordTypes, origins, connectorIds,
          createdAt, updatedAt, size, indexingStatus
        Note: sortBy and sortOrder are NOT included as they don't trigger flattening.

        This is only the FALLBACK computation used when the caller doesn't pass
        an explicit `flattened` flag — see get_nodes() for the precedence rule.
        """
        return any([q, node_types, record_types, origins, connector_ids,
                    indexing_status, created_at, updated_at, size])

    async def get_nodes(
        self,
        user_id: str,
        org_id: str,
        parent_id: str | None = None,
        parent_type: str | None = None,
        only_containers: bool = False,
        page: int = 1,
        limit: int = 50,
        sort_by: str = "updatedAt",
        sort_order: str = "desc",
        q: str | None = None,
        node_types: list[str] | None = None,
        record_types: list[str] | None = None,
        origins: list[str] | None = None,
        connector_ids: list[str] | None = None,
        indexing_status: list[str] | None = None,
        created_at: dict[str, int | None] | None = None,
        updated_at: dict[str, int | None] | None = None,
        size: dict[str, int | None] | None = None,
        flattened: bool | None = None,
        include: list[str] | None = None,
        record_group_ids: list[str] | None = None,
        depth: int | None = None,
        include_typed_records: bool = False,
        cursor: str | None = None,
    ) -> KnowledgeHubNodesResponse:
        """
        Get nodes for the Knowledge Hub unified browse API

        `flattened` precedence: if the caller passes it explicitly (True or
        False), it always decides search-vs-browse mode. Only when it's
        omitted (None) do we fall back to computing it from which filters
        are present (see _has_flattening_filters).
        """
        try:
            page = max(1, page)
            limit = min(max(1, limit), 200)  # Max 200

            # Get user key
            user = await self._resolve_user(user_id, org_id)
            if not user:
                return KnowledgeHubNodesResponse(
                    success=False,
                    error="User not found",
                    errorCode=404,
                    id=parent_id,
                    items=[],
                    pagination=PaginationInfo(
                        page=page, limit=limit, totalItems=0, totalPages=0,
                        hasNext=False, hasPrev=False
                    ),
                    filters=FiltersInfo(applied=AppliedFilters()),
                )
            user_key = user.get('_key')

            # `flattened`, when explicitly passed by the caller, always wins.
            # Otherwise fall back to computing it from which filters are present
            # (any of q/nodeTypes/recordTypes/origins/connectorIds/indexingStatus/
            # createdAt/updatedAt/size triggers the flattened/recursive search).
            if flattened is not None:
                use_search_mode = flattened
            else:
                use_search_mode = self._has_flattening_filters(
                    q, node_types, record_types, origins, connector_ids,
                    indexing_status, created_at, updated_at, size
                )

            # Browse applies filters too under v2: one query serves browse,
            # flatten and search, so a filtered browse no longer has to become
            # a search to be filtered.
            filters = {
                "search_query": q,
                "node_types": node_types,
                "record_types": record_types,
                "indexing_status": indexing_status,
                "created_at": created_at,
                "updated_at": updated_at,
                "size": size,
                "origins": origins,
                "connector_ids": connector_ids,
                "record_group_ids": record_group_ids,
                "only_containers": only_containers,
            }
            # Who the user is, resolved once: the listing queries, the global
            # search and the filter options all gate on this same answer
            # (D42, D43), and asking three times would let them disagree.
            access = await self.graph_provider.get_knowledge_hub_access_context_v2(
                user_key=user_key, org_id=org_id
            )
            page_result, scope = await self._v2_nodes(
                user_id=user_id,
                user_key=user_key,
                org_id=org_id,
                access=access,
                parent_id=parent_id,
                parent_type=parent_type,
                limit=limit,
                page=page,
                cursor=cursor,
                sort_by=sort_by,
                sort_order=sort_order,
                filters=filters,
                flatten=use_search_mode,
                want_counts=bool(include and 'counts' in include),
            )

            # The start node's own admission decides this, and the 404 body is
            # constant: naming the node, or its type, would confirm it exists
            # to someone who may not see it (SEC-02).
            if scope is not None and not scope.get('admitted'):
                raise BrowseRequestError("Node not found", 404)

            # Browsing with the wrong type in the URL stays a 400, but the type
            # now comes from the listing query's own scope rather than a
            # separate node_info lookup. A folder *is* a record in the graph,
            # so those two are one type here — comparing the raw values would
            # reject every folder browse.
            actual_type = ((scope or {}).get('currentNode') or {}).get('nodeType')
            if parent_type and actual_type:
                same = {'folder': 'record'}
                if same.get(actual_type, actual_type) != same.get(parent_type, parent_type):
                    raise BrowseRequestError(
                        f"Node type mismatch: node '{parent_id}' is not '{parent_type}', "
                        f"it is '{actual_type}'. Use /nodes/{actual_type}/{parent_id} instead.",
                        400,
                    )

            items = [self._doc_to_node_item(row) for row in page_result.rows]
            total_count = page_result.total or 0

            available_filters = None
            if (parent_id is None and use_search_mode) or (include and 'availableFilters' in include):
                available_filters = await self._get_available_filters(
                    user_key, org_id, access
                )

            # Permissions come from the query itself (userRole), and so do
            # currentNode, parentNode and the breadcrumbs — the listing query
            # returns them, replacing three round trips (D5, D58, NV-29).
            total_pages = (total_count + limit - 1) // limit if total_count > 0 else 0
            current_node = self._to_current_node((scope or {}).get('currentNode'))
            parent_node = self._to_current_node((scope or {}).get('parentNode'))

            # Build applied filters
            applied_filters = AppliedFilters(
                q=q,
                nodeTypes=node_types,
                recordTypes=record_types,
                origins=origins,
                connectorIds=connector_ids,
                indexingStatus=indexing_status,
                createdAt=created_at,
                updatedAt=updated_at,
                size=size,
                sortBy=sort_by,
                sortOrder=sort_order,
            )

            # Build filters info (without available filters initially)
            filters_info = FiltersInfo(applied=applied_filters)

            # Build response
            response = KnowledgeHubNodesResponse(
                success=True,
                id=parent_id,
                currentNode=current_node,
                parentNode=parent_node,
                items=items,
                pagination=PaginationInfo(
                    limit=limit,
                    totalItems=total_count,
                    hasNext=page_result.next_cursor is not None,
                    hasPrev=page_result.prev_cursor is not None,
                    startIndex=page_result.start_index,
                    endIndex=page_result.end_index,
                    currentPageItems=len(items),
                    nextCursor=page_result.next_cursor,
                    prevCursor=page_result.prev_cursor,
                    # Legacy, and meaningless once the caller pages by cursor.
                    page=page if cursor is None else None,
                    totalPages=total_pages if cursor is None else None,
                ),
                filters=filters_info,
            )

            # Fetch typed records if requested (for LLM context enrichment)
            if include_typed_records and items:
                record_ids = [
                    item.id for item in items
                    if item.nodeType in (NodeType.RECORD, NodeType.FOLDER)
                ]
                if record_ids:
                    try:
                        response.typed_records = await self.graph_provider.get_typed_records_batch(
                            record_ids
                        )
                    except Exception as e:
                        self.logger.warning("Failed to fetch typed records: %s", e)

            # Add optional expansions
            if include:
                if 'availableFilters' in include:
                    # Add available filters only when requested
                    response.filters.available = available_filters

                if 'breadcrumbs' in include and scope:
                    # The placement trail: an ancestor the user cannot open is
                    # replaced by where the node actually appears, never named
                    # (NV-28, SEC-01).
                    response.breadcrumbs = [
                        BreadcrumbItem(
                            id=crumb['id'],
                            name=crumb.get('name') or '',
                            nodeType=crumb.get('nodeType') or '',
                            subType=crumb.get('subType'),
                        )
                        for crumb in (scope.get('breadcrumbs') or [])
                        if crumb.get('id')
                    ]

                if 'counts' in include:
                    # Whole-result counts, taken over the union of every
                    # partition's matching ids on the first page and then
                    # carried in the cursor (PG-32, PG-33). The old breakdown
                    # counted only the current page, which disagreed with the
                    # total sitting beside it.
                    type_counts = page_result.counts_by_type or {}

                    # Map nodeType to display label
                    label_map = {
                        'app': 'apps',
                        'folder': 'folders',
                        'recordGroup': 'groups',
                        'record': 'records',
                    }

                    count_items = [
                        CountItem(
                            label=label_map.get(node_type, node_type),
                            count=count
                        )
                        for node_type, count in sorted(type_counts.items())
                    ]

                    response.counts = CountsInfo(
                        items=count_items,
                        total=total_count,  # Use actual total count, not paginated length
                    )

                if 'permissions' in include:
                    response.permissions = await self._get_permissions(
                        user_key, org_id, parent_id, parent_type
                    )

            return response

        except BrowseRequestError as request_error:
            self.logger.warning("⚠️ Browse request refused: %s", request_error.message)
            return KnowledgeHubNodesResponse(
                success=False,
                error=request_error.message,  # user-written message
                errorCode=request_error.status_code,
                id=parent_id,
                items=[],
                pagination=PaginationInfo(
                    page=page, limit=limit, totalItems=0, totalPages=0,
                    hasNext=False, hasPrev=False
                ),
                filters=FiltersInfo(applied=AppliedFilters()),
            )
        except Exception as e:
            self.logger.error("❌ Failed to get nodes: %s", e, exc_info=True)
            return KnowledgeHubNodesResponse(
                success=False,
                error=action_failed("open this collection"),
                errorCode=500,
                id=parent_id,
                items=[],
                pagination=PaginationInfo(
                    page=page, limit=limit, totalItems=0, totalPages=0,
                    hasNext=False, hasPrev=False
                ),
                filters=FiltersInfo(applied=AppliedFilters()),
            )

    async def _get_available_filters(
        self,
        user_key: str,
        org_id: str,
        access: dict[str, list[str]] | None = None,
    ) -> AvailableFilters:
        """The static filter enums, plus the sources this user can actually open.

        PG-34. The source list used to come from `get_user_apps`, which answers
        a different question in three ways: it is **not org-scoped** (the query
        has no `orgId` predicate at all, so another org's App could be listed),
        it misses an App reachable only through a grantee's permission rather
        than a user-app relation (D43), and it dropped collections outright
        (`type != 'KB'`), so a collection the user can open never appeared even
        though the case calls for "reachable Apps *and KBs*".

        The v2 gate answers all three at once: the same `gated_app_ids` the
        listing queries admit, read back through the v2 root listing so the
        label and type come from the row the user would actually see, already
        ordered case-insensitively by name.

        `get_knowledge_hub_filter_options` stays for its other callers (the
        agent catalog and the chat bridge), which ask a different question.
        """
        try:
            if access is None:
                access = await self.graph_provider.get_knowledge_hub_access_context_v2(
                    user_key=user_key, org_id=org_id
                )

            app_options: list[FilterOption] = []
            if access.get("gated_app_ids"):
                listing = await self.graph_provider.get_knowledge_hub_root_nodes_v2(
                    user_key=user_key,
                    org_id=org_id,
                    user_app_ids=access["gated_app_ids"],
                    limit=_MAX_FILTER_SOURCES,
                )
                app_options = [
                    FilterOption(
                        id=row["id"],
                        label=row.get("name") or row["id"],
                        connectorType=row.get("connector") or row.get("origin"),
                    )
                    for row in listing["partitions"][0]["rows"]
                ]

            # Node type labels mapping
            node_type_labels = {
                NodeType.FOLDER: "Folder",
                NodeType.RECORD: "File",
                NodeType.RECORD_GROUP: "Drive/Root",
                NodeType.APP: "Connector",
            }

            return AvailableFilters(
                nodeTypes=[
                    FilterOption(
                        id=nt.value,
                        label=node_type_labels.get(nt, nt.value)
                    )
                    for nt in NodeType
                ],
                recordTypes=[
                    FilterOption(
                        id=rt.value,
                        label=self._format_enum_label(rt.value)
                    )
                    for rt in RecordType
                ],
                origins=[
                    FilterOption(
                        id=ot.value,
                        label="Collection" if ot == OriginType.COLLECTION else "External Connector"
                    )
                    for ot in OriginType
                ],
                connectors=app_options,
                indexingStatus=[
                    FilterOption(
                        id=status.value,
                        label=self._format_enum_label(status.value, {"AUTO_INDEX_OFF": "Manual Indexing"})
                    )
                    for status in ProgressStatus
                ],
                sortBy=[
                    FilterOption(
                        id=sf.value,
                        label=self._format_enum_label(sf.value, {"createdAt": "Created Date", "updatedAt": "Modified Date"})
                    )
                    for sf in SortField
                ],
                sortOrder=[
                    FilterOption(
                        id=so.value,
                        label="Ascending" if so == SortOrder.ASC else "Descending"
                    )
                    for so in SortOrder
                ]
            )
        except Exception as e:
            self.logger.error(f"Failed to get available filters: {e}")
            return AvailableFilters()

    async def _get_permissions(
        self,
        user_key: str,
        org_id: str,
        parent_id: str | None,
        parent_type: str | None = None,
    ) -> PermissionsInfo | None:
        """Get user permissions for the current context. Returns None if user has no permission."""
        try:
            perm_data = await self.graph_provider.get_knowledge_hub_context_permissions(
                user_key=user_key,
                org_id=org_id,
                parent_id=parent_id,
                parent_type=parent_type,
            )

            # If role is None, user has no permission - return None
            role = perm_data.get('role')
            if role is None:
                return None

            return PermissionsInfo(
                role=role,
                canUpload=perm_data.get('canUpload', False),
                canCreateFolders=perm_data.get('canCreateFolders', False),
                canEdit=perm_data.get('canEdit', False),
                canDelete=perm_data.get('canDelete', False),
                canManagePermissions=perm_data.get('canManagePermissions', False),
            )

        except Exception as e:
            self.logger.error(f"❌ Failed to get permissions: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Return None on error (no permission granted)
            return None

    def _doc_to_node_item(self, doc: dict[str, Any]) -> NodeItem:
        """Convert a database document to a NodeItem"""
        # Extract ID - prefer 'id' field, fallback to '_key' or parse from '_id'
        doc_id = doc.get('id')
        if not isinstance(doc_id, str) or not doc_id.strip():
            if doc.get('_key'):
                doc_id = doc['_key']
            elif doc.get('_id'):
                _id_value = doc['_id']
                if isinstance(_id_value, str) and '/' in _id_value:
                    doc_id = _id_value.split('/', 1)[1]
                else:
                    doc_id = _id_value
            else:
                doc_id = ''

        node_type_str = doc.get('nodeType', 'record')
        try:
            node_type = NodeType(node_type_str)
        except ValueError:
            node_type = NodeType.RECORD

        # Get origin
        origin_str = doc.get('origin', 'COLLECTION')
        origin = OriginType.COLLECTION if origin_str == 'COLLECTION' else OriginType.CONNECTOR

        # Convert userRole to ItemPermission if present
        permission = None
        user_role = doc.get('userRole')
        if user_role:
            # Handle case where userRole might be a list (defensive safeguard)
            if isinstance(user_role, list):
                user_role = user_role[0] if user_role else None
            if user_role:
                permission = self._role_to_permission(user_role)

        # The parent triple travels with the row (D69), so a search hit renders
        # "in <folder>" without a second lookup. Only v2 rows carry the type;
        # without it a client cannot build the parent's own URL.
        parent = None
        if doc.get('parentId') and doc.get('parentType'):
            parent = ParentRef(
                id=doc['parentId'],
                nodeType=doc['parentType'],
                name=doc.get('parentName'),
            )

        # Build NodeItem
        return NodeItem(
            id=doc_id,
            name=doc.get('name', ''),
            nodeType=node_type,
            parentId=doc.get('parentId'),
            parent=parent,
            origin=origin,
            connector=doc.get('connector'),
            recordType=doc.get('recordType'),
            recordGroupType=doc.get('recordGroupType'),
            indexingStatus=doc.get('indexingStatus'),
            reason=doc.get('reason'),
            createdAt=doc.get('createdAt', 0),
            updatedAt=doc.get('updatedAt', 0),
            sizeInBytes=doc.get('sizeInBytes'),
            mimeType=doc.get('mimeType'),
            extension=doc.get('extension'),
            webUrl=doc.get('webUrl'),
            hasChildren=doc.get('hasChildren', False),
            previewRenderable=doc.get('previewRenderable'),
            permission=permission,
            sharingStatus=doc.get('sharingStatus'),
            isInternal=bool(doc.get('isInternal', False)),
            isPlaceholder=bool(doc.get('isPlaceholder', False)),
        )


    def _role_to_permission(self, role: str) -> ItemPermission:
        """
        Convert a user role string to ItemPermission object with computed flags.

        Permission hierarchy:
        - OWNER: Full control (edit + delete all)
        - EDITOR: Can edit content
        - WRITER: Can edit and delete folders/records
        - COMMENTER, READER: Read-only (no edit, no delete)
        """
        role_upper = role.upper() if role else ''

        # Determine edit and delete permissions based on role
        can_edit = role_upper in ['OWNER', 'WRITER']
        can_delete = role_upper in ['OWNER', 'WRITER']

        return ItemPermission(
            role=role,
            canEdit=can_edit,
            canDelete=can_delete,
        )

    def _format_enum_label(self, value: str, special_cases: dict[str, str] | None = None) -> str:
        """
        Convert enum value to human-readable label.

        Handles both UPPER_SNAKE_CASE and camelCase:
        - "FILE_NAME" → "File Name"
        - "createdAt" → "Created At"
        - "autoIndexOff" → "Auto Index Off"

        Args:
            value: The enum value to format
            special_cases: Optional dict of special case mappings that differ from generic formatting

        Returns:
            Human-readable label
        """
        if special_cases and value in special_cases:
            return special_cases[value]

        # Handle camelCase by inserting space before uppercase letters
        # Insert space before uppercase letters that follow lowercase letters
        spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', value)
        # Replace underscores with spaces
        spaced = spaced.replace("_", " ")
        # Title case each word
        return spaced.title()
