"""Scenarios built by the real write path, asserted in a real database.

Every other module here loads a hand-built fixture. This one drives
``DataSourceEntitiesProcessor`` — the same code a connector sync runs — so the
graph under test is one the write path can actually produce. A hand-built
fixture proves the query is self-consistent; this proves the two halves agree.

Covers the Phase 1 edges that did not exist before: RecordGroup -> App
inheritance and its removal when a group stops inheriting; record -> parent
*record* inheritance; the App -> group and group -> record hierarchy the
traversal descends, including its deliberate *absence* for a nested record,
which is what stops the walk reaching one while skipping the restrictions
above it; and the re-parenting of survivors on both delete paths.

Every case runs against both backends, because the two providers do not
interpret the same call identically. Arango writes an edge straight through the
document API without checking its endpoints; Neo4j MATCHes both endpoints first
and silently writes nothing when either is missing. Neither raises. The same
asymmetry applies to deletes, where Neo4j resolves node *labels* from the
from/to collection arguments that Arango uses only to build ``_from``/``_to``
handles — so a wrong collection name there is a silent no-op on Neo4j and a
working delete on Arango.
"""

import uuid

import aiohttp
import pytest
from neo4j import AsyncGraphDatabase

from app.config.constants.arangodb import (
    AccessRule,
    CollectionNames,
    Connectors,
    OriginTypes,
)
from app.config.constants.neo4j import (
    collection_to_label,
    edge_collection_to_relationship,
)
from app.models.entities import RecordGroup, RecordGroupType, RecordType, WebpageRecord
from app.models.permission import EntityType, Permission, PermissionType

from .processor_harness import build_processor

pytestmark = pytest.mark.integration

ORG = "org-write-path"
TS = 1700000000000


def _suffix() -> str:
    return uuid.uuid4().hex[:8]


class _ArangoBackend:
    name = "arango"

    def __init__(self, provider, settings: dict) -> None:
        self.provider = provider
        self._settings = settings

    async def _aql(self, query: str, bind: dict) -> list:
        auth = aiohttp.BasicAuth(self._settings["username"], self._settings["password"])
        url = f"{self._settings['url'].rstrip('/')}/_db/{self._settings['db']}/_api/cursor"
        async with aiohttp.ClientSession(auth=auth) as session:
            async with session.post(url, json={"query": query, "bindVars": bind}) as resp:
                body = await resp.json()
                assert resp.status in (200, 201), body
                return body["result"]

    async def edge_exists(
        self, edge_collection: str, from_collection: str, from_id: str,
        to_collection: str, to_id: str,
    ) -> bool:
        rows = await self._aql(
            f"RETURN LENGTH(FOR e IN {edge_collection} "
            "FILTER e._from == @from AND e._to == @to LIMIT 1 RETURN 1) > 0",
            {"from": f"{from_collection}/{from_id}", "to": f"{to_collection}/{to_id}"},
        )
        return rows[0]

    async def access_rule(self, collection: str, key: str) -> list[dict]:
        return await self._aql(
            f"FOR n IN {collection} FILTER n._key == @key "
            "RETURN {accessRule: n.accessRule}",
            {"key": key},
        )

    async def reachable_from_app(self, app_id: str, grantees: list[str]) -> set[str]:
        from .test_aql_parity import _TRAVERSAL
        from .test_qpp_semantics import HIERARCHY_TYPES

        rows = await self._aql(_TRAVERSAL, {
            "seeds": [app_id],
            "seedCollection": CollectionNames.APPS.value,
            "types": HIERARCHY_TYPES,
            "grantees": grantees,
            "allowStrict": True,
            "skipChecks": False,
            "maxDepth": 50,
        })
        return set(rows)


class _Neo4jBackend:
    name = "neo4j"

    def __init__(self, provider, settings: dict) -> None:
        self.provider = provider
        self._settings = settings

    async def _cypher(self, query: str, params: dict) -> list[dict]:
        driver = AsyncGraphDatabase.driver(
            self._settings["uri"],
            auth=(self._settings["username"], self._settings["password"]),
        )
        try:
            async with driver.session(database=self._settings["database"]) as session:
                result = await session.run(query, params)
                return [record.data() async for record in result]
        finally:
            await driver.close()

    async def edge_exists(
        self, edge_collection: str, from_collection: str, from_id: str,
        to_collection: str, to_id: str,
    ) -> bool:
        # Labels and relationship types come from the production maps, so the
        # assertion reads the graph exactly as the provider wrote it.
        rel = edge_collection_to_relationship(edge_collection)
        from_label = collection_to_label(from_collection)
        to_label = collection_to_label(to_collection)
        rows = await self._cypher(
            f"MATCH (:{from_label} {{id: $from}})-[r:{rel}]->(:{to_label} {{id: $to}}) "
            "RETURN count(r) AS n",
            {"from": from_id, "to": to_id},
        )
        return rows[0]["n"] > 0

    async def access_rule(self, collection: str, key: str) -> list[dict]:
        label = collection_to_label(collection)
        return await self._cypher(
            f"MATCH (n:{label} {{id: $key}}) "
            "RETURN n.accessRule AS accessRule",
            {"key": key},
        )

    async def reachable_from_app(self, app_id: str, grantees: list[str]) -> set[str]:
        from .test_qpp_semantics import HIERARCHY_TYPES, PER_HOP_RULE

        rows = await self._cypher(
            f"MATCH (root:App {{id: $app}}) "
            f"      ((p)-[r:NODE_RELATION]->(c) WHERE {PER_HOP_RULE})+ (n) "
            "RETURN DISTINCT n.id AS id",
            {"app": app_id, "types": HIERARCHY_TYPES, "grantees": grantees,
             "allowStrict": True, "skipChecks": False},
        )
        return {r["id"] for r in rows}


@pytest.fixture(params=["arango", "neo4j"])
def backend(request, arango_provider, arango_settings, neo4j_provider, neo4j_settings):
    if request.param == "arango":
        return _ArangoBackend(arango_provider, arango_settings)
    return _Neo4jBackend(neo4j_provider, neo4j_settings)


async def _seed_org(backend) -> None:
    """The Organization node every record group belongs to.

    Same reason as the App: Neo4j's batch_create_edges MATCHes both endpoints,
    so without this node it drops the group -> org edge while Arango writes the
    same edge dangling, and the two backends disagree for a reason unrelated to
    the code under test. `orgId` is deliberately absent -- the orgs validator
    does not declare it and runs with additionalProperties false.
    """
    await backend.provider.batch_upsert_nodes(
        [{
            "_key": ORG,
            "accountType": "enterprise",
            "isActive": True,
            "name": "Test Org",
            "createdAtTimestamp": TS,
            "updatedAtTimestamp": TS,
        }],
        collection=CollectionNames.ORGS.value,
    )


async def _seed_app(backend, connector_id: str) -> None:
    """Create the App node connector registration would already have written.

    Without it Neo4j drops every RecordGroup -> App edge (its batch_create_edges
    MATCHes both endpoints) while Arango writes the same edge dangling, so the
    two backends would disagree for a reason unrelated to the code under test.
    `_key` rather than `id`: the strict Arango validator forbids unknown fields,
    and batch_upsert_nodes renames it for Neo4j.
    """
    await _seed_org(backend)
    await backend.provider.batch_upsert_nodes(
        [{
            "_key": connector_id,
            "orgId": ORG,
            "name": "Confluence",
            "type": Connectors.CONFLUENCE.value,
            "appGroup": "Atlassian",
            "scope": "team",
            "isActive": True,
            "createdAtTimestamp": TS,
        }],
        collection=CollectionNames.APPS.value,
    )


async def _seed_user(backend, email: str) -> str:
    """A user node the processor can resolve a grant against.

    `_handle_record_permissions` looks a USER grant up by email and silently
    skips when no user exists, so without this the grant would never become an
    edge and the assertions below would pass for the wrong reason.
    """
    user_id = f"user-{_suffix()}"
    await backend.provider.batch_upsert_nodes(
        [{"_key": user_id, "email": email, "orgId": ORG,
          "isActive": True, "createdAtTimestamp": TS, "updatedAtTimestamp": TS}],
        collection=CollectionNames.USERS.value,
    )
    return user_id


def _group(external_id: str, *, inherit: bool, connector_id: str,
           rule: AccessRule = AccessRule.RESTRICTED) -> RecordGroup:
    return RecordGroup(
        org_id=ORG,
        name=f"Space {external_id}",
        external_group_id=external_id,
        connector_name=Connectors.CONFLUENCE,
        connector_id=connector_id,
        group_type=RecordGroupType.CONFLUENCE_SPACES,
        inherit_permissions=inherit,
        access_rule=rule,
    )


def _record(external_id: str, *, connector_id: str, parent_external: str | None = None,
            group_external: str | None = None, inherit: bool = True,
            rule: AccessRule = AccessRule.STRICT) -> WebpageRecord:
    # CONFLUENCE_PAGE is in RECORD_TYPE_COLLECTION_MAPPING, so both providers
    # call to_arango_record() for the type document — which only the subclasses
    # implement. A bare Record raises AttributeError inside batch_upsert_records.
    return WebpageRecord(
        org_id=ORG,
        record_name=f"Page {external_id}",
        external_record_id=external_id,
        record_type=RecordType.CONFLUENCE_PAGE,
        parent_external_record_id=parent_external,
        parent_record_type=RecordType.CONFLUENCE_PAGE if parent_external else None,
        external_record_group_id=group_external,
        record_group_type=RecordGroupType.CONFLUENCE_SPACES if group_external else None,
        version=1,
        origin=OriginTypes.CONNECTOR,
        connector_name=Connectors.CONFLUENCE,
        connector_id=connector_id,
        inherit_permissions=inherit,
        access_rule=rule,
    )


async def test_top_level_group_inherits_from_its_app(backend) -> None:
    """The edge the Confluence model depends on, and which did not exist before."""
    processor, _ = build_processor(backend.provider, ORG)
    connector_id = f"conn-{_suffix()}"
    await _seed_app(backend, connector_id)
    group = _group(f"space-{_suffix()}", inherit=True, connector_id=connector_id)

    await processor.on_new_record_groups([(group, [])])

    assert await backend.edge_exists(
        CollectionNames.INHERIT_PERMISSIONS.value,
        CollectionNames.RECORD_GROUPS.value, group.id,
        CollectionNames.APPS.value, connector_id,
    ), "a top-level group with inherit_permissions=True must inherit from its App"


async def test_group_that_stops_inheriting_loses_the_edge(backend) -> None:
    """The reconcile gap: a stale edge would widen access on re-sync."""
    processor, _ = build_processor(backend.provider, ORG)
    connector_id = f"conn-{_suffix()}"
    await _seed_app(backend, connector_id)
    external_id = f"space-{_suffix()}"

    first = _group(external_id, inherit=True, connector_id=connector_id)
    await processor.on_new_record_groups([(first, [])])
    # Without this the test passes whenever the edge never existed — including
    # if the processor stopped reusing the stored group id, which is the only
    # reason the assertion below addresses the same vertex twice.
    assert await backend.edge_exists(
        CollectionNames.INHERIT_PERMISSIONS.value,
        CollectionNames.RECORD_GROUPS.value, first.id,
        CollectionNames.APPS.value, connector_id,
    ), "control: the group must inherit from its App before inheritance is turned off"

    stopped = _group(external_id, inherit=False, connector_id=connector_id)
    await processor.on_new_record_groups([(stopped, [])])

    assert not await backend.edge_exists(
        CollectionNames.INHERIT_PERMISSIONS.value,
        CollectionNames.RECORD_GROUPS.value, stopped.id,
        CollectionNames.APPS.value, connector_id,
    ), "turning inheritance off must remove the edge, not leave it behind"


async def test_access_rule_persists_through_the_write_path(backend) -> None:
    """accessRule survives the processor, not just the model."""
    processor, _ = build_processor(backend.provider, ORG)
    connector_id = f"conn-{_suffix()}"
    await _seed_app(backend, connector_id)
    group = _group(f"space-{_suffix()}", inherit=True, connector_id=connector_id)

    await processor.on_new_record_groups([(group, [])])

    stored = await backend.access_rule(CollectionNames.RECORD_GROUPS.value, group.id)
    assert stored == [{"accessRule": "RESTRICTED"}], stored


async def test_nested_record_inherits_from_its_parent_record(backend) -> None:
    """D6: inheritance follows the hierarchy, not a shortcut to the record group.

    Without this a restriction part way down a page tree cannot take effect,
    because every descendant inherits straight past it from the group.
    """
    processor, _ = build_processor(backend.provider, ORG)
    connector_id = f"conn-{_suffix()}"
    await _seed_app(backend, connector_id)
    parent_external = f"page-{_suffix()}"
    child_external = f"page-{_suffix()}"

    parent = _record(parent_external, connector_id=connector_id)
    await processor.on_new_records([(parent, [])])

    child = _record(child_external, connector_id=connector_id,
                    parent_external=parent_external)
    await processor.on_new_records([(child, [])])

    assert await backend.edge_exists(
        CollectionNames.INHERIT_PERMISSIONS.value,
        CollectionNames.RECORDS.value, child.id,
        CollectionNames.RECORDS.value, parent.id,
    ), "a nested record must inherit from the record directly above it"


async def test_top_level_group_hangs_off_its_app(backend) -> None:
    """The traversal descends NODE_RELATION from the App; BELONGS_TO runs the
    other way and cannot be walked downwards."""
    processor, _ = build_processor(backend.provider, ORG)
    connector_id = f"conn-{_suffix()}"
    await _seed_app(backend, connector_id)
    group = _group(f"space-{_suffix()}", inherit=True, connector_id=connector_id)

    await processor.on_new_record_groups([(group, [])])

    assert await backend.edge_exists(
        CollectionNames.NODE_RELATIONS.value,
        CollectionNames.APPS.value, connector_id,
        CollectionNames.RECORD_GROUPS.value, group.id,
    ), "a top-level group must be reachable downwards from its App"
    # The upward edges the read side never sees, because the hand-built fixture
    # omits both. A collection was broken for exactly this reason: the query
    # required an edge no writer emits, and nothing compared the two.
    assert await backend.edge_exists(
        CollectionNames.BELONGS_TO.value,
        CollectionNames.RECORD_GROUPS.value, group.id,
        CollectionNames.APPS.value, connector_id,
    ), "a top-level group must also belong to its App"
    assert await backend.edge_exists(
        CollectionNames.BELONGS_TO.value,
        CollectionNames.RECORD_GROUPS.value, group.id,
        CollectionNames.ORGS.value, ORG,
    ), "every record group belongs to its org"


async def test_top_level_record_hangs_off_its_record_group(backend) -> None:
    """Without this edge the root pass cannot reach a record at all."""
    processor, _ = build_processor(backend.provider, ORG)
    connector_id = f"conn-{_suffix()}"
    await _seed_app(backend, connector_id)
    group_external = f"space-{_suffix()}"
    group = _group(group_external, inherit=True, connector_id=connector_id)
    await processor.on_new_record_groups([(group, [])])

    record = _record(f"page-{_suffix()}", connector_id=connector_id,
                     group_external=group_external)
    await processor.on_new_records([(record, [])])

    assert await backend.edge_exists(
        CollectionNames.NODE_RELATIONS.value,
        CollectionNames.RECORD_GROUPS.value, group.id,
        CollectionNames.RECORDS.value, record.id,
    ), "a record with no parent record must hang off its record group"
    # Placement reads this edge to find a chain-top's own group, so a record
    # that hangs off a group without belonging to it would list correctly and
    # then be placed under the App.
    assert await backend.edge_exists(
        CollectionNames.BELONGS_TO.value,
        CollectionNames.RECORDS.value, record.id,
        CollectionNames.RECORD_GROUPS.value, group.id,
    ), "a record must belong to the record group it hangs off"


async def test_nested_record_gets_no_edge_from_the_group(backend) -> None:
    """The bypass guard for D6: a hierarchy edge straight from the group would
    let the traversal reach a nested record while skipping every restriction
    between them, which is the whole reason inheritance follows the parent."""
    processor, _ = build_processor(backend.provider, ORG)
    connector_id = f"conn-{_suffix()}"
    await _seed_app(backend, connector_id)
    group_external = f"space-{_suffix()}"
    group = _group(group_external, inherit=True, connector_id=connector_id)
    await processor.on_new_record_groups([(group, [])])

    parent_external = f"page-{_suffix()}"
    parent = _record(parent_external, connector_id=connector_id,
                     group_external=group_external)
    await processor.on_new_records([(parent, [])])

    child = _record(f"page-{_suffix()}", connector_id=connector_id,
                    parent_external=parent_external, group_external=group_external)
    await processor.on_new_records([(child, [])])

    assert not await backend.edge_exists(
        CollectionNames.NODE_RELATIONS.value,
        CollectionNames.RECORD_GROUPS.value, group.id,
        CollectionNames.RECORDS.value, child.id,
    ), "a nested record must be reached through its parent, not from the group"


async def _tree(backend, processor):
    """A group holding a parent page with one child page under it."""
    connector_id = f"conn-{_suffix()}"
    await _seed_app(backend, connector_id)
    group_external = f"space-{_suffix()}"
    group = _group(group_external, inherit=True, connector_id=connector_id)
    await processor.on_new_record_groups([(group, [])])

    parent_external = f"page-{_suffix()}"
    parent = _record(parent_external, connector_id=connector_id,
                     group_external=group_external)
    await processor.on_new_records([(parent, [])])

    child = _record(f"page-{_suffix()}", connector_id=connector_id,
                    parent_external=parent_external, group_external=group_external)
    await processor.on_new_records([(child, [])])
    return connector_id, group, parent, child


async def _assert_reparented(backend, group, child) -> None:
    assert await backend.edge_exists(
        CollectionNames.NODE_RELATIONS.value,
        CollectionNames.RECORD_GROUPS.value, group.id,
        CollectionNames.RECORDS.value, child.id,
    ), "a survivor must hang off its record group once its parent is gone"
    assert await backend.edge_exists(
        CollectionNames.INHERIT_PERMISSIONS.value,
        CollectionNames.RECORDS.value, child.id,
        CollectionNames.RECORD_GROUPS.value, group.id,
    ), "a survivor with no inheritance edge is invisible without a direct grant"


async def test_cascade_delete_reparents_the_survivor(backend) -> None:
    """Decision 73. The edge sweep takes the survivor's only inheritance edge
    with it, because decision 6 made it inherit from its parent record."""
    processor, _ = build_processor(backend.provider, ORG)
    connector_id, group, parent, child = await _tree(backend, processor)

    await processor.on_records_deleted_cascade(
        [parent.id], connector_id, cascade_children=False,
    )

    await _assert_reparented(backend, group, child)


async def test_single_delete_reparents_the_survivor(backend) -> None:
    """The same gap on the per-record path connectors hit most often."""
    processor, _ = build_processor(backend.provider, ORG)
    _connector_id, group, parent, child = await _tree(backend, processor)

    await processor.on_record_deleted(parent.id)

    await _assert_reparented(backend, group, child)
    assert not await backend.edge_exists(
        CollectionNames.NODE_RELATIONS.value,
        CollectionNames.RECORDS.value, parent.id,
        CollectionNames.RECORDS.value, child.id,
    ), "the edge from the deleted parent must not be left dangling"


async def test_a_restricted_page_needs_a_grant_as_well_as_inheritance(backend) -> None:
    """Decisions 26 and 66, end to end on a graph the write path produced.

    The space is strict but *not* restricted, so both users clear the group hop
    and the assertions isolate the record-level rule. The two pages differ only
    in the restriction flag and the grant: both inherit from the same space.
    """
    processor, _ = build_processor(backend.provider, ORG)
    connector_id = f"conn-{_suffix()}"
    await _seed_app(backend, connector_id)

    granted_email = f"u-{_suffix()}@example.com"
    granted_user = await _seed_user(backend, granted_email)
    other_user = await _seed_user(backend, f"v-{_suffix()}@example.com")

    group_external = f"space-{_suffix()}"
    group = _group(group_external, inherit=True, connector_id=connector_id,
                   rule=AccessRule.STRICT)
    await processor.on_new_record_groups([(group, [])])

    restricted = _record(f"page-{_suffix()}", connector_id=connector_id,
                         group_external=group_external, rule=AccessRule.RESTRICTED)
    await processor.on_new_records([(restricted, [Permission(
        email=granted_email, type=PermissionType.READ, entity_type=EntityType.USER,
    )])])

    open_page = _record(f"page-{_suffix()}", connector_id=connector_id,
                        group_external=group_external)
    await processor.on_new_records([(open_page, [])])

    seen_by_granted = await backend.reachable_from_app(connector_id, [granted_user])
    seen_by_other = await backend.reachable_from_app(connector_id, [other_user])

    # Guards the guard: if the open page were invisible to both, the assertion
    # below would hold for the wrong reason.
    assert open_page.id in seen_by_granted, "the granted user must see the open page"
    assert open_page.id in seen_by_other, (
        "an unrestricted page must be reachable through inheritance alone"
    )
    assert restricted.id in seen_by_granted, (
        "the granted user must see the restricted page"
    )
    assert restricted.id not in seen_by_other, (
        "B2: inheritance alone must never reveal a restricted page"
    )


async def test_a_record_that_gains_a_parent_loses_its_group_edge(backend) -> None:
    """D6 by the route the negative test above cannot reach.

    That test creates the record already nested, so the group edge is never
    written. Here it is written correctly at root and the record is only then
    re-parented — and nothing removed it, leaving the traversal able to reach
    the record from the group while skipping every restriction in between.
    """
    processor, _ = build_processor(backend.provider, ORG)
    connector_id = f"conn-{_suffix()}"
    await _seed_app(backend, connector_id)
    group_external = f"space-{_suffix()}"
    group = _group(group_external, inherit=True, connector_id=connector_id)
    await processor.on_new_record_groups([(group, [])])

    parent_external = f"page-{_suffix()}"
    await processor.on_new_records([(
        _record(parent_external, connector_id=connector_id,
                group_external=group_external), [])])

    child_external = f"page-{_suffix()}"
    at_root = _record(child_external, connector_id=connector_id,
                      group_external=group_external)
    await processor.on_new_records([(at_root, [])])
    assert await backend.edge_exists(
        CollectionNames.NODE_RELATIONS.value,
        CollectionNames.RECORD_GROUPS.value, group.id,
        CollectionNames.RECORDS.value, at_root.id,
    ), "control: a root record hangs off its group before it is re-parented"

    nested = _record(child_external, connector_id=connector_id,
                     parent_external=parent_external, group_external=group_external)
    await processor.on_new_records([(nested, [])])

    assert not await backend.edge_exists(
        CollectionNames.NODE_RELATIONS.value,
        CollectionNames.RECORD_GROUPS.value, group.id,
        CollectionNames.RECORDS.value, at_root.id,
    ), "a record that gains a parent must stop hanging off its group"


async def test_a_record_re_synced_at_root_keeps_its_group_edge(backend) -> None:
    """The mirror image, and why the stale-parent delete is scoped to records.

    An unscoped delete matched every incoming PARENT_CHILD edge, including the
    group's — written moments earlier in the same sync — leaving the record
    with no hierarchy parent at all and unreachable from the App.
    """
    processor, _ = build_processor(backend.provider, ORG)
    connector_id = f"conn-{_suffix()}"
    await _seed_app(backend, connector_id)
    group_external = f"space-{_suffix()}"
    group = _group(group_external, inherit=True, connector_id=connector_id)
    await processor.on_new_record_groups([(group, [])])

    parent_external = f"page-{_suffix()}"
    parent = _record(parent_external, connector_id=connector_id,
                     group_external=group_external)
    await processor.on_new_records([(parent, [])])

    child_external = f"page-{_suffix()}"
    nested = _record(child_external, connector_id=connector_id,
                     parent_external=parent_external, group_external=group_external)
    await processor.on_new_records([(nested, [])])

    promoted = _record(child_external, connector_id=connector_id,
                       group_external=group_external)
    await processor.on_new_records([(promoted, [])])

    assert await backend.edge_exists(
        CollectionNames.NODE_RELATIONS.value,
        CollectionNames.RECORD_GROUPS.value, group.id,
        CollectionNames.RECORDS.value, nested.id,
    ), "a record re-synced at root must hang off its group"
    assert not await backend.edge_exists(
        CollectionNames.NODE_RELATIONS.value,
        CollectionNames.RECORDS.value, parent.id,
        CollectionNames.RECORDS.value, nested.id,
    ), "and must lose the edge from the parent it no longer has"


async def test_a_re_parented_record_stops_inheriting_from_its_old_parent(backend) -> None:
    """Decision 6 made a record inherit from its parent record, so a stale
    inheritance edge keeps granting access through a parent it no longer has."""
    processor, _ = build_processor(backend.provider, ORG)
    connector_id = f"conn-{_suffix()}"
    await _seed_app(backend, connector_id)
    group_external = f"space-{_suffix()}"
    group = _group(group_external, inherit=True, connector_id=connector_id)
    await processor.on_new_record_groups([(group, [])])

    first_external = f"page-{_suffix()}"
    first = _record(first_external, connector_id=connector_id,
                    group_external=group_external)
    second_external = f"page-{_suffix()}"
    second = _record(second_external, connector_id=connector_id,
                     group_external=group_external)
    await processor.on_new_records([(first, []), (second, [])])

    child_external = f"page-{_suffix()}"
    child = _record(child_external, connector_id=connector_id,
                    parent_external=first_external, group_external=group_external)
    await processor.on_new_records([(child, [])])
    assert await backend.edge_exists(
        CollectionNames.INHERIT_PERMISSIONS.value,
        CollectionNames.RECORDS.value, child.id,
        CollectionNames.RECORDS.value, first.id,
    ), "control: the record must inherit from its first parent"

    moved = _record(child_external, connector_id=connector_id,
                    parent_external=second_external, group_external=group_external)
    await processor.on_new_records([(moved, [])])

    assert await backend.edge_exists(
        CollectionNames.INHERIT_PERMISSIONS.value,
        CollectionNames.RECORDS.value, child.id,
        CollectionNames.RECORDS.value, second.id,
    ), "it must inherit from the parent it moved to"
    assert not await backend.edge_exists(
        CollectionNames.INHERIT_PERMISSIONS.value,
        CollectionNames.RECORDS.value, child.id,
        CollectionNames.RECORDS.value, first.id,
    ), "and must stop inheriting from the one it left"


async def test_a_group_that_gains_a_parent_stops_hanging_off_the_app(backend) -> None:
    """D16/D66: those edges belong to a top-level group only.

    Left behind, they let the traversal reach the group straight from the App,
    never passing the parent group's own checks.
    """
    processor, _ = build_processor(backend.provider, ORG)
    connector_id = f"conn-{_suffix()}"
    await _seed_app(backend, connector_id)

    parent_external = f"space-{_suffix()}"
    parent_group = _group(parent_external, inherit=True, connector_id=connector_id)
    await processor.on_new_record_groups([(parent_group, [])])

    child_external = f"space-{_suffix()}"
    top_level = _group(child_external, inherit=True, connector_id=connector_id)
    await processor.on_new_record_groups([(top_level, [])])
    assert await backend.edge_exists(
        CollectionNames.NODE_RELATIONS.value,
        CollectionNames.APPS.value, connector_id,
        CollectionNames.RECORD_GROUPS.value, top_level.id,
    ), "control: a top-level group hangs off its App"

    nested = _group(child_external, inherit=True, connector_id=connector_id)
    nested.parent_external_group_id = parent_external
    await processor.on_new_record_groups([(nested, [])])

    assert await backend.edge_exists(
        CollectionNames.NODE_RELATIONS.value,
        CollectionNames.RECORD_GROUPS.value, parent_group.id,
        CollectionNames.RECORD_GROUPS.value, top_level.id,
    ), "the group must hang off its new parent"
    assert not await backend.edge_exists(
        CollectionNames.NODE_RELATIONS.value,
        CollectionNames.APPS.value, connector_id,
        CollectionNames.RECORD_GROUPS.value, top_level.id,
    ), "and must no longer hang off the App directly"
    assert not await backend.edge_exists(
        CollectionNames.INHERIT_PERMISSIONS.value,
        CollectionNames.RECORD_GROUPS.value, top_level.id,
        CollectionNames.APPS.value, connector_id,
    ), "nor inherit from the App over its parent group's head"


async def test_deleting_a_record_leaves_no_shared_with_me_edge(backend) -> None:
    """D55 gives a record a second hierarchy parent, and that id is never
    stored on the record document.

    So a delete that cleans only the record group's edge — the one it *can*
    find, from `recordGroupId` — leaves the Shared with Me edge pointing at a
    vertex that no longer exists. Arango removes the document without touching
    its edges, so nothing downstream catches it and a traversal from the inbox
    crosses into a null vertex.
    """
    processor, _ = build_processor(backend.provider, ORG)
    connector_id = f"conn-{_suffix()}"
    await _seed_app(backend, connector_id)

    drive_external = f"space-{_suffix()}"
    drive = _group(drive_external, inherit=True, connector_id=connector_id)
    inbox_external = f"space-{_suffix()}"
    inbox = _group(inbox_external, inherit=True, connector_id=connector_id)
    await processor.on_new_record_groups([(drive, []), (inbox, [])])

    record = _record(f"page-{_suffix()}", connector_id=connector_id,
                     group_external=drive_external)
    record.shared_with_me_record_group_ids = [inbox_external]
    await processor.on_new_records([(record, [])])

    assert await backend.edge_exists(
        CollectionNames.NODE_RELATIONS.value,
        CollectionNames.RECORD_GROUPS.value, inbox.id,
        CollectionNames.RECORDS.value, record.id,
    ), "control: Shared with Me must be a real second hierarchy parent (D55)"

    await processor.on_record_deleted(record.id)

    assert not await backend.edge_exists(
        CollectionNames.NODE_RELATIONS.value,
        CollectionNames.RECORD_GROUPS.value, inbox.id,
        CollectionNames.RECORDS.value, record.id,
    ), "the second hierarchy edge must not outlive the record it pointed at"
    assert not await backend.edge_exists(
        CollectionNames.NODE_RELATIONS.value,
        CollectionNames.RECORD_GROUPS.value, drive.id,
        CollectionNames.RECORDS.value, record.id,
    ), "nor the record group's own edge"
