"""Targeted coverage tests for NotionConnector gaps (95% goal)."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from app.config.constants.arangodb import Connectors, MimeTypes, OriginTypes, ProgressStatus
from app.connectors.core.registry.filters import IndexingFilterKey
from app.connectors.sources.notion.block_parser import NotionBlockParser
from app.connectors.sources.notion.connector import NotionConnector
from app.models.blocks import (
    Block,
    BlockComment,
    BlockContainerIndex,
    BlockGroup,
    BlockGroupChildren,
    BlockSubType,
    BlockType,
    ChildRecord,
    ChildType,
    CommentAttachment,
    DataFormat,
    GroupSubType,
    GroupType,
)
from app.models.entities import FileRecord, RecordGroupType, RecordType, WebpageRecord


def _connector():
    logger = MagicMock()
    dep = MagicMock()
    dep.org_id = "org-1"
    dep.on_new_app_users = AsyncMock()
    dep.on_new_records = AsyncMock()
    dep.on_new_record_groups = AsyncMock()
    dsp = MagicMock()
    mock_tx = MagicMock()
    mock_tx.get_record_by_external_id = AsyncMock(return_value=None)
    mock_tx.get_record_group_by_external_id = AsyncMock(return_value=None)
    mock_tx.get_user_by_source_id = AsyncMock(return_value=None)
    mock_tx.__aenter__ = AsyncMock(return_value=mock_tx)
    mock_tx.__aexit__ = AsyncMock(return_value=None)
    dsp.transaction.return_value = mock_tx
    conn = NotionConnector(
        logger=logger,
        data_entities_processor=dep,
        data_store_provider=dsp,
        config_service=AsyncMock(),
        connector_id="notion-conn-1",
        scope="team",
        created_by="test-user",
    )
    conn.workspace_id = "ws-1"
    return conn


def _api_resp(success=True, data=None, error=None):
    resp = MagicMock()
    resp.success = success
    resp.error = error
    if data is not None:
        resp.data = MagicMock()
        resp.data.json.return_value = data
    else:
        resp.data = None
    return resp


def _webpage(**kwargs):
    defaults = dict(
        org_id="org-1",
        record_name="Page",
        record_type=RecordType.WEBPAGE,
        external_record_id="page-1",
        connector_id="notion-conn-1",
        connector_name=Connectors.NOTION,
        record_group_type=RecordGroupType.NOTION_WORKSPACE,
        external_record_group_id="ws-1",
        mime_type=MimeTypes.BLOCKS.value,
        version=1,
        origin=OriginTypes.CONNECTOR,
    )
    defaults.update(kwargs)
    return WebpageRecord(**defaults)


def _file_record(**kwargs):
    defaults = dict(
        org_id="org-1",
        record_name="doc.pdf",
        record_type=RecordType.FILE,
        external_record_id="block-file-1",
        connector_id="notion-conn-1",
        connector_name=Connectors.NOTION,
        record_group_type=RecordGroupType.NOTION_WORKSPACE,
        external_record_group_id="ws-1",
        mime_type=MimeTypes.PDF.value,
        version=1,
        origin=OriginTypes.CONNECTOR,
        is_file=True,
        signed_url="https://fallback.example/doc.pdf",
    )
    defaults.update(kwargs)
    return FileRecord(**defaults)


class TestCreatePageLevelCommentGroups:
    @pytest.mark.asyncio
    async def test_creates_thread_groups_with_attachments(self):
        conn = _connector()
        parser = NotionBlockParser(conn.logger)
        file_rec = _file_record(external_record_id="ca_c1_doc.pdf")
        file_rec.id = "fr-1"
        block_comment = BlockComment(
            text="Comment text",
            format=DataFormat.TXT,
            author_id="u1",
            author_name="Alice",
            attachments=[CommentAttachment(name="doc.pdf", id="fr-1")],
        )
        conn._create_block_comment_from_notion_comment = AsyncMock(
            return_value=(block_comment, [file_rec])
        )

        blocks = []
        groups = []
        page_comments = [
            ({"id": "c1", "discussion_id": "thread-1", "rich_text": []}, "page-1"),
            ({"id": "c2", "discussion_id": "thread-1", "rich_text": []}, "page-1"),
        ]
        result = await conn._create_page_level_comment_groups(
            groups, blocks, page_comments, "page-1", parser, "https://notion.so/page-1"
        )
        assert len(result) == 2
        assert any(g.sub_type == GroupSubType.COMMENT for g in groups)
        assert any(g.sub_type == GroupSubType.COMMENT_THREAD for g in groups)
        assert any(b.sub_type == BlockSubType.CHILD_RECORD for b in blocks)

    @pytest.mark.asyncio
    async def test_skips_missing_comment_id_and_failed_parse(self):
        conn = _connector()
        parser = NotionBlockParser(conn.logger)
        conn._create_block_comment_from_notion_comment = AsyncMock(return_value=(None, []))

        groups = []
        blocks = []
        page_comments = [
            ({"discussion_id": "t1"}, "page-1"),
            ({"id": "c2", "discussion_id": "t1"}, "page-1"),
        ]
        result = await conn._create_page_level_comment_groups(
            groups, blocks, page_comments, "page-1", parser
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_comment_loop_exception_continues(self):
        conn = _connector()
        parser = NotionBlockParser(conn.logger)
        conn._create_block_comment_from_notion_comment = AsyncMock(
            side_effect=[Exception("fail"), (BlockComment(text="ok", format=DataFormat.TXT), [])]
        )
        groups = []
        blocks = []
        page_comments = [
            ({"id": "c1", "discussion_id": "t1"}, "page-1"),
            ({"id": "c2", "discussion_id": "t1"}, "page-1"),
        ]
        await conn._create_page_level_comment_groups(groups, blocks, page_comments, "page-1", parser)
        conn.logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_attachment_without_matching_file_record_skipped(self):
        conn = _connector()
        parser = NotionBlockParser(conn.logger)
        block_comment = BlockComment(
            text="x",
            format=DataFormat.TXT,
            attachments=[CommentAttachment(name="x.pdf", id="missing-id")],
        )
        conn._create_block_comment_from_notion_comment = AsyncMock(
            return_value=(block_comment, [])
        )
        groups, blocks = [], []
        await conn._create_page_level_comment_groups(
            groups, blocks,
            [({"id": "c1", "discussion_id": "t1"}, "page-1")],
            "page-1", parser,
        )
        assert blocks == []


class _FakeHttpxStreamContext:
    def __init__(self, chunks):
        self._chunks = chunks

    async def __aenter__(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()

        async def _aiter():
            for chunk in self._chunks:
                yield chunk

        response.aiter_bytes = _aiter
        return response

    async def __aexit__(self, *_args):
        return None


class _FakeHttpxClient:
    def __init__(self, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def stream(self, _method, _url, **_kwargs):
        return _FakeHttpxStreamContext([b"part-a", b"part-b"])


class TestStreamRecordFileChunks:
    @pytest.mark.asyncio
    async def test_file_stream_yields_bytes(self):
        conn = _connector()
        conn.data_source = MagicMock()
        conn.get_signed_url = AsyncMock(return_value="https://signed.example/file.pdf")
        record = _file_record()

        with patch.object(httpx, "AsyncClient", _FakeHttpxClient):
            response = await conn.stream_record(record)
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
        assert chunks == [b"part-a", b"part-b"]


class TestGetBlockAndCommentUrls:
    @pytest.mark.asyncio
    async def test_block_file_url_missing_returns_none(self):
        conn = _connector()
        ds = MagicMock()
        ds.retrieve_block = AsyncMock(return_value=_api_resp(True, {
            "id": "b1", "type": "pdf", "pdf": {"type": "external", "external": {"url": ""}}},
        ))
        conn._get_fresh_datasource = AsyncMock(return_value=ds)
        record = _file_record(external_record_id="b1")
        assert await conn._get_block_file_url(record) is None

    @pytest.mark.asyncio
    async def test_comment_attachment_no_match_returns_signed_url(self):
        conn = _connector()
        ds = MagicMock()
        ds.retrieve_comment = AsyncMock(return_value=_api_resp(True, {
            "attachments": [{"file": {"url": "https://x.com/other.pdf"}, "name": "other.pdf"}],
        }))
        conn._get_fresh_datasource = AsyncMock(return_value=ds)
        conn._normalize_filename_for_id = MagicMock(side_effect=lambda name: name)
        record = _file_record(
            external_record_id="ca_comment1_target.pdf",
            signed_url="https://fallback.example/f.pdf",
        )
        url = await conn._get_comment_attachment_url(record)
        assert url == "https://fallback.example/f.pdf"


class TestResolveDatabaseToDataSourcesParents:
    @pytest.mark.asyncio
    async def test_database_parent_types(self):
        conn = _connector()
        ds = MagicMock()
        ds.retrieve_database = AsyncMock(return_value=_api_resp(True, {
            "data_sources": [{"id": "ds-1", "name": "View 1"}],
            "parent": {"type": "database_id", "database_id": "parent-db"},
        }))
        conn._get_fresh_datasource = AsyncMock(return_value=ds)
        conn._batch_get_or_create_child_records = AsyncMock(return_value={
            "ds-1": ChildRecord(child_type=ChildType.RECORD, child_id="r1", child_name="View 1"),
        })
        result = await conn._resolve_database_to_data_sources("db-1")
        assert len(result) == 1
        args = conn._batch_get_or_create_child_records.await_args[0][0]
        assert args["ds-1"][2] == "parent-db"

    @pytest.mark.asyncio
    async def test_block_id_and_data_source_id_parents(self):
        conn = _connector()
        ds = MagicMock()
        ds.retrieve_database = AsyncMock(side_effect=[
            _api_resp(True, {
                "data_sources": [{"id": "ds-a", "name": "A"}],
                "parent": {"type": "block_id", "block_id": "blk-parent"},
            }),
            _api_resp(True, {
                "data_sources": [{"id": "ds-b", "name": "B"}],
                "parent": {"type": "data_source_id", "data_source_id": "ds-parent"},
            }),
        ])
        conn._get_fresh_datasource = AsyncMock(return_value=ds)
        conn._resolve_block_parent_recursive = AsyncMock(return_value=("page-1", RecordType.WEBPAGE))
        conn._batch_get_or_create_child_records = AsyncMock(return_value={
            "ds-a": ChildRecord(child_type=ChildType.RECORD, child_id="r1", child_name="A"),
            "ds-b": ChildRecord(child_type=ChildType.RECORD, child_id="r2", child_name="B"),
        })
        await conn._resolve_database_to_data_sources("db-a")
        await conn._resolve_database_to_data_sources("db-b")
        assert conn._resolve_block_parent_recursive.awaited

    @pytest.mark.asyncio
    async def test_skips_data_source_without_id(self):
        conn = _connector()
        ds = MagicMock()
        ds.retrieve_database = AsyncMock(return_value=_api_resp(True, {
            "data_sources": [{"name": "No ID"}],
            "parent": {"type": "page_id", "page_id": "p1"},
        }))
        conn._get_fresh_datasource = AsyncMock(return_value=ds)
        result = await conn._resolve_database_to_data_sources("db-1")
        assert result == []


class TestProcessBlocksRecursiveSynced:
    @pytest.mark.asyncio
    async def test_synced_block_reference_fetches_from_original(self):
        conn = _connector()
        parser = NotionBlockParser(conn.logger)
        blocks, groups = [], []

        child_blocks = [{"id": "child-1", "type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "x"}]}}]
        conn._fetch_block_children_recursive = AsyncMock(side_effect=[
            [{
                "id": "sync-1",
                "type": "synced_block",
                "synced_block": {"synced_from": {"type": "block_id", "block_id": "orig-1"}},
                "has_children": True,
            }],
            child_blocks,
            [],
        ])

        await conn._process_blocks_recursive(
            "page-1", parser, blocks, groups, parent_group_index=None, parent_page_id="page-1"
        )
        assert conn._fetch_block_children_recursive.await_count >= 2
        calls = [c.args[0] for c in conn._fetch_block_children_recursive.await_args_list]
        assert "orig-1" in calls

    @pytest.mark.asyncio
    async def test_unknown_block_with_children_still_recurses(self):
        conn = _connector()
        parser = NotionBlockParser(conn.logger)
        blocks, groups = [], []

        conn._fetch_block_children_recursive = AsyncMock(side_effect=[
            [{"id": "unk-1", "type": "unsupported_type_xyz", "unsupported_type_xyz": {}, "has_children": True}],
            [],
        ])

        await conn._process_blocks_recursive(
            "page-1", parser, blocks, groups,
            parent_group_index=None, parent_page_id="page-1",
        )
        conn.logger.warning.assert_called()


class TestTransformWebpageRecordExtended:
    @pytest.mark.asyncio
    async def test_database_object_type(self):
        conn = _connector()
        record = await conn._transform_to_webpage_record(
            {"id": "db-1", "title": [{"plain_text": "DB"}], "created_time": "2025-01-01T00:00:00.000Z"},
            "database",
        )
        assert record.record_type == RecordType.DATABASE

    @pytest.mark.asyncio
    async def test_data_source_with_database_parent_id(self):
        conn = _connector()
        record = await conn._transform_to_webpage_record(
            {
                "id": "ds-1",
                "title": [{"plain_text": "DS"}],
                "created_time": "2025-01-01T00:00:00.000Z",
            },
            "data_source",
            database_parent_id="page-parent",
        )
        assert record.parent_external_record_id == "page-parent"
        assert record.parent_record_type == RecordType.WEBPAGE

    @pytest.mark.asyncio
    async def test_page_parent_block_id_and_data_source_id(self):
        conn = _connector()
        conn._resolve_block_parent_recursive = AsyncMock(return_value=("resolved-page", RecordType.WEBPAGE))
        record_block = await conn._transform_to_webpage_record(
            {
                "id": "p1",
                "properties": {"title": {"type": "title", "title": [{"plain_text": "P"}]}},
                "parent": {"type": "block_id", "block_id": "blk-1"},
            },
            "page",
        )
        assert record_block.parent_external_record_id == "resolved-page"

        record_ds_parent = await conn._transform_to_webpage_record(
            {
                "id": "p2",
                "properties": {"title": {"type": "title", "title": [{"plain_text": "P2"}]}},
                "parent": {"type": "data_source_id", "data_source_id": "ds-parent"},
            },
            "page",
        )
        assert record_ds_parent.parent_record_type == RecordType.DATASOURCE


class TestSyncObjectsByTypeExtended:
    @pytest.mark.asyncio
    async def test_delta_sync_stops_at_threshold(self):
        conn = _connector()
        conn.indexing_filters = MagicMock()
        conn.indexing_filters.is_enabled = MagicMock(return_value=True)
        conn.pages_sync_point = MagicMock()
        conn.pages_sync_point.read_sync_point = AsyncMock(return_value={"last_sync_time": "2025-06-01T00:00:00.000Z"})
        conn.pages_sync_point.update_sync_point = AsyncMock()

        ds = MagicMock()
        ds.search = AsyncMock(return_value=_api_resp(True, {
            "results": [{
                "id": "old-page",
                "last_edited_time": "2025-05-01T00:00:00.000Z",
                "archived": False,
                "in_trash": False,
                "properties": {"title": {"type": "title", "title": [{"plain_text": "Old"}]}},
            }],
            "has_more": False,
        }))
        conn._get_fresh_datasource = AsyncMock(return_value=ds)
        conn._transform_to_webpage_record = AsyncMock(return_value=_webpage())
        conn.data_entities_processor.on_new_records = AsyncMock()

        await conn._sync_objects_by_type("page")
        conn.data_entities_processor.on_new_records.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_data_source_database_parent_fetch_error(self):
        conn = _connector()
        conn.indexing_filters = MagicMock()
        conn.indexing_filters.is_enabled = MagicMock(return_value=True)
        conn.pages_sync_point = MagicMock()
        conn.pages_sync_point.read_sync_point = AsyncMock(return_value=None)
        conn.pages_sync_point.update_sync_point = AsyncMock()

        ds = MagicMock()
        ds.search = AsyncMock(return_value=_api_resp(True, {
            "results": [{
                "id": "ds-1",
                "last_edited_time": "2025-06-15T00:00:00.000Z",
                "title": [{"plain_text": "DS"}],
                "parent": {"type": "database_id", "database_id": "db-parent"},
            }],
            "has_more": False,
        }))
        conn._get_fresh_datasource = AsyncMock(return_value=ds)
        conn._get_database_parent_page_id = AsyncMock(side_effect=RuntimeError("parent fail"))
        conn._transform_to_webpage_record = AsyncMock(return_value=_webpage(record_type=RecordType.DATASOURCE))
        conn.data_entities_processor.on_new_records = AsyncMock()

        await conn._sync_objects_by_type("data_source")
        conn.logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_page_attachment_fetch_error_continues(self):
        conn = _connector()
        conn.indexing_filters = MagicMock()
        conn.indexing_filters.is_enabled = MagicMock(side_effect=lambda _k: _k != "files")
        conn.pages_sync_point = MagicMock()
        conn.pages_sync_point.read_sync_point = AsyncMock(return_value=None)
        conn.pages_sync_point.update_sync_point = AsyncMock()

        ds = MagicMock()
        ds.search = AsyncMock(return_value=_api_resp(True, {
            "results": [{
                "id": "p1",
                "last_edited_time": "2025-06-15T00:00:00.000Z",
                "url": "https://notion.so/p1",
                "properties": {"title": {"type": "title", "title": [{"plain_text": "P"}]}},
            }],
            "has_more": False,
        }))
        conn._get_fresh_datasource = AsyncMock(return_value=ds)
        conn._transform_to_webpage_record = AsyncMock(return_value=_webpage())
        conn._fetch_page_attachments_and_comments = AsyncMock(side_effect=RuntimeError("attach fail"))
        conn.data_entities_processor.on_new_records = AsyncMock()

        await conn._sync_objects_by_type("page")
        conn.logger.warning.assert_called()


class TestResolveUserAndPage:
    @pytest.mark.asyncio
    async def test_resolve_user_bot_with_owner(self):
        conn = _connector()
        ds = MagicMock()
        ds.retrieve_user = AsyncMock(return_value=_api_resp(True, {
            "object": "user",
            "type": "bot",
            "name": "",
            "bot": {"owner": {"type": "user", "user": {"name": "Owner Name"}}},
        }))
        conn._get_fresh_datasource = AsyncMock(return_value=ds)
        assert await conn.resolve_user_name_by_id("bot-1") == "Owner Name"

    @pytest.mark.asyncio
    async def test_resolve_page_title_from_api(self):
        conn = _connector()
        conn.data_store_provider.transaction.return_value.__aenter__ = AsyncMock(
            return_value=conn.data_store_provider.transaction.return_value
        )
        conn.data_store_provider.transaction.return_value.get_record_by_external_id = AsyncMock(return_value=None)
        ds = MagicMock()
        ds.retrieve_page = AsyncMock(return_value=_api_resp(True, {
            "properties": {"title": {"type": "title", "title": [{"plain_text": "API Title"}]}},
        }))
        conn._get_fresh_datasource = AsyncMock(return_value=ds)
        assert await conn.resolve_page_title_by_id("page-x") == "API Title"


class TestSyncUsersWorkspaceWarning:
    @pytest.mark.asyncio
    async def test_bot_without_workspace_id_logs_warning(self):
        conn = _connector()
        conn.workspace_id = None
        ds = MagicMock()
        ds.list_users = AsyncMock(return_value=_api_resp(True, {
            "results": [
                {"type": "bot", "id": "bot-1", "bot": {}},
                {"type": "person", "id": "person-1"},
            ],
            "has_more": False,
        }))
        ds.retrieve_user = AsyncMock(return_value=_api_resp(False, error="not found"))
        conn._get_fresh_datasource = AsyncMock(return_value=ds)
        conn._create_workspace_record_group = AsyncMock()
        await conn._sync_users()
        conn.logger.warning.assert_called()


class TestResolveChildReferenceApplyPaths:
    @pytest.mark.asyncio
    async def test_database_map_empty_skips_metadata(self):
        conn = _connector()
        conn._resolve_database_to_data_sources = AsyncMock(return_value=[])
        blocks = [Block(
            id="b1", index=0, type=BlockType.TEXT, sub_type=BlockSubType.CHILD_RECORD,
            format=DataFormat.TXT, data="DB", source_id="db-1", source_type="child_database",
        )]
        await conn._resolve_child_reference_blocks(blocks)
        assert blocks[0].table_row_metadata is None

    @pytest.mark.asyncio
    async def test_non_db_child_missing_from_map(self):
        conn = _connector()
        conn._batch_get_or_create_child_records = AsyncMock(return_value={})
        blocks = [Block(
            id="b1", index=0, type=BlockType.TEXT, sub_type=BlockSubType.CHILD_RECORD,
            format=DataFormat.TXT, data="Page", source_id="missing", source_type="child_page", name="WEBPAGE",
        )]
        await conn._resolve_child_reference_blocks(blocks)
        assert blocks[0].table_row_metadata is None


class TestTransformCommentFileRecordError:
    @pytest.mark.asyncio
    async def test_comment_attachment_transform_exception(self):
        conn = _connector()
        conn._normalize_filename_for_id = MagicMock(side_effect=RuntimeError("bad"))
        result = await conn._transform_to_comment_file_record(
            {"file": {"url": "https://x.com/f.pdf"}}, "c1", "page-1"
        )
        assert result is None


class TestStreamRecordJsonGenerators:
    @pytest.mark.asyncio
    async def test_webpage_stream_consumes_json_chunks(self):
        conn = _connector()
        conn.data_source = MagicMock()
        payload = '{"blocks":[]}'
        mock_container = MagicMock()
        mock_container.model_dump_json.return_value = payload
        mock_container.blocks = []
        conn._fetch_page_attachments_and_comments = AsyncMock(return_value=([], {}))
        conn._fetch_page_as_blocks = AsyncMock(return_value=mock_container)
        conn._resolve_child_reference_blocks = AsyncMock()

        response = await conn.stream_record(_webpage())
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        assert b"".join(chunks) == payload.encode("utf-8")

    @pytest.mark.asyncio
    async def test_datasource_stream_consumes_json_chunks(self):
        conn = _connector()
        conn.data_source = MagicMock()
        payload = '{"tables":[]}'
        mock_container = MagicMock()
        mock_container.model_dump_json.return_value = payload
        mock_container.blocks = []
        conn._fetch_data_source_as_blocks = AsyncMock(return_value=mock_container)
        conn._resolve_table_row_children = AsyncMock()

        response = await conn.stream_record(
            _webpage(record_type=RecordType.DATASOURCE, external_record_id="ds-1")
        )
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        assert b"".join(chunks) == payload.encode("utf-8")


class TestExtractCommentAttachmentFileRecords:
    @pytest.mark.asyncio
    async def test_duplicate_attachment_skipped(self):
        conn = _connector()
        conn._normalize_filename_for_id = MagicMock(return_value="same.pdf")
        fr = _file_record(external_record_id="ca_c1_same.pdf")
        conn._transform_to_comment_file_record = AsyncMock(return_value=fr)
        comments = {
            "page-1": [({
                "id": "c1",
                "attachments": [
                    {"file": {"url": "https://x.com/same.pdf"}, "name": "same.pdf"},
                    {"file": {"url": "https://x.com/other/same.pdf"}, "name": "same.pdf"},
                ],
            }, "page-1")],
        }
        result = await conn._extract_comment_attachment_file_records(comments, "page-1", "")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_block_level_attachment_error_continues(self):
        conn = _connector()
        comments = {
            "blk-1": [({"id": "c1", "attachments": [{"file": {"url": "https://x.com/a.pdf"}}]}, "blk-1")],
        }
        conn._transform_to_comment_file_record = AsyncMock(side_effect=RuntimeError("fail"))
        result = await conn._extract_comment_attachment_file_records(comments, "page-1", "")
        assert result == []


class TestConvertImageBlocksToBase64:
    @pytest.mark.asyncio
    async def test_converts_image_block(self):
        conn = _connector()
        block = Block(
            id="img-1", index=0, type=BlockType.IMAGE, format=DataFormat.TXT,
            data="caption", source_id="img-block-1",
            public_data_link="https://example.com/photo.jpg",
        )

        class _FakeResponse:
            def __init__(self):
                self.status = 200
                self.headers = MagicMock()
                self.headers.get = MagicMock(return_value="image/jpeg")

            def raise_for_status(self):
                return None

            async def read(self):
                return b"\xff\xd8\xff"

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                pass

        class _FakeGetContext:
            async def __aenter__(self):
                return _FakeResponse()

            async def __aexit__(self, *_args):
                pass

        class _FakeSession:
            def get(self, *_args, **_kwargs):
                return _FakeGetContext()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                pass

        with patch("app.connectors.sources.notion.connector.aiohttp.ClientSession", _FakeSession):
            await conn._convert_image_blocks_to_base64([block], None)
        assert block.format == DataFormat.BASE64
        assert "uri" in block.data


class TestAttachmentTraversal:
    @pytest.mark.asyncio
    async def test_image_block_with_children_recurses(self):
        conn = _connector()
        ds = MagicMock()
        ds.retrieve_block_children = AsyncMock(side_effect=[
            _api_resp(True, {
                "results": [
                    {"id": "img-1", "type": "image", "has_children": True, "image": {"type": "external", "external": {"url": "https://x.com/i.png"}}},
                ],
                "has_more": False,
            }),
            _api_resp(True, {"results": [], "has_more": False}),
        ])
        conn._get_fresh_datasource = AsyncMock(return_value=ds)
        attachments, block_ids = await conn._fetch_attachment_blocks_and_block_ids_recursive("page-1")
        assert "img-1" in block_ids
        assert attachments == []


class TestGetRecordChildByExternalId:
    @pytest.mark.asyncio
    async def test_creates_temporary_row_record(self):
        conn = _connector()
        existing = MagicMock()
        existing.id = "rec-existing"
        existing.record_name = "Existing"
        conn.get_record_by_external_id = AsyncMock(return_value=None)
        conn.resolve_page_title_by_id = AsyncMock(return_value="Row Title")
        child = await conn.get_record_child_by_external_id("row-page", parent_data_source_id="ds-1")
        assert child is not None
        conn.data_entities_processor.on_new_records.assert_awaited()

    @pytest.mark.asyncio
    async def test_relation_without_parent_returns_title_only(self):
        conn = _connector()
        conn.get_record_by_external_id = AsyncMock(return_value=None)
        conn.resolve_page_title_by_id = AsyncMock(return_value="Related Page")
        child = await conn.get_record_child_by_external_id("page-rel")
        assert child.child_name == "Related Page"


class TestTransformUserError:
    def test_transform_user_exception(self):
        conn = _connector()
        bad = MagicMock()
        bad.get = MagicMock(side_effect=RuntimeError("bad"))
        assert conn._transform_to_app_user(bad) is None


class TestSyncObjectsIndexingDisabled:
    @pytest.mark.asyncio
    async def test_data_source_indexing_disabled(self):
        conn = _connector()
        conn.indexing_filters = MagicMock()
        conn.indexing_filters.is_enabled = MagicMock(
            side_effect=lambda k: k != IndexingFilterKey.DATABASES
        )
        conn.pages_sync_point = MagicMock()
        conn.pages_sync_point.read_sync_point = AsyncMock(return_value=None)
        conn.pages_sync_point.update_sync_point = AsyncMock()
        ds = MagicMock()
        ds.search = AsyncMock(return_value=_api_resp(True, {
            "results": [{
                "id": "ds-1",
                "last_edited_time": "2025-06-15T00:00:00.000Z",
                "title": [{"plain_text": "DS"}],
            }],
            "has_more": False,
        }))
        conn._get_fresh_datasource = AsyncMock(return_value=ds)
        synced_records = []

        async def capture_transform(obj, obj_type, **kwargs):
            rec = _webpage(record_type=RecordType.DATASOURCE, external_record_id=obj["id"])
            synced_records.append(rec)
            return rec

        conn._transform_to_webpage_record = capture_transform
        conn.data_entities_processor.on_new_records = AsyncMock()
        await conn._sync_objects_by_type("data_source")
        assert synced_records[0].indexing_status == ProgressStatus.AUTO_INDEX_OFF.value


class TestResolveTableRowChildrenFailures:
    @pytest.mark.asyncio
    async def test_api_failure_returns_empty_children(self):
        conn = _connector()
        ds = MagicMock()
        ds.retrieve_block_children = AsyncMock(return_value=_api_resp(False, error="fail"))
        conn._get_fresh_datasource = AsyncMock(return_value=ds)
        blocks = [Block(
            id="r1", index=0, type=BlockType.TABLE_ROW, format=DataFormat.JSON,
            data={}, source_id="row-page-1",
        )]
        await conn._resolve_table_row_children(blocks)
        assert blocks[0].table_row_metadata is None

    @pytest.mark.asyncio
    async def test_archived_child_page_skipped(self):
        conn = _connector()
        ds = MagicMock()
        ds.retrieve_block_children = AsyncMock(return_value=_api_resp(True, {
            "results": [
                {"type": "child_page", "id": "cp-1", "archived": True, "child_page": {"title": "Archived"}},
            ],
        }))
        conn._get_fresh_datasource = AsyncMock(return_value=ds)
        conn._batch_get_or_create_child_records = AsyncMock(return_value={})
        blocks = [Block(
            id="r1", index=0, type=BlockType.TABLE_ROW, format=DataFormat.JSON,
            data={}, source_id="row-page-1",
        )]
        await conn._resolve_table_row_children(blocks)
        conn._batch_get_or_create_child_records.assert_not_awaited()
