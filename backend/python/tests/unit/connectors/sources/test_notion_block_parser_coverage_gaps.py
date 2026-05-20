"""Targeted coverage tests for NotionBlockParser gaps (95% goal)."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.connectors.sources.notion.block_parser import NotionBlockParser
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
    GroupType,
    GroupSubType,
    ListMetadata,
    TableMetadata,
    TableRowMetadata,
)


def _parser():
    return NotionBlockParser(logger=MagicMock())


def _notion_block(block_type, type_data, block_id="blk-1", **kwargs):
    block = {
        "id": block_id,
        "type": block_type,
        block_type: type_data,
        "has_children": kwargs.get("has_children", False),
        "archived": kwargs.get("archived", False),
        "in_trash": kwargs.get("in_trash", False),
        "created_time": "2025-01-01T00:00:00Z",
        "last_edited_time": "2025-01-15T00:00:00Z",
    }
    return block


class TestExtractMediaFileUrl:
    def test_external_url(self):
        data = {"type": "external", "external": {"url": "https://example.com/f.pdf"}}
        assert NotionBlockParser._extract_media_file_url(data) == "https://example.com/f.pdf"

    def test_notion_hosted_file_url(self):
        data = {"type": "file", "file": {"url": "https://notion.so/f.pdf"}}
        assert NotionBlockParser._extract_media_file_url(data) == "https://notion.so/f.pdf"

    def test_external_not_dict_returns_none(self):
        data = {"type": "external", "external": "bad"}
        assert NotionBlockParser._extract_media_file_url(data) is None

    def test_file_not_dict_returns_none(self):
        data = {"file": "bad"}
        assert NotionBlockParser._extract_media_file_url(data) is None

    def test_missing_keys_returns_none(self):
        assert NotionBlockParser._extract_media_file_url({}) is None


class TestParseBlockControlFlow:
    @pytest.mark.asyncio
    async def test_parser_returns_three_tuple(self):
        parser = _parser()

        async def fake_parse(*_args, **_kwargs):
            return (None, None, [])

        parser._parse_paragraph = fake_parse
        block = _notion_block("paragraph", {"rich_text": [{"plain_text": "hi"}]})
        result = await parser.parse_block(block, block_index=0)
        assert result == (None, None, [])

    @pytest.mark.asyncio
    async def test_skips_block_with_empty_data(self):
        parser = _parser()
        block = _notion_block("paragraph", {"rich_text": []})
        blk, grp, _ = await parser.parse_block(block, block_index=0)
        assert blk is None and grp is None

    @pytest.mark.asyncio
    async def test_skips_block_group_with_empty_data(self):
        parser = _parser()

        async def empty_group(*_args, **_kwargs):
            return BlockGroup(
                index=0,
                type=GroupType.TEXT_SECTION,
                sub_type=GroupSubType.SYNCED_BLOCK,
                data="",
            )

        parser._parse_column_list = empty_group
        block = _notion_block("column_list", {})
        blk, grp, _ = await parser.parse_block(block, block_index=0)
        assert blk is None and grp is None

    @pytest.mark.asyncio
    async def test_parser_exception_returns_none(self):
        parser = _parser()

        async def boom(*_args, **_kwargs):
            raise ValueError("parse failed")

        parser._parse_paragraph = boom
        block = _notion_block("paragraph", {"rich_text": [{"plain_text": "x"}]})
        blk, grp, _ = await parser.parse_block(block, block_index=0)
        assert blk is None and grp is None
        parser.logger.warning.assert_called()


class TestParagraphLinkMention:
    @pytest.mark.asyncio
    async def test_single_link_mention_becomes_link_block(self):
        parser = _parser()
        block = _notion_block("paragraph", {
            "rich_text": [{
                "type": "mention",
                "plain_text": "Example",
                "href": "https://example.com",
                "mention": {
                    "type": "link_mention",
                    "link_mention": {
                        "href": "https://example.com",
                        "title": "Example Site",
                    },
                },
            }],
        })
        blk, _, _ = await parser.parse_block(block, block_index=0)
        assert blk is not None
        assert blk.sub_type == BlockSubType.LINK
        assert blk.link_metadata is not None
        assert str(blk.link_metadata.link_url).startswith("https://example.com")

    @pytest.mark.asyncio
    async def test_link_mention_without_href(self):
        parser = _parser()
        block = _notion_block("paragraph", {
            "rich_text": [{
                "type": "mention",
                "plain_text": "No URL",
                "mention": {
                    "type": "link_mention",
                    "link_mention": {"title": "No URL"},
                },
            }],
        })
        blk, _, _ = await parser.parse_block(block, block_index=0)
        assert blk.sub_type == BlockSubType.LINK
        assert blk.link_metadata is None


class TestCalloutIconTypes:
    @pytest.mark.asyncio
    async def test_callout_external_icon(self):
        parser = _parser()
        block = _notion_block("callout", {
            "rich_text": [{"plain_text": "Note"}],
            "icon": {"type": "external", "external": {"url": "https://icon.example/icon.png"}},
        })
        blk, _, _ = await parser.parse_block(block, block_index=0)
        assert blk is not None
        assert blk.link_metadata is not None

    @pytest.mark.asyncio
    async def test_callout_file_icon(self):
        parser = _parser()
        block = _notion_block("callout", {
            "rich_text": [{"plain_text": "Note"}],
            "icon": {"type": "file", "file": {"url": "https://notion.so/icon.png"}},
        })
        blk, _, _ = await parser.parse_block(block, block_index=0)
        assert blk.link_metadata is not None


class TestMarkdownRichTextEdges:
    def test_equation_in_markdown(self):
        parser = _parser()
        result = parser._extract_markdown_text([{
            "type": "equation",
            "equation": {"expression": "E=mc^2"},
        }])
        assert "$$E=mc^2$$" in result

    def test_mention_link_mention_href_only(self):
        parser = _parser()
        result = parser._extract_markdown_text([{
            "type": "mention",
            "plain_text": "link",
            "mention": {
                "type": "link_mention",
                "link_mention": {"href": "https://x.com"},
            },
        }])
        assert "https://x.com" in result

    def test_mention_plain_text_only(self):
        parser = _parser()
        result = parser._extract_markdown_text([{
            "type": "mention",
            "plain_text": "User Name",
            "mention": {"type": "user", "user": {"id": "u1"}},
        }])
        assert result == "User Name"

    def test_unknown_rich_text_type_fallback(self):
        parser = _parser()
        result = parser._extract_markdown_text([{
            "type": "custom",
            "plain_text": "fallback",
        }])
        assert result == "fallback"

    def test_code_with_consecutive_backticks_uses_long_fence(self):
        parser = _parser()
        result = parser._extract_markdown_text([{
            "type": "text",
            "plain_text": "``code``",
            "annotations": {"code": True},
        }])
        assert "```" in result or "````" in result


class TestSyncedBlockReferenceMetadata:
    @pytest.mark.asyncio
    async def test_synced_block_reference_stores_original_id(self):
        parser = _parser()
        block = _notion_block("synced_block", {
            "synced_from": {"type": "block_id", "block_id": "orig-block"},
        })
        _, grp, _ = await parser.parse_block(block, block_index=0)
        assert grp is not None
        assert grp.data["is_reference"] is True
        assert grp.data["original_block_id"] == "orig-block"


class TestExtractRelationsAndPeopleExtended:
    def test_created_by_and_last_edited_by(self):
        parser = _parser()
        props = {
            "cb": {"type": "created_by", "created_by": {"id": "user-cb"}},
            "leb": {"type": "last_edited_by", "last_edited_by": {"id": "user-leb"}},
        }
        relations, people = parser._extract_relations_and_people(props)
        assert relations == []
        assert "user-cb" in people
        assert "user-leb" in people

    def test_rollup_array_people_and_relation(self):
        parser = _parser()
        props = {
            "roll": {
                "type": "rollup",
                "rollup": {
                    "type": "array",
                    "array": [
                        {"type": "people", "people": [{"id": "u1"}]},
                        {"type": "relation", "id": "rel-page-1"},
                    ],
                },
            },
        }
        relations, people = parser._extract_relations_and_people(props)
        assert "rel-page-1" in relations
        assert "u1" in people

    def test_non_dict_property_skipped(self):
        parser = _parser()
        relations, people = parser._extract_relations_and_people({"bad": "x"})
        assert relations == [] and people == []


class TestCreateDataRowBlocks:
    @pytest.mark.asyncio
    async def test_row_with_title_and_child_callbacks(self):
        parser = _parser()
        table_group = BlockGroup(
            index=0,
            type=GroupType.TABLE,
            children=BlockGroupChildren(),
            table_metadata=TableMetadata(num_of_cols=1, num_of_rows=2, has_header=True),
        )
        blocks = []
        rows = [{
            "id": "row-page-1",
            "url": "https://notion.so/row-1",
            "created_time": "2025-01-01T00:00:00Z",
            "last_edited_time": "2025-01-02T00:00:00Z",
            "properties": {
                "Name": {"type": "title", "title": [{"plain_text": "Row Title"}]},
            },
        }]
        child = ChildRecord(child_type=ChildType.RECORD, child_id="rec-1", child_name="Row Title")

        async def record_cb(page_id):
            if page_id == "row-page-1":
                return child
            if page_id == "rel-1":
                return ChildRecord(child_type=ChildType.RECORD, child_id="rec-2", child_name="Rel")
            return None

        await parser._create_data_row_blocks(
            blocks,
            table_group,
            rows,
            [["Row Title"]],
            None,
            "\u200B|\u200B",
            get_record_child_callback=record_cb,
            relations_and_people_list=[(["rel-1"], [])],
        )
        assert len(blocks) == 1
        assert blocks[0].table_row_metadata.children_records

    @pytest.mark.asyncio
    async def test_row_callback_exceptions_still_create_block(self):
        parser = _parser()
        table_group = BlockGroup(
            index=0,
            type=GroupType.TABLE,
            children=BlockGroupChildren(),
        )
        blocks = []

        async def failing_cb(_page_id):
            raise RuntimeError("lookup failed")

        await parser._create_data_row_blocks(
            blocks,
            table_group,
            [{"id": "row-1", "properties": {}}],
            [["cell"]],
            ["desc"],
            "|",
            get_record_child_callback=failing_cb,
            relations_and_people_list=[(["rel-bad"], [])],
        )
        assert len(blocks) == 1
        assert blocks[0].table_row_metadata is None or not blocks[0].table_row_metadata.children_records


class TestPropertyValueWithResolutionExtended:
    @pytest.mark.asyncio
    async def test_created_by_with_callback(self):
        parser = _parser()
        user_cb = AsyncMock(return_value=ChildRecord(
            child_type=ChildType.USER, child_id="u1", child_name="Alice"
        ))
        prop = {"type": "created_by", "created_by": {"id": "u1"}}
        result = await parser._extract_property_value_with_resolution(
            prop, get_user_child_callback=user_cb
        )
        assert result == "Alice"

    @pytest.mark.asyncio
    async def test_created_by_no_user_id(self):
        parser = _parser()
        prop = {"type": "created_by", "created_by": {}}
        assert await parser._extract_property_value_with_resolution(prop) == ""

    @pytest.mark.asyncio
    async def test_last_edited_by_fallback_name(self):
        parser = _parser()
        prop = {"type": "last_edited_by", "last_edited_by": {"id": "u2", "name": "Bob"}}
        result = await parser._extract_property_value_with_resolution(prop)
        assert result == "Bob"

    @pytest.mark.asyncio
    async def test_rollup_people_and_relation(self):
        parser = _parser()
        user_cb = AsyncMock(return_value=ChildRecord(
            child_type=ChildType.USER, child_id="u1", child_name="User One"
        ))
        record_cb = AsyncMock(return_value=ChildRecord(
            child_type=ChildType.RECORD, child_id="p1", child_name="Linked Page"
        ))
        prop = {
            "type": "rollup",
            "rollup": {
                "type": "array",
                "array": [
                    {"type": "people", "people": [{"id": "u1"}]},
                    {"type": "relation", "id": "p1"},
                ],
            },
        }
        result = await parser._extract_property_value_with_resolution(
            prop, get_record_child_callback=record_cb, get_user_child_callback=user_cb
        )
        assert "User One" in result
        assert "Linked Page" in result


class TestExtractPropertyValueException:
    def test_unknown_property_type(self):
        parser = _parser()
        prop = {"type": "custom_unknown", "custom_unknown": {"foo": "bar"}}
        result = parser._extract_property_value(prop)
        assert "foo" in result or "bar" in result

    def test_exception_returns_empty(self):
        parser = _parser()
        prop = {"type": "files", "files": None}
        result = parser._extract_property_value(prop)
        assert result == ""


class TestConvertIndicesToBlockGroupChildren:
    def test_none_and_empty(self):
        parser = _parser()
        assert parser._convert_indices_to_block_group_children(None) is None
        assert parser._convert_indices_to_block_group_children([]) is None

    def test_mixed_indices(self):
        parser = _parser()
        indices = [
            BlockContainerIndex(block_index=0),
            BlockContainerIndex(block_group_index=1),
        ]
        children = parser._convert_indices_to_block_group_children(indices)
        assert children is not None


class TestListGroupingTrailingGroup:
    def test_closes_list_group_at_end_of_blocks(self):
        parser = _parser()
        blocks = [
            Block(
                id="b0", index=0, type=BlockType.TEXT, format=DataFormat.TXT, data="1. one",
                list_metadata=ListMetadata(list_style="numbered", indent_level=0),
            ),
            Block(
                id="b1", index=1, type=BlockType.TEXT, format=DataFormat.TXT, data="tail",
            ),
        ]
        groups = []
        parser._group_list_items(blocks, groups)
        assert any(g.type == GroupType.ORDERED_LIST for g in groups)


class TestPostProcessTableHeaderFlag:
    def test_first_row_marked_header_when_table_has_header(self):
        parser = _parser()
        table_group = BlockGroup(
            index=0,
            type=GroupType.TABLE,
            table_metadata=TableMetadata(has_header=True, num_of_cols=2, num_of_rows=2),
        )
        row = Block(
            id="r1", index=1, parent_index=0, type=BlockType.TABLE_ROW,
            format=DataFormat.JSON, data={},
            table_row_metadata=TableRowMetadata(row_number=0, is_header=False),
        )
        parser.post_process_blocks([row], [table_group])
        assert row.table_row_metadata.is_header is True


class TestCommentParsing:
    def test_parse_notion_timestamp_invalid(self):
        assert NotionBlockParser.parse_notion_timestamp("not-a-date") is None
        assert NotionBlockParser.parse_notion_timestamp(None) is None

    def test_parse_comment_missing_id(self):
        parser = _parser()
        assert parser.parse_notion_comment_to_block_comment({}) is None

    def test_parse_comment_success(self):
        parser = _parser()
        comment = parser.parse_notion_comment_to_block_comment(
            {
                "id": "c1",
                "rich_text": [{"plain_text": "Hello"}],
                "created_by": {"id": "u1"},
                "created_time": "2025-01-01T00:00:00.000Z",
                "last_edited_time": "2025-01-02T00:00:00.000Z",
                "discussion_id": "d1",
            },
            author_name="Author",
            comment_attachments=[
                CommentAttachment(name="file.pdf", id="att-1"),
            ],
        )
        assert comment is not None
        assert comment.text == "Hello"
        assert comment.author_name == "Author"

    def test_parse_comment_exception(self):
        parser = _parser()
        bad = MagicMock()
        bad.get = MagicMock(side_effect=RuntimeError("fail"))
        assert parser.parse_notion_comment_to_block_comment(bad) is None

    def test_create_comment_group_and_thread(self):
        parser = _parser()
        bc = BlockComment(
            text="Hi",
            format=DataFormat.TXT,
            author_id="u1",
            author_name="Author",
            created_at=datetime(2025, 1, 1),
        )
        group = parser.create_comment_group(
            bc, group_index=0, parent_group_index=1, source_id="c1",
            attachment_block_indices=[BlockContainerIndex(block_index=0)],
        )
        assert group.sub_type == GroupSubType.COMMENT
        assert group.children is not None

        thread = parser.create_comment_thread_group(
            "d1", group_index=1,
            comment_group_indices=[BlockContainerIndex(block_group_index=0)],
        )
        assert thread.sub_type == GroupSubType.COMMENT_THREAD
