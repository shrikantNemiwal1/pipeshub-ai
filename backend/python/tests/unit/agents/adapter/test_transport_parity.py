"""LangChain vs direct-SDK transport: same input, same events out.

The direct transport exists to remove LangChain from the streaming path without
changing what the agent loop sees. Individual unit tests check each transport in
isolation; this one drives both with the same logical response and asserts the
emitted event stream is equivalent, which is the property that actually matters.

The two SDKs shape their chunks differently, so the fixtures differ by
construction -- what is compared is the events, not the inputs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessageChunk

from app.agent_loop_lib.core.messages import UserMessage
from app.agent_loop_lib.core.streaming import (
    StreamCompleteEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
)
from app.agent_loop_lib.transport.azure_openai import AzureOpenAITransport
from app.agents.agent_loop.langchain_transport import LangChainTransport

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class _FakeLangChainModel:
    def __init__(self, chunks: list[AIMessageChunk]) -> None:
        self._chunks = chunks

    def bind_tools(self, tools: Any) -> "_FakeLangChainModel":
        return self

    async def astream(self, messages: list, config: Any = None) -> "AsyncIterator[AIMessageChunk]":
        for c in self._chunks:
            yield c


def _openai_chunk(content=None, tool_calls=None, finish=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content=content, tool_calls=tool_calls),
            finish_reason=finish,
        )],
        usage=None,
    )


def _openai_tc(index, id=None, name=None, args=None):
    return SimpleNamespace(index=index, id=id,
                           function=SimpleNamespace(name=name, arguments=args))


def _azure_transport(chunks: list) -> AzureOpenAITransport:
    t = AzureOpenAITransport(
        api_key="k", azure_endpoint="https://e.openai.azure.com",
        api_version="2024-10-01-preview", deployment="dep",
    )

    class _Stream:
        def __aiter__(self):
            async def gen() -> None:
                for c in chunks:
                    yield c
            return gen()

    t._client = MagicMock()
    t._client.chat.completions.create = AsyncMock(return_value=_Stream())
    return t


async def _events(transport: object) -> list:
    return [e async for e in transport.stream(messages=[UserMessage(content="hi")])]


def _shape(events: list) -> list[tuple]:
    """Comparable summary: event type plus the fields the agent loop reads."""
    out = []
    for e in events:
        if isinstance(e, TextDeltaEvent):
            out.append(("text", e.delta))
        elif isinstance(e, ToolCallDeltaEvent):
            out.append(("tool_delta", e.index, e.name, e.arguments_delta))
        elif isinstance(e, StreamCompleteEvent):
            calls = e.response.message.tool_calls or []
            out.append(("complete", tuple((c.name, tuple(sorted(c.arguments))) for c in calls)))
        else:
            out.append((type(e).__name__,))
    return out


class TestStreamParity:
    @pytest.mark.asyncio
    async def test_text_only_response(self) -> None:
        lc = LangChainTransport(
            _FakeLangChainModel([AIMessageChunk(content="Hello "), AIMessageChunk(content="world")]),
            model_name="m",
        )
        az = _azure_transport([
            _openai_chunk(content="Hello "), _openai_chunk(content="world"),
            _openai_chunk(finish="stop"),
        ])

        assert _shape(await _events(lc)) == _shape(await _events(az))

    @pytest.mark.asyncio
    async def test_tool_call_streamed_in_fragments(self) -> None:
        """The final_answer case: fragments must arrive with matching index,
        name and payload, because live answer streaming is driven off them."""
        lc = LangChainTransport(
            _FakeLangChainModel([
                AIMessageChunk(content="", tool_call_chunks=[
                    {"name": "final_answer", "args": '{"answer_markdown"', "id": "c1", "index": 0},
                ]),
                AIMessageChunk(content="", tool_call_chunks=[
                    {"name": None, "args": ': "done"}', "id": None, "index": 0},
                ]),
            ]),
            model_name="m",
        )
        az = _azure_transport([
            _openai_chunk(tool_calls=[_openai_tc(0, id="c1", name="final_answer",
                                                 args='{"answer_markdown"')]),
            _openai_chunk(tool_calls=[_openai_tc(0, args=': "done"}')]),
            _openai_chunk(finish="tool_calls"),
        ])

        lc_shape, az_shape = _shape(await _events(lc)), _shape(await _events(az))

        # both must stream two fragments for index 0 and assemble one call
        assert [s for s in lc_shape if s[0] == "tool_delta"] == [
            s for s in az_shape if s[0] == "tool_delta"
        ]
        assert lc_shape[-1] == az_shape[-1] == (
            "complete", (("final_answer", ("answer_markdown",)),)
        )

    @pytest.mark.asyncio
    async def test_both_end_with_exactly_one_complete_event(self) -> None:
        lc = LangChainTransport(_FakeLangChainModel([AIMessageChunk(content="x")]), model_name="m")
        az = _azure_transport([_openai_chunk(content="x"), _openai_chunk(finish="stop")])

        for events in (await _events(lc), await _events(az)):
            completes = [e for e in events if isinstance(e, StreamCompleteEvent)]
            assert len(completes) == 1
            assert isinstance(events[-1], StreamCompleteEvent)
