"""Azure direct transport: behaviour must match the LangChain path it replaces.

The point of this transport is to drop LangChain from the streaming path without
changing what the agent loop sees, so most of these assert equivalence rather
than correctness in isolation.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from app.agent_loop_lib.core.messages import UserMessage
from app.agent_loop_lib.core.streaming import (
    StreamCompleteEvent,
    ToolCallDeltaEvent,
)
from app.agent_loop_lib.transport.azure_openai import AzureOpenAITransport

if TYPE_CHECKING:
    from collections.abc import Iterator


def _transport() -> AzureOpenAITransport:
    return AzureOpenAITransport(
        api_key="k",
        azure_endpoint="https://example.openai.azure.com",
        api_version="2024-10-01-preview",
        deployment="gpt-5-4-mini",
    )


def _chunk(content=None, reasoning=None, tool_calls=None, finish=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    if reasoning is not None:
        delta.reasoning_content = reasoning
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=None
    )


def _tc(index: int, id=None, name=None, args=None):
    return SimpleNamespace(
        index=index, id=id, function=SimpleNamespace(name=name, arguments=args)
    )


def _scripted(chunks: list):
    class _Stream:
        def __aiter__(self):
            async def gen() -> "Iterator":
                for c in chunks:
                    yield c
            return gen()
    return _Stream()


def _wire(transport: AzureOpenAITransport, chunks: list) -> MagicMock:
    transport._client = MagicMock()
    transport._client.chat.completions.create = AsyncMock(return_value=_scripted(chunks))
    return transport._client


class TestClientConstruction:
    def test_uses_the_azure_client_not_the_plain_one(self) -> None:
        t = _transport()
        assert type(t._client).__name__ == "AsyncAzureOpenAI"
        assert t.provider == "azure_direct"

    def test_model_defaults_to_the_deployment(self) -> None:
        """Azure routes on deployment name, so a public model id would 404."""
        assert _transport().model_name == "gpt-5-4-mini"

    def test_built_from_the_langchain_model_the_other_transport_uses(self) -> None:
        llm = MagicMock()
        llm.azure_endpoint = "https://example.openai.azure.com"
        llm.deployment_name = "gpt-5-4-mini"
        llm.openai_api_key = SecretStr("sk-secret")
        llm.openai_api_version = "2024-10-01-preview"

        t = AzureOpenAITransport.from_langchain_model(llm, model_name="gpt-5.4-mini")

        assert t.provider == "azure_direct"
        assert t.model_name == "gpt-5.4-mini"

    def test_non_azure_model_is_rejected_loudly(self) -> None:
        """Silently falling back would hide a misconfiguration behind LangChain."""
        llm = MagicMock()
        llm.azure_endpoint = None
        llm.deployment_name = None
        llm.openai_api_key = None
        llm.openai_api_version = None
        with pytest.raises(ValueError, match="azure_direct only supports"):
            AzureOpenAITransport.from_langchain_model(llm)


class TestStreamEvents:
    @pytest.mark.asyncio
    async def test_emits_the_same_event_types_as_the_langchain_path(self) -> None:
        t = _transport()
        _wire(t, [
            _chunk(reasoning="pondering"),
            _chunk(content="Hello"),
            _chunk(tool_calls=[_tc(0, id="call_1", name="final_answer", args='{"a"')]),
            _chunk(tool_calls=[_tc(0, args=':1}')]),
            _chunk(finish="tool_calls"),
        ])

        events = [e async for e in t.stream(messages=[UserMessage(content="hi")])]
        kinds = [type(e).__name__ for e in events]

        assert kinds == [
            "ThinkingDeltaEvent", "TextDeltaEvent",
            "ToolCallDeltaEvent", "ToolCallDeltaEvent", "StreamCompleteEvent",
        ]
        assert isinstance(events[-1], StreamCompleteEvent)

    @pytest.mark.asyncio
    async def test_tool_call_fragments_carry_the_index_live_streaming_keys_off(self) -> None:
        """agent/__init__ keys its final_answer extractor off `index`; without
        it the answer only reaches the user when the whole turn ends."""
        t = _transport()
        _wire(t, [
            _chunk(tool_calls=[_tc(0, id="c0", name="final_answer", args='{"x"')]),
            _chunk(tool_calls=[_tc(1, id="c1", name="search", args='{"q"')]),
            _chunk(tool_calls=[_tc(0, args=':1}')]),
            _chunk(finish="tool_calls"),
        ])

        deltas = [e async for e in t.stream(messages=[UserMessage(content="hi")])]
        frags = [e for e in deltas if isinstance(e, ToolCallDeltaEvent)]

        assert [f.index for f in frags] == [0, 1, 0]
        # name/id appear only on the opening fragment of each call, matching what
        # the LangChain transport passes through; the loop reads name on the
        # first delta per index only.
        assert [f.name for f in frags] == ["final_answer", "search", None]

    @pytest.mark.asyncio
    async def test_empty_fragments_are_skipped(self) -> None:
        """Matches the LangChain transport, which drops empty args deltas."""
        t = _transport()
        _wire(t, [
            _chunk(tool_calls=[_tc(0, id="c0", name="t", args=None)]),
            _chunk(tool_calls=[_tc(0, args="")]),
            _chunk(tool_calls=[_tc(0, args="{}")]),
            _chunk(finish="tool_calls"),
        ])

        events = [e async for e in t.stream(messages=[UserMessage(content="hi")])]
        assert len([e for e in events if isinstance(e, ToolCallDeltaEvent)]) == 1

    @pytest.mark.asyncio
    async def test_fragments_reassemble_into_the_final_tool_call(self) -> None:
        t = _transport()
        _wire(t, [
            _chunk(tool_calls=[_tc(0, id="c1", name="final_answer", args='{"answer_markdown"')]),
            _chunk(tool_calls=[_tc(0, args=': "done"}')]),
            _chunk(finish="tool_calls"),
        ])

        events = [e async for e in t.stream(messages=[UserMessage(content="hi")])]
        calls = events[-1].response.message.tool_calls

        assert len(calls) == 1
        assert calls[0].name == "final_answer"
        assert calls[0].arguments == {"answer_markdown": "done"}

    @pytest.mark.asyncio
    async def test_text_only_response_still_terminates_with_one_complete(self) -> None:
        t = _transport()
        _wire(t, [_chunk(content="a"), _chunk(content="b"), _chunk(finish="stop")])

        events = [e async for e in t.stream(messages=[UserMessage(content="hi")])]

        assert [type(e).__name__ for e in events] == [
            "TextDeltaEvent", "TextDeltaEvent", "StreamCompleteEvent",
        ]
        # content is normalised to parts by AssistantMessage
        assert "".join(p.text for p in events[-1].response.message.content) == "ab"


class TestRequestShapeFallbacks:
    """Ported from LangChainTransport. These exist because the failures were
    hit in production, so losing them would be a real regression."""

    @pytest.mark.asyncio
    async def test_api_shape_conflict_retries_once_without_reasoning(self) -> None:
        t = _transport()
        t._client = MagicMock()
        calls: list = []

        async def _create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("Please use /v1/responses instead")
            return _scripted([_chunk(content="ok"), _chunk(finish="stop")])

        t._client.chat.completions.create = AsyncMock(side_effect=_create)
        events = [e async for e in t.stream(messages=[UserMessage(content="hi")], effort="low")]

        assert len(calls) == 2
        assert calls[0].get("reasoning_effort") == "low"
        assert "reasoning_effort" not in calls[1]
        assert isinstance(events[-1], StreamCompleteEvent)

    @pytest.mark.asyncio
    async def test_reasoning_mandatory_conflict_bumps_off_a_disabled_value(self) -> None:
        t = _transport()
        t._client = MagicMock()
        calls: list = []

        async def _create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("Reasoning is mandatory for this endpoint")
            return _scripted([_chunk(content="ok"), _chunk(finish="stop")])

        t._client.chat.completions.create = AsyncMock(side_effect=_create)
        events = [e async for e in t.stream(messages=[UserMessage(content="hi")], effort="none")]

        assert len(calls) == 2
        assert calls[1]["reasoning_effort"] == "low"
        assert isinstance(events[-1], StreamCompleteEvent)

    @pytest.mark.asyncio
    async def test_unrelated_errors_are_not_retried(self) -> None:
        t = _transport()
        t._client = MagicMock()
        calls: list = []

        async def _create(**kwargs):
            calls.append(kwargs)
            raise RuntimeError("rate limit exceeded")

        t._client.chat.completions.create = AsyncMock(side_effect=_create)
        with pytest.raises(Exception, match="rate limit"):
            [e async for e in t.stream(messages=[UserMessage(content="hi")], effort="low")]

        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_no_retry_once_events_have_been_emitted(self) -> None:
        """Re-opening a stream mid-flight would replay deltas to the client."""
        t = _transport()
        t._client = MagicMock()
        calls: list = []

        class _Failing:
            def __aiter__(self):
                async def gen():
                    yield _chunk(content="partial")
                    raise RuntimeError("Please use /v1/responses instead")
                return gen()

        async def _create(**kwargs):
            calls.append(kwargs)
            return _Failing()

        t._client.chat.completions.create = AsyncMock(side_effect=_create)
        seen = []
        with pytest.raises(Exception, match="responses"):
            async for e in t.stream(messages=[UserMessage(content="hi")], effort="low"):
                seen.append(e)

        assert len(calls) == 1
        assert [type(e).__name__ for e in seen] == ["TextDeltaEvent"]
