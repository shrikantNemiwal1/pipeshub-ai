"""Azure OpenAI through the official SDK, without LangChain.

`LangChainTransport` builds an `AIMessageChunk` per streamed token and pydantic
runs two model validators on each one; that measured 8.96% of query-service CPU
under load, and there is no upstream fix (langchain-core 1.5.3 still carries the
validators, marked with its own "TODO: remove this logic if possible").

Everything except the client construction is inherited from `OpenAITransport` --
Azure speaks the same Chat Completions shape, so `_format_messages`,
`_format_tools`, `complete`, `complete_structured` and the `stream()`
accumulation loop are identical. Only the client differs, plus the two
single-retry fallbacks `LangChainTransport` carries for request-shape conflicts.

Registered as provider "azure_direct"; "langchain" stays the default so this can
be switched per deployment and reverted without a code change.

Not covered here, deliberately: Opik tracing (which hooks LangChain callbacks),
and every non-Azure provider -- those keep going through LangChain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.agent_loop_lib.core.streaming import StreamEvent
from app.agent_loop_lib.transport.openai import OpenAITransport
from app.agent_loop_lib.transport.provider_conflicts import (
    is_api_shape_conflict,
    is_reasoning_mandatory_conflict,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from app.agent_loop_lib.core.messages import Message
    from app.agent_loop_lib.core.tool_schema import ToolSchema
    from app.agent_loop_lib.core.types import ModelResponse

# Reasoning values a provider may reject outright when it refuses to let
# reasoning be disabled. Matches LangChainTransport._reasoning_mandatory_fallback,
# which bumps off a disabled value rather than clearing the field.
_DISABLED_REASONING_VALUES = ("none", "off", "disabled", "minimal")
_REASONING_BUMP_TARGET = "low"


class AzureOpenAITransport(OpenAITransport):
    """Azure OpenAI via AsyncAzureOpenAI.

    `deployment` is Azure's routing key: the model name in a request is the
    deployment name, not the underlying model. Callers pass whatever
    `aimodels.py` already resolved from configuration.
    """

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
        deployment: str,
        model: str | None = None,
    ) -> None:
        # Skip OpenAITransport.__init__ (it builds AsyncOpenAI) but keep the
        # LLMTransport base contract and the cumulative counters its callers read.
        super(OpenAITransport, self).__init__()
        try:
            import openai as _openai
        except ImportError as exc:
            raise ImportError(
                "openai SDK is required for AzureOpenAITransport. "
                "Install it with: pip install 'agent-loop[openai]'"
            ) from exc
        self._openai = _openai
        self._deployment = deployment
        # Azure routes on the deployment name; the model field of a request must
        # carry it, so default the model to the deployment rather than a public
        # model id.
        self._model = model or deployment
        self._client = _openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            azure_deployment=deployment,
        )
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_llm_calls: int = 0
        self.total_cache_read_tokens: int = 0
        self.total_cache_write_tokens: int = 0

    @classmethod
    def from_langchain_model(cls, llm: Any, model_name: str = "") -> "AzureOpenAITransport":
        """Build from an already-configured `AzureChatOpenAI`.

        The credentials are read off the model the LangChain path already uses,
        rather than re-resolving them from configuration, so the two transports
        cannot end up pointed at different deployments.

        Raises ValueError when `llm` is not Azure-shaped -- this transport is
        Azure-only, and silently falling back would hide a misconfiguration
        behind a slower path.
        """
        def _val(name: str) -> str:
            raw = getattr(llm, name, None)
            # Credentials arrive as pydantic SecretStr on the LangChain model.
            secret = getattr(raw, "get_secret_value", None)
            return (secret() if callable(secret) else raw) or ""

        endpoint = _val("azure_endpoint")
        deployment = _val("deployment_name")
        api_key = _val("openai_api_key")
        api_version = _val("openai_api_version")
        missing = [
            n for n, v in (
                ("azure_endpoint", endpoint), ("deployment_name", deployment),
                ("openai_api_key", api_key), ("openai_api_version", api_version),
            ) if not v
        ]
        if missing:
            raise ValueError(
                f"{type(llm).__name__} is missing Azure settings {missing}; "
                "azure_direct only supports AzureChatOpenAI-configured models"
            )
        return cls(
            api_key=api_key, azure_endpoint=endpoint, api_version=api_version,
            deployment=deployment, model=model_name or deployment,
        )

    @property
    def provider(self) -> str:
        return "azure_direct"

    # -- request-shape recovery -------------------------------------------
    #
    # LangChainTransport retries once against a differently-configured model
    # copy. There is no model object here, so the equivalent is retrying the
    # same call with the offending parameter adjusted.

    def _retry_kwargs(self, exc: Exception, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        """Kwargs for a single retry of a request the provider rejected on shape.

        Returns None when the error is not one of the known conflicts, or when
        there is nothing to change -- a marker match with no adjustment to make
        means the 400 has some other cause and retrying would just repeat it.
        """
        if is_api_shape_conflict(exc):
            # The conflict is reasoning + bound tools going through a shape the
            # deployment does not accept for that combination. Dropping the
            # reasoning hint is the one adjustment available on this API, and it
            # keeps the tools -- a turn that needs a tool must still get one.
            if kwargs.get("reasoning_effort") is not None:
                retry = dict(kwargs)
                retry.pop("reasoning_effort", None)
                return retry
            return None
        if is_reasoning_mandatory_conflict(exc):
            effort = kwargs.get("reasoning_effort")
            if effort is None or str(effort).lower() in _DISABLED_REASONING_VALUES:
                retry = dict(kwargs)
                retry["reasoning_effort"] = _REASONING_BUMP_TARGET
                return retry
            return None
        return None

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        system: str | None = None,
        model: str | None = None,
        thinking_budget: int | None = None,
        effort: str | None = None,
        system_blocks: list[str] | None = None,
    ) -> ModelResponse:
        try:
            return await super().complete(
                messages=messages, tools=tools, system=system, model=model,
                thinking_budget=thinking_budget, effort=effort,
                system_blocks=system_blocks,
            )
        except Exception as exc:
            if effort is not None and self._retry_kwargs(exc, {"reasoning_effort": effort}):
                retry = self._retry_kwargs(exc, {"reasoning_effort": effort}) or {}
                return await super().complete(
                    messages=messages, tools=tools, system=system, model=model,
                    thinking_budget=thinking_budget,
                    effort=retry.get("reasoning_effort"),
                    system_blocks=system_blocks,
                )
            raise

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        system: str | None = None,
        model: str | None = None,
        thinking_budget: int | None = None,
        effort: str | None = None,
        system_blocks: list[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream, retrying once on a request-shape conflict.

        The retry only happens if it fires before anything was yielded --
        re-opening a stream that already emitted deltas would replay them to the
        client.
        """
        emitted = False
        try:
            async for event in super().stream(
                messages=messages, tools=tools, system=system, model=model,
                thinking_budget=thinking_budget, effort=effort,
                system_blocks=system_blocks,
            ):
                emitted = True
                yield event
            return
        except Exception as exc:
            retry = self._retry_kwargs(exc, {"reasoning_effort": effort})
            if emitted or retry is None:
                raise
            retry_effort = retry.get("reasoning_effort")

        async for event in super().stream(
            messages=messages, tools=tools, system=system, model=model,
            thinking_budget=thinking_budget, effort=retry_effort,
            system_blocks=system_blocks,
        ):
            yield event
