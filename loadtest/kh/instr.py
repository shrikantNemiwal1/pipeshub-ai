"""Per-call Neo4j instrumentation, attributed across a concurrent fan-out.

`loadtest/instr/backend_timing.py:100` already patches `AsyncSession.run`, but it
aggregates into 5-second windows and cannot say *which* connector a call belongs
to. A KH global flatten fires one access query and N connector queries at once,
so attribution is the whole point here — hence a contextvar the harness sets
around each provider call.

It also patches `AsyncResult.data`, because `Neo4jClient._run_autocommit`
(neo4j_client.py:599-601) returns `await result.data()` and never calls
`consume()`: the ResultSummary — and with it `result_available_after` /
`result_consumed_after`, the only split between server execution and row
streaming — is discarded before the provider ever sees it. Consuming here is
safe: on an exhausted result the driver returns the summary it already holds.
"""

from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass, field

_span: contextvars.ContextVar[tuple[str, str, str] | None] = contextvars.ContextVar(
    "kh_span", default=None
)


@dataclass
class Event:
    request_id: str
    phase: str            # "access" | "connector"
    app_id: str
    wall_run_ms: float
    wall_data_ms: float = 0.0
    server_available_ms: float | None = None
    server_consumed_ms: float | None = None
    rows: int = 0

    @property
    def wall_ms(self) -> float:
        return self.wall_run_ms + self.wall_data_ms

    @property
    def server_ms(self) -> float | None:
        if self.server_available_ms is None or self.server_consumed_ms is None:
            return None
        return self.server_available_ms + self.server_consumed_ms

    @property
    def py_ms(self) -> float | None:
        """Client-side + network share: everything the server did not report."""
        server = self.server_ms
        return None if server is None else self.wall_ms - server


@dataclass
class Collector:
    events: list[Event] = field(default_factory=list)
    enabled: bool = True

    def reset(self) -> None:
        self.events = []

    def for_request(self, request_id: str) -> list[Event]:
        return [e for e in self.events if e.request_id == request_id]


COLLECTOR = Collector()
_patched = False


def span(request_id: str, phase: str, app_id: str = ""):
    """Context manager binding every driver call inside it to one logical step."""

    class _Span:
        def __enter__(self):
            self._token = _span.set((request_id, phase, app_id))
            return self

        def __exit__(self, *exc):
            _span.reset(self._token)
            return False

    return _Span()


def patch() -> None:
    global _patched
    if _patched:
        return
    from neo4j import AsyncResult, AsyncSession

    orig_run = AsyncSession.run
    orig_data = AsyncResult.data

    async def timed_run(self, *a, **k):  # noqa: ANN001
        t0 = time.perf_counter()
        result = await orig_run(self, *a, **k)
        elapsed = (time.perf_counter() - t0) * 1000.0
        current = _span.get()
        if COLLECTOR.enabled and current is not None:
            request_id, phase, app_id = current
            event = Event(request_id, phase, app_id, wall_run_ms=elapsed)
            COLLECTOR.events.append(event)
            # The result carries its own event so `data` can find it without
            # guessing: several results are open at once during a fan-out.
            try:
                result._kh_event = event
            except Exception:
                pass
        return result

    async def timed_data(self, *a, **k):  # noqa: ANN001
        t0 = time.perf_counter()
        rows = await orig_data(self, *a, **k)
        elapsed = (time.perf_counter() - t0) * 1000.0
        event = getattr(self, "_kh_event", None)
        if event is not None:
            event.wall_data_ms = elapsed
            event.rows = len(rows) if rows is not None else 0
            try:
                summary = await self.consume()
                event.server_available_ms = summary.result_available_after
                event.server_consumed_ms = summary.result_consumed_after
            except Exception:
                pass
        return rows

    AsyncSession.run = timed_run
    AsyncResult.data = timed_data
    _patched = True
