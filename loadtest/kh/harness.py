"""In-process driver for one KH v3 global-flatten request, with decomposition.

Level (b) of the harness: the real fan-out, the real merge, the real
`result.data()`, but no FastAPI and no auth. It runs on a quiet event loop, so
treat its totals as an optimistic lower bound on what a user feels.

Arms are knobs this module already owns (`max_concurrency` is a parameter at
kh_search.py:69; the log level is a `setLevel`). Nothing under `backend/` is
modified to switch an arm.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import env as kh_env  # noqa: E402
import instr  # noqa: E402

LOG_FORMAT = (
    "%(asctime)s [%(levelname)s] %(name)s %(filename)s:%(lineno)d - %(message)s"
)

# The driver logs every server notification at WARNING; this query legitimately
# names properties that are absent on some labels, so the stream is constant and
# would dominate any console. app/utils/logger.py:155 does the same thing.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
logging.getLogger("neo4j").setLevel(logging.ERROR)


def build_logger(name: str, *, to_devnull: bool = True) -> logging.Logger:
    """Replicate `app/utils/logger.py:169 create_logger`'s handler shape.

    Two handlers, formatted twice per record, `propagate=False` -- that shape is
    the cost the INFO logging at neo4j_provider.py:15194/:15714 actually pays.
    The stream goes to devnull rather than a terminal, so a measured log cost is
    a LOWER bound: a real console is slower than devnull.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter(LOG_FORMAT)
        results = Path(__file__).resolve().parent / "results"
        results.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(results / (name + ".log"), encoding="utf-8")
        file_handler.setFormatter(fmt)
        stream = open(os.devnull, "w", encoding="utf-8") if to_devnull else sys.stdout
        console = logging.StreamHandler(stream)
        console.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.addHandler(console)
        logger.propagate = False
    return logger


@dataclass
class ConnectorTiming:
    app_id: str
    wall_ms: float
    server_ms: float | None
    data_ms: float
    py_ms: float | None
    rows: int
    n_granted_ids: int


@dataclass
class RequestRecord:
    request_id: str
    arm: str
    request_ms: float
    access_ms: float
    fanout_wall_ms: float
    connectors: list[ConnectorTiming] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    total: int | None = None
    counts: dict[str, int] | None = None
    n_connectors: int = 0

    @property
    def merge_ms(self) -> float:
        """The part of the fan-out window no connector query accounts for.

        `merge_pages` is synchronous and sits inside `search_page`, so with a
        perfectly parallel fan-out this is merge plus scheduling overhead.
        """
        return max(self.fanout_wall_ms - self.max_conn_ms, 0.0)

    @property
    def other_ms(self) -> float:
        return self.request_ms - self.access_ms - self.fanout_wall_ms

    @property
    def sum_conn_ms(self) -> float:
        return sum(c.wall_ms for c in self.connectors)

    @property
    def max_conn_ms(self) -> float:
        return max((c.wall_ms for c in self.connectors), default=0.0)

    @property
    def sum_conn_server_ms(self) -> float:
        return sum((c.server_ms or 0.0) for c in self.connectors)

    @property
    def sum_conn_data_ms(self) -> float:
        return sum(c.data_ms for c in self.connectors)

    @property
    def ids_digest(self) -> str:
        """Ordered, not set-based: a reordering silently breaks keyset paging
        even when the set is identical."""
        return hashlib.sha256("\n".join(self.ids).encode("utf-8")).hexdigest()

    def as_row(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "arm": self.arm,
            "request_ms": round(self.request_ms, 2),
            "access_ms": round(self.access_ms, 2),
            "fanout_wall_ms": round(self.fanout_wall_ms, 2),
            "merge_ms": round(self.merge_ms, 2),
            "other_ms": round(self.other_ms, 2),
            "sum_conn_ms": round(self.sum_conn_ms, 2),
            "max_conn_ms": round(self.max_conn_ms, 2),
            "sum_conn_server_ms": round(self.sum_conn_server_ms, 2),
            "sum_conn_data_ms": round(self.sum_conn_data_ms, 2),
            "n_connectors": self.n_connectors,
            "n_rows": len(self.ids),
            "total": self.total,
            "ids_digest": self.ids_digest[:16],
        }


class _StubConfig:
    """Neo4jProvider reads its credentials from the environment; config_service
    is only touched on an unrelated path (neo4j_provider.py:11440)."""

    async def get_config(self, key: str, **_kwargs):  # noqa: ANN001, ANN201
        return None


class Harness:
    def __init__(self, *, log_level: int = logging.INFO) -> None:
        self.settings = kh_env.settings()
        kh_env.export(self.settings)
        kh_env.add_backend_to_path()
        self.logger = build_logger("kh_harness")
        self.provider = None
        self.user: dict[str, str] = {}
        self._log_level = log_level

    async def connect(self, *, email: str | None = None,
                      user_id: str | None = None) -> None:
        from app.services.graph_db.neo4j.neo4j_provider import Neo4jProvider

        instr.patch()
        provider = Neo4jProvider(logger=self.logger, config_service=_StubConfig())
        if not await provider.connect():
            raise SystemExit("Could not connect to " + self.settings.uri)
        # connect() opens a session and builds no schema. ensure_schema() is
        # never called here: this points at the developer's real store.
        self.provider = provider
        self.user = await kh_env.resolve_user(
            provider.client, email=email, user_id=user_id
        )
        self.logger.setLevel(self._log_level)

    def set_log_level(self, level: int) -> None:
        self._log_level = level
        self.logger.setLevel(level)

    async def counts(self) -> dict[str, int]:
        return await kh_env.corpus_counts(self.provider.client)

    async def access(self, request_id: str = "adhoc") -> dict[str, Any]:
        with instr.span(request_id, "access"):
            return await self.provider.get_knowledge_hub_access_v3(
                user_key=self.user["user_key"], org_id=self.user["org_id"],
            )

    async def run_global_flatten(
        self, *, arm: str = "control", max_concurrency: int | None = None,
        limit: int = 50, cursor_token: str | None = None,
        filters: dict[str, Any] | None = None,
        access: dict[str, Any] | None = None,
        sort_field: str = "name", sort_dir: str = "ASC",
    ) -> RequestRecord:
        """One page of the global flatten, decomposed.

        `cursor_token` is the ONLY supported way to reach page 2. The service's
        `_walk_pages` (knowledge_hub_service.py:139) replays every prior page
        serially when given a page number, so `page=2` measures page 1 plus page
        2 and inverts the very ablation it looks like it provides.
        """
        from app.connectors.sources.localKB.handlers import kh_search

        request_id = uuid.uuid4().hex[:12]
        instr.COLLECTOR.reset()

        t_request = time.perf_counter()
        t0 = time.perf_counter()
        if access is None:
            access = await self.access(request_id)
        access_ms = (time.perf_counter() - t0) * 1000.0

        granted = access["by_connector"]
        original = self.provider.get_knowledge_hub_connector_page_v3

        async def spanned(*a, **k):
            with instr.span(request_id, "connector", k.get("app_id", "")):
                return await original(*a, **k)

        self.provider.get_knowledge_hub_connector_page_v3 = spanned
        try:
            t_fanout = time.perf_counter()
            page = await kh_search.search_page(
                self.provider,
                user_key=self.user["user_key"],
                user_id=self.user["user_key"],
                org_id=self.user["org_id"],
                limit=limit,
                sort_field=sort_field,
                sort_dir=sort_dir,
                cursor_token=cursor_token,
                filters=filters or {},
                access=access,
                max_concurrency=max_concurrency,
            )
            fanout_wall = (time.perf_counter() - t_fanout) * 1000.0
        finally:
            self.provider.get_knowledge_hub_connector_page_v3 = original

        request_ms = (time.perf_counter() - t_request) * 1000.0

        events = [e for e in instr.COLLECTOR.for_request(request_id)
                  if e.phase == "connector"]
        connectors = [
            ConnectorTiming(
                app_id=e.app_id, wall_ms=e.wall_ms, server_ms=e.server_ms,
                data_ms=e.wall_data_ms, py_ms=e.py_ms, rows=e.rows,
                n_granted_ids=len(granted.get(e.app_id) or []),
            )
            for e in events
        ]

        record = RequestRecord(
            request_id=request_id, arm=arm, request_ms=request_ms,
            access_ms=access_ms, fanout_wall_ms=fanout_wall,
            connectors=connectors,
            ids=[str(r.get("id")) for r in page.rows],
            total=page.total, counts=page.counts_by_type,
            n_connectors=len(connectors),
        )
        record.next_cursor = page.next_cursor
        return record

    async def close(self) -> None:
        if self.provider is not None:
            await self.provider.disconnect()
