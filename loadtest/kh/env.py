"""Connection, target user, and the read-only guards for the KH v3 harness.

This harness points at the DEVELOPER's real store on the default ports, which is
the exact opposite of `tests/integration/graph_permissions/conftest.py` — that
suite refuses 7687 because it wipes what it connects to. So the guard here is
inverted: assert we only ever read. `ensure_schema()` is never called, and
`assert_corpus_unchanged` fails the run if node or relationship counts moved
underneath it (the live stack is running on the same host and can be indexing).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "backend" / "python"


def load_dotenv(path: Path) -> dict[str, str]:
    """Parse a KEY=VALUE .env. No interpolation, no export handling — the file
    this reads is written by install.sh and uses neither."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True)
class Settings:
    uri: str
    username: str
    password: str
    database: str


def settings() -> Settings:
    """Neo4j credentials: real environment first, then backend/python/.env.

    The password is never printed; callers that echo settings must use
    `redacted()`.
    """
    dotenv = load_dotenv(BACKEND / ".env")

    def pick(key: str, default: str) -> str:
        return os.environ.get(key) or dotenv.get(key) or default

    resolved = Settings(
        uri=pick("NEO4J_URI", "bolt://localhost:7687"),
        username=pick("NEO4J_USERNAME", "neo4j"),
        password=pick("NEO4J_PASSWORD", ""),
        database=pick("NEO4J_DATABASE", "neo4j"),
    )
    if not resolved.password:
        raise SystemExit(
            "No Neo4j password. Set NEO4J_PASSWORD or put it in backend/python/.env"
        )
    return resolved


def redacted(s: Settings) -> dict[str, str]:
    return {"uri": s.uri, "username": s.username, "database": s.database,
            "password": "<redacted>"}


def export(s: Settings) -> None:
    """Neo4jProvider.connect() reads these straight from os.environ."""
    os.environ["NEO4J_URI"] = s.uri
    os.environ["NEO4J_USERNAME"] = s.username
    os.environ["NEO4J_PASSWORD"] = s.password
    os.environ["NEO4J_DATABASE"] = s.database


def add_backend_to_path() -> None:
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))


COUNT_CYPHER = """
CALL () { MATCH (n) RETURN count(n) AS nodes }
CALL () { MATCH ()-[r]->() RETURN count(r) AS rels }
RETURN nodes, rels
"""


async def corpus_counts(client) -> dict[str, int]:
    rows = await client.execute_query(COUNT_CYPHER, parameters={})
    return {"nodes": rows[0]["nodes"], "rels": rows[0]["rels"]}


def assert_corpus_unchanged(before: dict[str, int], after: dict[str, int]) -> None:
    """A store that grew mid-run invalidates every comparison in that run."""
    if before != after:
        raise SystemExit(
            f"Corpus changed during the run ({before} -> {after}). "
            "Something wrote to the store; the numbers are not comparable."
        )


USER_CYPHER = """
MATCH (u:User)
WHERE ($email IS NULL OR u.email = $email)
  AND ($user_id IS NULL OR u.id = $user_id)
OPTIONAL MATCH (u)-[:BELONGS_TO]->(o:Organization)
RETURN u.id AS user_key, u.email AS email, coalesce(o.id, u.orgId) AS org_id
ORDER BY user_key
LIMIT 1
"""


async def resolve_user(client, *, email: str | None = None,
                       user_id: str | None = None) -> dict[str, str]:
    rows = await client.execute_query(
        USER_CYPHER, parameters={"email": email, "user_id": user_id}
    )
    if not rows:
        raise SystemExit(f"No such user (email={email!r} id={user_id!r})")
    row = rows[0]
    if not row.get("org_id"):
        raise SystemExit(f"User {row['user_key']} has no organization; cannot scope a request")
    return row
