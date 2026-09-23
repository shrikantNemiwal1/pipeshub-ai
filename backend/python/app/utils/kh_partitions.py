"""Which partitions a knowledge hub v2 global search runs (§3.9).

Both graph providers report the same facts per gated App: whether it is a
collection, its top-level record groups, and whether anything hangs off it
outside a group. This module turns those facts into partition descriptors, so
the rules for what a partition is live once instead of in two query dialects.
"""

from __future__ import annotations

from typing import Any

# Not a node id: every graph id is a uuid-like key, so this cannot collide with
# a group or App partition in the cursor.
APPS_PARTITION_ID = "__apps__"


def build_partitions(app_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Partition descriptors ``{partitionId, partitionKind, appId}``, in a stable order.

    * one ``APPS`` partition, which lists the Apps and collections themselves (D45);
    * ``COLLECTION`` for a knowledge base, which is one partition whole (D7);
    * ``GROUP`` for each top-level record group of a connector App;
    * ``APP_DIRECT`` for a connector App with nodes outside every group.

    ``app_rows`` are ``{appId, isCollection, groups, hasDirect}`` per gated App.
    """
    partitions: list[dict[str, Any]] = [
        {"partitionId": APPS_PARTITION_ID, "partitionKind": "APPS", "appId": None}
    ]
    for row in sorted(app_rows, key=lambda r: r["appId"]):
        app_id = row["appId"]
        if row.get("isCollection"):
            partitions.append({"partitionId": app_id, "partitionKind": "COLLECTION", "appId": app_id})
            continue
        for group_id in sorted(set(row.get("groups") or [])):
            partitions.append({"partitionId": group_id, "partitionKind": "GROUP", "appId": app_id})
        if row.get("hasDirect"):
            partitions.append({"partitionId": app_id, "partitionKind": "APP_DIRECT", "appId": app_id})
    return partitions
