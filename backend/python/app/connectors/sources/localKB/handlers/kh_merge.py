"""Merge per-partition rows into one ordered page.

Each partition is sorted by the query itself; this combines them without
re-sorting and without loading more than one row per partition at a time.

**One comparator, defined once.** `SortKey` orders by the ``nullRank`` and
``sortKey`` the query emitted, then by id. It is the only ordering in the read
path: the query sorts with it and the merge re-uses its own output, so the two
cannot drift apart the way two implementations of "the same" ordering do.

**The page's edges are its boundaries.** Every partition sorts on the same
comparator, so the page's last row is where every partition resumes going
forward and its first row where they resume going back (``kh_cursor``). Those
are the rows the merge *emitted*, never a row it merely buffered.

**Not `heapq.merge`.** That helper buffers one row ahead per iterator, and a
wrapper reporting "the last row I pulled" reports a row that was never
emitted; resuming after it skips that row — a record silently missing from a
paged listing, which no assertion about page length would catch. The explicit
head-heap below pops exactly the row it consumes.

**Duplicates across partitions.** A record reachable by two hierarchy parents
(a Drive file in both its folder and Shared with Me) arrives once per partition
with an identical sort key, so the copies are adjacent. The real hierarchy
parent wins over the internal one (decision 67), both copies are consumed, and
one more row is pulled so the page stays full.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from app.utils.kh_cursor import Boundary

# A page that drops this many duplicates in a row is not a page of duplicates,
# it is a bug — most likely every partition returning the same subtree. Raising
# beats spinning through an unbounded result set.
_MAX_CONSECUTIVE_DROPS = 1000


class MergeError(RuntimeError):
    """The merge cannot produce a correct page."""


@dataclass(frozen=True)
class SortKey:
    """The query's own comparator output for one row.

    ``null_rank`` sorts ahead of the value in **both** directions: nulls keep
    their place when the sort flips, which is what the query's own ORDER BY
    does. The id tiebreak is likewise always ascending, so a descending page
    and its reverse walk agree on the order of tied rows — without that,
    ``prev`` would not return exactly the page the user came from.
    """

    null_rank: int
    value: Any
    row_id: str
    descending: bool = False

    def __lt__(self, other: "SortKey") -> bool:
        if self.null_rank != other.null_rank:
            return self.null_rank < other.null_rank
        if self.value != other.value:
            try:
                less = self.value < other.value
            except TypeError as exc:
                raise MergeError(
                    f"cannot order {self.value!r} against {other.value!r}; the "
                    f"query emitted mixed types for one sort field"
                ) from exc
            return not less if self.descending else less
        return self.row_id < other.row_id


@dataclass(frozen=True)
class _Backward:
    """A `SortKey` walked from the end: the row nearest the boundary pops first."""

    key: SortKey

    def __lt__(self, other: "_Backward") -> bool:
        return other.key < self.key


def key_for(row: dict, descending: bool) -> SortKey:
    try:
        return SortKey(
            null_rank=row["nullRank"],
            value=row["sortKey"],
            row_id=row["id"],
            descending=descending,
        )
    except KeyError as exc:
        raise MergeError(
            f"row is missing {exc.args[0]!r}; the query must return its own "
            f"sortKey and nullRank so the merge uses the same comparator"
        ) from exc


@dataclass
class PartitionFeed:
    """One partition's rows, in the order the merge walks them.

    Forward, that is page order. Backward (``reverse=True``), nearest the
    boundary first — the reverse of page order.
    """

    partition_id: str
    partition_kind: str
    rows: Iterator[dict]


@dataclass
class MergeResult:
    rows: list[dict] = field(default_factory=list)
    first: Boundary | None = None
    last: Boundary | None = None
    exhausted: set[str] = field(default_factory=set)
    dropped: int = 0


def merge_pages(
    feeds: Sequence[PartitionFeed],
    *,
    limit: int,
    descending: bool = False,
    reverse: bool = False,
) -> MergeResult:
    """Take the next `limit` rows across all partitions, returned in page order.

    ``reverse`` takes the `limit` rows nearest the boundary going backwards, for
    a previous page. ``exhausted`` names the feeds that ran dry while merging.
    """
    if limit <= 0:
        raise MergeError(f"limit must be positive, got {limit}")

    result = MergeResult()
    heap: list[tuple[Any, int]] = []
    head: dict[int, dict] = {}

    def pull(index: int) -> None:
        row = next(feeds[index].rows, None)
        if row is None:
            result.exhausted.add(feeds[index].partition_id)
            return
        head[index] = row
        key = key_for(row, descending)
        heapq.heappush(heap, (_Backward(key) if reverse else key, index))

    for index in range(len(feeds)):
        pull(index)

    emitted: set[str] = set()
    consecutive_drops = 0

    while heap and len(result.rows) < limit:
        _, index = heapq.heappop(heap)
        row = head.pop(index)

        # Copies of one row arrive adjacent, because their sort keys are equal.
        group = [(index, row)]
        while heap and head[heap[0][1]]["id"] == row["id"]:
            _, other = heapq.heappop(heap)
            group.append((other, head.pop(other)))

        if row["id"] in emitted:
            result.dropped += 1
            consecutive_drops += 1
            if consecutive_drops > _MAX_CONSECUTIVE_DROPS:
                raise MergeError(
                    f"dropped {consecutive_drops} consecutive duplicate rows; "
                    f"the partitions are probably overlapping wholesale"
                )
        else:
            result.rows.append(_winner(group))
            emitted.add(row["id"])
            consecutive_drops = 0
            result.dropped += len(group) - 1

        for member_index, _ in group:
            pull(member_index)

    if reverse:
        result.rows.reverse()
    if result.rows:
        result.first = Boundary.of(result.rows[0])
        result.last = Boundary.of(result.rows[-1])
    return result


def _winner(group: list[tuple[int, dict]]) -> dict:
    """Decision 67: a real hierarchy parent beats an internal container.

    Shared with Me is internal only because it has no source id of its own. If
    another connector ever writes an internal group, placement changes here and
    nowhere else — which is why the test names the rule rather than the group.
    """
    for _, row in group:
        if not row.get("parentIsInternal"):
            return row
    return group[0][1]
