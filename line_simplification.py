"""Line simplification algorithm.

Implements Visvalingam-Whyatt algorithm using priority queue.

See
---
[Visvalingam-Whyatt algorithm](https://en.wikipedia.org/wiki/Visvalingam%E2%80%93Whyatt_algorithm)
"""
from __future__ import annotations

import itertools
import math
import typing as t
from dataclasses import dataclass
from heapq import heappop, heappush

################################################################################
### Double linked list of indices
################################################################################


@dataclass(slots=True)
class IndexLink:
    """Index link for double linked list of indices"""

    prev: int | None
    next: int | None


def _iter_linked_indices(n: int) -> t.Iterable[IndexLink]:
    yield IndexLink(None, 1)
    yield from (IndexLink(i - 1, i + 1) for i in range(1, n - 1))
    yield IndexLink(n - 1, None)


def make_linked_indices(length: int) -> tuple[IndexLink]:
    """Create a double linked list of indices of given length."""
    return tuple(_iter_linked_indices(n=length))


def iter_linked_indices(linked_indices: t.Sequence[IndexLink]) -> t.Iterable[int]:
    """Iterate linked indices in ascending order from first link."""
    cur = 0
    yield cur
    while linked_indices[cur].next is not None:
        cur = linked_indices[cur].next
        yield cur


################################################################################
### Priority queue
################################################################################


@dataclass(slots=True, order=True)
class Entry:
    """Priority queue entry"""

    priority: float
    count: int
    index: int


class PriorityQueue:
    """Priority queue of indicies."""

    def __init__(self):
        self.queue: list[Entry] = []
        self.index_map: dict[int, Entry] = {}
        self.counter = itertools.count()

    def add(self, index: int, priority: float):
        """Add new element or update its priority."""
        if index in self.index_map:
            self.remove(index)
        count = next(self.counter)
        entry = Entry(priority, count, index)
        self.index_map[index] = entry
        heappush(self.queue, entry)

    def remove(self, index: int):
        """Remove an element from the queue.

        Mark the element as empty.

        Raise
        -----
        KeyError
            if index is not found
        """
        entry = self.index_map.pop(index)
        entry.index = -1

    def pop(self) -> tuple[int, float]:
        """Remove and return the lowest priority element with its priority.

        Raise
        -----
        KeyError
            raise `KeyError` when called on an empty queue.
        """
        while self.queue:
            entry = heappop(self.queue)
            if entry.index >= 0:
                del self.index_map[entry.index]
                return entry.index, entry.priority
        raise KeyError("pop from an empty priority queue")


################################################################################
### Simplification algorithm
################################################################################


Point = t.Sequence[float]


def compute_priority(p1: Point, p2: Point, p3: Point) -> float:
    """Return 2 times the area formed by the triangle of the points in the node"""
    return abs(
        p1[0] * p2[1]
        + p2[0] * p3[1]
        + p3[0] * p1[1]
        - p1[0] * p3[1]
        - p2[0] * p1[1]
        - p3[0] * p2[1]
    )


def simplify_eps(line: t.Sequence[Point], eps: float) -> list[Point]:
    """Simplify line by priority.

    Remove points which priority are lower then given argument
    """
    # Init
    n = len(line)
    lli = make_linked_indices(n)
    pqueue: PriorityQueue = PriorityQueue()
    pqueue.add(0, math.inf)
    for i, (pt_prev, pt_cur, pt_next) in enumerate(zip(line, line[1:], line[2:]), start=1):
        pqueue.add(i, compute_priority(pt_prev, pt_cur, pt_next))

    # While loop
    num_remaining = n
    while num_remaining > 2:
        ## Finding point with least importance
        cur_idx, priority = pqueue.pop()

        ## Break condition
        if priority > eps:
            break

        cur_link = lli[cur_idx]

        # we are sure cur_link.prev and cur_link.next exists because border points have math.inf prio
        # because border points have math.inf priority
        prev_idx = t.cast(int, cur_link.prev)
        next_idx = t.cast(int, cur_link.next)

        ## Removing point / relinking
        lli[prev_idx].next = cur_link.next
        lli[next_idx].prev = cur_link.prev
        num_remaining -= 1

        ## Updating priorities
        prev_prev_idx = lli[prev_idx].prev
        if prev_prev_idx is not None:
            pqueue.add(
                prev_idx,
                compute_priority(
                    line[prev_prev_idx],
                    line[prev_idx],
                    line[next_idx],
                ),
            )
        next_next_idx = lli[next_idx].next
        if next_next_idx is not None:
            pqueue.add(
                next_idx,
                compute_priority(
                    line[prev_idx],
                    line[next_idx],
                    line[next_next_idx],
                ),
            )

    # Rebuilding points
    return [line[i] for i in iter_linked_indices(lli)]


def simplify_num(line: t.Sequence[Point], num: int) -> list[Point]:
    """Simplify line down to a given number of points.

    Return the line as is if it has fewer points than the target
    """
    # Init
    n = len(line)
    if n <= num:
        return list(line)
    lli = make_linked_indices(n)
    pqueue: PriorityQueue = PriorityQueue()
    # Initialisizing priority queue
    pqueue.add(0, math.inf)
    for i, (pt_prev, pt_cur, pt_next) in enumerate(zip(line, line[1:], line[2:]), start=1):
        pqueue.add(i, compute_priority(pt_prev, pt_cur, pt_next))

    # While loop
    num_remaining = n
    while num_remaining > num:
        ## Finding point with least importance
        cur_idx, _ = pqueue.pop()
        cur_link = lli[cur_idx]

        # we are sure cur_link.prev and cur_link.next exists
        # (meaning current is not a border point)
        # because border points have math.inf priority
        prev_idx = t.cast(int, cur_link.prev)
        next_idx = t.cast(int, cur_link.next)

        ## Removing point / relinking
        lli[prev_idx].next = cur_link.next
        lli[next_idx].prev = cur_link.prev
        num_remaining -= 1

        ## Updating priorities
        prev_prev_idx = lli[prev_idx].prev
        if prev_prev_idx is not None:
            pqueue.add(
                prev_idx,
                compute_priority(
                    line[prev_prev_idx],
                    line[prev_idx],
                    line[next_idx],
                ),
            )
        next_next_idx = lli[next_idx].next
        if next_next_idx is not None:
            pqueue.add(
                next_idx,
                compute_priority(
                    line[prev_idx],
                    line[next_idx],
                    line[next_next_idx],
                ),
            )

    # Rebuilding points
    return [line[i] for i in iter_linked_indices(lli)]


def simplify(
    line: t.Sequence[Point], eps: float | None = None, num: int | None = None
) -> list[Point]:
    """Simplify line.

    Either pass `eps` or `num`.
    
    Parameters
    ----------
    line
        sequence of points
    eps
        removed points which priorities are lower than `eps`
    num
        removed points by priority down to `num` point
    """
    if eps is not None:
        if num is not None:
            raise ValueError("Either pass a value for `eps` or `num`")
        return simplify_eps(line=line, eps=eps)
    else:
        if num is None:
            raise ValueError("Pass a value for either `eps` or `num`")
        return simplify_num(line=line, num=num)


__all__ = (
    "simplify",
    "simplify_eps",
    "simplify_num"
)
