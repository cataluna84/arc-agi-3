"""state_graph.py - shared state-graph wrapper for search-based ARC-AGI-3 agents.

Used by `agents/trigger_bfs_agent.py` (D5+D6 of `experiments/SPEC_4WEEKS.md`).
Designed so that future search agents (Goose++, GoExplore, ForgeV20) can
reuse the same primitives without duplicating frame-hash and frontier code.

Public surface:
    hash_frame(frame_layers)              -> bytes (8-byte blake2b digest)
    StateNode (dataclass)                  -> per-state record
    StateGraph                             -> graph + frontier + level reset

The graph stores only what the agent actually observes. There is no
simulator-state restore (forbidden by SDK section 0.9.3 COMPETITION mode);
"replay" means re-issuing recorded actions from a level reset.
"""

from __future__ import annotations

import contextlib
import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def hash_frame(frame_layers: Iterable) -> bytes:
    """Stable 8-byte digest over a list of 64x64 grids (numpy arrays or list-of-lists)."""
    h = hashlib.blake2b(digest_size=8)
    if frame_layers is None:
        return h.digest()
    for layer in frame_layers:
        tobytes = getattr(layer, "tobytes", None)
        if callable(tobytes):
            h.update(tobytes())
        else:
            for row in layer:
                h.update(bytes(int(v) % 256 for v in row))
    return h.digest()


@dataclass
class StateNode:
    state_hash: bytes
    visit_count: int = 0
    untried_actions: set[int] = field(default_factory=set)
    edges: dict[int, bytes] = field(default_factory=dict)
    last_score: int = 0
    last_levels: int = 0
    incoming_change_score: float = 0.0


class StateGraph:
    """Node + edge store with a frontier of states that still have untried actions.

    Untried actions are seeded from `available_actions` the first time a state
    is observed. They are removed by `observe(prev, action, next, ...)`.
    On level transitions, the entire graph is cleared because game mechanics
    may change (this is the canonical `dolphin-in-a-coma` bug from gotcha #4).
    """

    def __init__(self) -> None:
        self.nodes: dict[bytes, StateNode] = {}
        self.frontier: deque[bytes] = deque()
        self.action_history: list[tuple[bytes, int, dict]] = []
        self.current_levels: int = 0

    def reset(self) -> None:
        self.nodes.clear()
        self.frontier.clear()
        self.action_history.clear()

    def maybe_reset_for_level(self, levels: int) -> bool:
        if levels != self.current_levels:
            self.reset()
            self.current_levels = levels
            return True
        return False

    def add_or_get(
        self,
        state_hash: bytes,
        available_actions: Iterable[int] | None,
        levels: int,
        score: int = 0,
    ) -> StateNode:
        node = self.nodes.get(state_hash)
        if node is None:
            untried = {int(a) for a in (available_actions or []) if int(a) != 0}
            node = StateNode(
                state_hash=state_hash,
                untried_actions=untried,
                last_levels=levels,
                last_score=score,
            )
            self.nodes[state_hash] = node
            if untried:
                self.frontier.append(state_hash)
        return node

    def observe(
        self,
        prev_hash: bytes | None,
        action_id: int,
        next_hash: bytes,
        change_score: float = 0.0,
    ) -> None:
        if prev_hash is not None and prev_hash in self.nodes:
            node = self.nodes[prev_hash]
            node.edges[action_id] = next_hash
            node.untried_actions.discard(action_id)
            if not node.untried_actions:
                with contextlib.suppress(ValueError):
                    self.frontier.remove(prev_hash)
        if next_hash in self.nodes:
            self.nodes[next_hash].visit_count += 1
            self.nodes[next_hash].incoming_change_score = change_score

    def record_action(self, state_hash: bytes, action_id: int, data: dict) -> None:
        self.action_history.append((state_hash, action_id, dict(data)))

    def stats(self) -> dict:
        return {
            "n_nodes": len(self.nodes),
            "n_frontier": len(self.frontier),
            "n_actions_recorded": len(self.action_history),
            "current_levels": self.current_levels,
        }


__all__ = ["StateGraph", "StateNode", "hash_frame"]
