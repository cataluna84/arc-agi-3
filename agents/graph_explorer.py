"""GraphExplorer primitives for ARC-AGI-3 systematic exploration.

This module ports the action-scheduler half of the graph-exploration
approach described by Rudakov et al. (arXiv:2512.24156): keep a directed
state graph, test all actions up to the current priority threshold, and
walk shortest paths back to frontier states when the current state is
exhausted.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .state_graph import hash_frame


@dataclass(frozen=True, order=True)
class ActionKey:
    """State-local action identity.

    Simple actions use an empty target. ACTION6 uses a segment target so
    different object clicks are tracked independently instead of all
    overwriting action id 6.
    """

    action_id: int
    target: tuple[int, ...] = ()


@dataclass(frozen=True)
class ActionCandidate:
    key: ActionKey
    action_id: int
    priority: int
    data: dict[str, int] = field(default_factory=dict)


@dataclass
class FrameAnalysis:
    state_hash: bytes
    label_map: Any | None = None
    segments: list[Any] = field(default_factory=list)
    priority_tiers: list[set[int]] = field(default_factory=list)


@dataclass
class ExplorerNode:
    state_hash: bytes
    candidates: dict[ActionKey, ActionCandidate] = field(default_factory=dict)
    untested: set[ActionKey] = field(default_factory=set)
    edges: dict[ActionKey, bytes] = field(default_factory=dict)
    visit_count: int = 0
    incoming_change_score: float = 0.0


def _to_ndarray(layer: Any):
    try:
        import numpy as np

        if isinstance(layer, np.ndarray):
            return layer
        return np.asarray(layer, dtype=np.uint8)
    except ImportError:
        return None


def _legal_action_ids(available_actions: Any) -> list[int]:
    ids: list[int] = []
    for action in available_actions or []:
        try:
            action_id = int(action)
        except TypeError:
            value = getattr(action, "value", action)
            action_id = int(value)
        if action_id != 0:
            ids.append(action_id)
    return ids or [1, 2, 3, 4, 5, 6, 7]


class GraphExplorer:
    """Priority-threshold state graph with shortest-path frontier routing."""

    def __init__(self, rng: random.Random | None = None, max_priority: int = 4) -> None:
        self.rng = rng or random.Random()
        self.max_priority = max_priority
        self.current_priority = 0
        self.nodes: dict[bytes, ExplorerNode] = {}
        self.current_levels = 0
        self.action_history: list[tuple[bytes, ActionKey, dict[str, int]]] = []

    def reset(self) -> None:
        self.current_priority = 0
        self.nodes.clear()
        self.action_history.clear()

    def maybe_reset_for_level(self, levels: int) -> bool:
        if levels >= 0 and levels != self.current_levels:
            self.reset()
            self.current_levels = levels
            return True
        return False

    def analyze_layers(self, layers: list | None) -> FrameAnalysis:
        if not layers:
            return FrameAnalysis(state_hash=hash_frame(layers or []))

        grid = _to_ndarray(layers[-1])
        if grid is None or getattr(grid, "ndim", 0) != 2:
            return FrameAnalysis(state_hash=hash_frame(layers))

        try:
            from . import frame_segmenter as fs

            label_map, segments = fs.segment_frame(grid)
            status_mask, status_groups = fs.identify_status_bars(label_map, segments)
            status_ids: set[int] = set()
            for group in status_groups:
                status_ids.update(group)
            tiers = fs.frame_segments_to_priority_tiers(
                segments,
                status_bar_segment_ids=status_ids,
            )
            return FrameAnalysis(
                state_hash=fs.hash_masked_frame(grid, status_mask),
                label_map=label_map,
                segments=segments,
                priority_tiers=tiers,
            )
        except Exception:
            return FrameAnalysis(state_hash=hash_frame(layers))

    def add_or_get_state(
        self,
        state_hash: bytes,
        available_actions: Any,
        levels: int,
        analysis: FrameAnalysis,
    ) -> ExplorerNode:
        del levels
        node = self.nodes.get(state_hash)
        if node is None:
            node = ExplorerNode(state_hash=state_hash)
            self.nodes[state_hash] = node

        for candidate in self._build_candidates(available_actions, analysis):
            if candidate.key not in node.candidates:
                node.candidates[candidate.key] = candidate
                node.untested.add(candidate.key)
        return node

    def observe(
        self,
        prev_hash: bytes | None,
        action_key: ActionKey | None,
        next_hash: bytes,
        change_score: float = 0.0,
    ) -> None:
        if prev_hash is not None and action_key is not None and prev_hash in self.nodes:
            node = self.nodes[prev_hash]
            node.edges[action_key] = next_hash
            node.untested.discard(action_key)
        if next_hash in self.nodes:
            self.nodes[next_hash].visit_count += 1
            self.nodes[next_hash].incoming_change_score = change_score

    def next_action(self, current_hash: bytes, available_actions: Any) -> ActionCandidate:
        legal = set(_legal_action_ids(available_actions))
        node = self.nodes[current_hash]

        while self.current_priority <= self.max_priority:
            local = self._untested_candidates(node, legal, self.current_priority)
            if local:
                return self._choose_candidate(local)

            first_key = self._shortest_path_to_frontier(
                current_hash,
                legal,
                self.current_priority,
            )
            if first_key is not None:
                return node.candidates[first_key]

            self.current_priority += 1

        return self._fallback_candidate(node, legal)

    def record_action(self, state_hash: bytes, candidate: ActionCandidate) -> None:
        self.action_history.append((state_hash, candidate.key, dict(candidate.data)))

    def stats(self) -> dict[str, int]:
        frontier = sum(1 for node in self.nodes.values() if node.untested)
        return {
            "n_nodes": len(self.nodes),
            "n_frontier": frontier,
            "n_actions_recorded": len(self.action_history),
            "current_levels": self.current_levels,
            "current_priority": self.current_priority,
        }

    def _build_candidates(
        self,
        available_actions: Any,
        analysis: FrameAnalysis,
    ) -> list[ActionCandidate]:
        legal = _legal_action_ids(available_actions)
        candidates: list[ActionCandidate] = []

        candidates.extend(
            [
                ActionCandidate(
                    key=ActionKey(action_id),
                    action_id=action_id,
                    priority=0,
                )
                for action_id in sorted(a for a in legal if a != 6)
            ]
        )

        if 6 not in legal:
            return candidates

        candidates.extend(self._action6_segment_candidates(analysis))
        if not any(c.action_id == 6 for c in candidates):
            data = {"x": self.rng.randint(0, 63), "y": self.rng.randint(0, 63)}
            candidates.append(
                ActionCandidate(
                    key=ActionKey(6, (-1,)),
                    action_id=6,
                    priority=self.max_priority,
                    data=data,
                )
            )
        return candidates

    def _action6_segment_candidates(self, analysis: FrameAnalysis) -> list[ActionCandidate]:
        if analysis.label_map is None or not analysis.priority_tiers:
            return []

        try:
            from . import frame_segmenter as fs
        except ImportError:
            return []

        frame_pixels = int(analysis.label_map.shape[0] * analysis.label_map.shape[1])
        half = frame_pixels // 2
        candidates: list[ActionCandidate] = []
        for priority, segment_ids in enumerate(analysis.priority_tiers[: self.max_priority + 1]):
            for sid in sorted(segment_ids):
                if sid >= len(analysis.segments):
                    continue
                if int(analysis.segments[sid].area) > half:
                    continue
                coord = fs.mask_to_click_coords(analysis.label_map, sid, rng=self.rng)
                if coord is None:
                    continue
                candidates.append(
                    ActionCandidate(
                        key=ActionKey(6, (priority, sid)),
                        action_id=6,
                        priority=priority,
                        data={"x": coord[0], "y": coord[1]},
                    )
                )
        return candidates

    def _untested_candidates(
        self,
        node: ExplorerNode,
        legal: set[int],
        priority: int,
    ) -> list[ActionCandidate]:
        return [
            node.candidates[key]
            for key in node.untested
            if key in node.candidates
            and node.candidates[key].action_id in legal
            and node.candidates[key].priority <= priority
        ]

    def _choose_candidate(self, candidates: list[ActionCandidate]) -> ActionCandidate:
        best_priority = min(c.priority for c in candidates)
        best = [c for c in candidates if c.priority == best_priority]
        return self.rng.choice(sorted(best, key=lambda c: c.key))

    def _shortest_path_to_frontier(
        self,
        start_hash: bytes,
        legal_first_actions: set[int],
        priority: int,
    ) -> ActionKey | None:
        seen = {start_hash}
        queue: deque[tuple[bytes, ActionKey | None]] = deque([(start_hash, None)])

        while queue:
            state_hash, first_key = queue.popleft()
            node = self.nodes.get(state_hash)
            if node is None:
                continue
            if first_key is not None and self._untested_candidates(
                node,
                set(range(1, 8)),
                priority,
            ):
                return first_key

            for key, next_hash in sorted(node.edges.items(), key=lambda item: item[0]):
                candidate = node.candidates.get(key)
                if candidate is None:
                    continue
                if first_key is None and candidate.action_id not in legal_first_actions:
                    continue
                if next_hash in seen:
                    continue
                seen.add(next_hash)
                queue.append((next_hash, key if first_key is None else first_key))
        return None

    def _fallback_candidate(self, node: ExplorerNode, legal: set[int]) -> ActionCandidate:
        simple = sorted(a for a in legal if a != 6)
        if simple:
            action_id = self.rng.choice(simple)
            return ActionCandidate(
                key=ActionKey(action_id, (-2, len(self.action_history))),
                action_id=action_id,
                priority=self.max_priority + 1,
            )

        action6 = [c for c in node.candidates.values() if c.action_id == 6]
        if action6:
            return self._choose_candidate(action6)

        data = {"x": self.rng.randint(0, 63), "y": self.rng.randint(0, 63)}
        return ActionCandidate(
            key=ActionKey(6, (-3, len(self.action_history))),
            action_id=6,
            priority=self.max_priority + 1,
            data=data,
        )


__all__ = [
    "ActionCandidate",
    "ActionKey",
    "ExplorerNode",
    "FrameAnalysis",
    "GraphExplorer",
]
