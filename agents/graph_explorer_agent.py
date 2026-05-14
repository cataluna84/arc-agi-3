"""GraphExplorerAgent - priority-threshold graph exploration agent."""

from __future__ import annotations

import random
from typing import Any

from . import GameAction, GameState
from .graph_explorer import ActionCandidate, ActionKey, GraphExplorer
from .trigger_bfs_agent import _sample_click_xy, _trigger_score


class GraphExplorerAgent:
    """ARC-AGI-3 agent using frame segmentation + graph frontier routing."""

    name = "graph-explorer"

    def __init__(self, seed: int = 0, **_: Any) -> None:
        self._rng = random.Random(seed)
        self.explorer = GraphExplorer(self._rng)
        self.graph = self.explorer
        self._prev_hash: bytes | None = None
        self._prev_key: ActionKey | None = None
        self._prev_layers: list | None = None
        self._prev_levels = 0

    def choose_action(self, frame: Any) -> GameAction:
        try:
            return self._choose_action(frame)
        except Exception:
            return self._fallback_action(frame)

    def _choose_action(self, frame: Any) -> GameAction:
        if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._prev_hash = None
            self._prev_key = None
            self._prev_layers = None
            return GameAction.RESET

        cur_layers = list(getattr(frame, "frame", []) or [])
        cur_levels = int(getattr(frame, "levels_completed", 0))
        self.explorer.maybe_reset_for_level(cur_levels)

        analysis = self.explorer.analyze_layers(cur_layers)
        state_hash = analysis.state_hash
        available = list(getattr(frame, "available_actions", []) or [1, 2, 3, 4, 5, 6, 7])
        self.explorer.add_or_get_state(state_hash, available, cur_levels, analysis)

        change_score = _trigger_score(self._prev_layers, cur_layers, self._prev_levels, cur_levels)
        if self._prev_hash is not None and self._prev_key is not None:
            self.explorer.observe(self._prev_hash, self._prev_key, state_hash, change_score)

        candidate = self.explorer.next_action(state_hash, available)
        action = self._candidate_to_action(candidate, cur_layers)

        self._prev_hash = state_hash
        self._prev_key = candidate.key
        self._prev_layers = cur_layers
        self._prev_levels = cur_levels
        self.explorer.record_action(state_hash, candidate)
        return action

    def _candidate_to_action(
        self,
        candidate: ActionCandidate,
        cur_layers: list | None,
    ) -> GameAction:
        action = GameAction.from_id(candidate.action_id)
        if action.is_complex():
            data = dict(candidate.data) or _sample_click_xy(cur_layers, self._rng)
            action.set_data({"x": int(data["x"]) % 64, "y": int(data["y"]) % 64})
        return action

    def _fallback_action(self, frame: Any) -> GameAction:
        cur_layers = list(getattr(frame, "frame", []) or [])
        available = list(getattr(frame, "available_actions", []) or [1, 2, 3, 4, 5, 6, 7])
        legal: list[int] = []
        for action in available:
            try:
                action_id = int(action)
            except TypeError:
                action_id = int(getattr(action, "value", action))
            if action_id not in (0, 6):
                legal.append(action_id)
        if legal:
            return GameAction.from_id(self._rng.choice(legal))

        action = GameAction.ACTION6
        data = _sample_click_xy(cur_layers, self._rng)
        action.set_data(data)
        return action

    def is_done(self, frame: Any) -> bool:
        return frame.state == GameState.WIN


__all__ = ["GraphExplorerAgent"]
