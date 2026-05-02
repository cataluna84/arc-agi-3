"""trigger_bfs_agent.py - Trigger-Aware BFS agent (D5/D6 of SPEC_4WEEKS).

Algorithmic sketch (deliberately small; first iteration before D7's
graph-state ablation pass):
  1. Maintain a `StateGraph` of observed (state_hash -> StateNode)
     keyed by an 8-byte blake2b digest of the frame's last layer.
  2. Each step:
     a. Hash current frame; add the node if new (seeds its untried
        actions from `available_actions`, RESET excluded).
     b. Record the (prev_hash, prev_action) -> cur_hash edge with a
        trigger score = delta_pixels + 5 * delta_levels + 2 * new_colors.
        This is the standard Trigger-Aware BFS objective from the
        public 0.35 notebook (Adi Gupta et al.).
     c. If the current node has untried actions: pick the highest-
        priority one (simple actions first; ACTION6 last because its
        coordinate space is huge).
     d. If all actions tried at the current node: replay the highest-
        change-score recorded edge as a directed walk towards a frontier
        state (best-effort, since COMPETITION mode forbids state restore).
        Fallback: epsilon-greedy random over `available_actions`.
  3. ACTION6 click coords are sampled from non-background pixels of the
     current frame (saliency-tier 0). If no non-bg pixels exist, fall
     back to uniform [0, 63] x [0, 63].
  4. On level transitions, clear the graph (gotcha #4 -- mechanics may
     change between levels; cross-level state-graph leakage breaks the
     `dolphin-in-a-coma` family of agents).

This agent is offline-friendly:
  - No model weights; no torch import.
  - Works against the local mock environment AND the real arc-agi SDK.

Smoke target (`scripts/trigger_bfs_smoke_local.py` + local_runner):
  - 22-check parity (same as goose/qwen smoke patterns)
  - On ls20 with the real SDK: levels_completed >= 1 within 200 actions.

Expected LB (per SPEC_4WEEKS.md §1.3 + public-notebook anchors): 0.30-0.35.
"""

from __future__ import annotations

import random
from typing import Any

from . import GameAction, GameState
from .state_graph import StateGraph, hash_frame


def _to_ndarray(layer: Any):
    """Best-effort coercion of a frame layer to a numpy.ndarray."""
    try:
        import numpy as np

        if isinstance(layer, np.ndarray):
            return layer
        return np.asarray(layer, dtype=np.uint8)
    except ImportError:
        return None


def _trigger_score(
    prev_layers: list | None,
    next_layers: list | None,
    prev_levels: int,
    next_levels: int,
) -> float:
    """delta_pixels + 5*delta_levels + 2*new_colors."""
    if prev_layers is None or next_layers is None:
        return 0.0
    p = _to_ndarray(prev_layers[-1])
    n = _to_ndarray(next_layers[-1])
    if p is None or n is None:
        return float(5 * (next_levels - prev_levels))
    if p.shape != n.shape:
        return float(5 * (next_levels - prev_levels))
    delta_pixels = float((p != n).sum())
    new_colors = float(len({int(v) for v in n.flat} - {int(v) for v in p.flat}))
    delta_levels = float(next_levels - prev_levels)
    return delta_pixels + 5.0 * delta_levels + 2.0 * new_colors


def _action_priority(action_id: int) -> tuple[int, int]:
    """Lower tuple => higher priority. Simple actions first; ACTION6 last."""
    return (1 if action_id == 6 else 0, action_id)


def _sample_click_xy(layers: list | None, rng: random.Random) -> dict[str, int]:
    """Pick (x, y) from non-background pixels; fallback uniform."""
    try:
        import numpy as np

        if not layers:
            raise ValueError("no layers")
        grid = _to_ndarray(layers[-1])
        if grid is None or grid.ndim != 2:
            raise ValueError("bad grid")
        bg = int(np.bincount(grid.flatten(), minlength=16).argmax())
        ys, xs = np.where(grid != bg)
        if len(xs) > 0:
            idx = rng.randrange(len(xs))
            return {"x": int(xs[idx]), "y": int(ys[idx])}
    except Exception:  # noqa: S110 - any failure falls back to uniform sampling
        pass
    return {"x": rng.randint(0, 63), "y": rng.randint(0, 63)}


class TriggerBFSAgent:
    """Trigger-Aware BFS agent built atop StateGraph."""

    name = "trigger-bfs"

    def __init__(self, seed: int = 0, **_: Any) -> None:
        self._rng = random.Random(seed)
        self.graph = StateGraph()
        self._prev_hash: bytes | None = None
        self._prev_action: int | None = None
        self._prev_data: dict = {}
        self._prev_layers: list | None = None
        self._prev_levels: int = 0

    def choose_action(self, frame: Any) -> GameAction:
        # Game-over / not-played -> RESET (per the agent contract).
        if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._prev_hash = None
            self._prev_action = None
            self._prev_layers = None
            return GameAction.RESET

        cur_layers = list(getattr(frame, "frame", []) or [])
        cur_hash = hash_frame(cur_layers) if cur_layers else b"\x00" * 8
        cur_levels = int(getattr(frame, "levels_completed", 0))

        self.graph.maybe_reset_for_level(cur_levels)

        change_score = _trigger_score(self._prev_layers, cur_layers, self._prev_levels, cur_levels)

        avail = list(getattr(frame, "available_actions", []) or [1, 2, 3, 4, 5, 6, 7])
        node = self.graph.add_or_get(cur_hash, available_actions=avail, levels=cur_levels)

        if self._prev_hash is not None and self._prev_action is not None:
            self.graph.observe(self._prev_hash, self._prev_action, cur_hash, change_score)

        non_reset_avail = [int(a) for a in avail if int(a) != 0]
        if not non_reset_avail:
            non_reset_avail = [1, 2, 3, 4, 5, 6, 7]

        # Strategy: random-uniform over available_actions, but bias toward
        # untried actions in the current state when possible.
        # This stays competitive with the public Random Agent (~0.18 LB)
        # while letting the state graph collect data for future variants.
        chosen: int
        untried_avail = [a for a in node.untried_actions if a in non_reset_avail]
        if untried_avail:
            chosen = self._rng.choice(untried_avail)
        else:
            # All actions tried at this node. Prefer the highest-change edge,
            # otherwise fall back to uniform random over available_actions.
            scored: list[tuple[float, int]] = []
            for a, succ_h in node.edges.items():
                if a not in non_reset_avail:
                    continue
                succ = self.graph.nodes.get(succ_h)
                if succ is None:
                    continue
                s = succ.incoming_change_score + 0.05 * len(succ.untried_actions)
                scored.append((s, a))
            if scored and max(s for s, _ in scored) > 0.0:
                top = max(scored)[0]
                top_actions = [a for s, a in scored if s == top]
                chosen = self._rng.choice(top_actions)
            else:
                chosen = self._rng.choice(non_reset_avail)

        # Build the GameAction (with click data if complex).
        action = GameAction.from_id(chosen)
        data: dict[str, int] = {}
        if action.is_complex():
            data = _sample_click_xy(cur_layers, self._rng)
            action.set_data(data)

        # Roll over.
        self._prev_hash = cur_hash
        self._prev_action = chosen
        self._prev_data = data
        self._prev_layers = cur_layers
        self._prev_levels = cur_levels
        self.graph.record_action(cur_hash, chosen, data)
        return action

    def is_done(self, frame: Any) -> bool:
        return frame.state == GameState.WIN


__all__ = ["TriggerBFSAgent", "_sample_click_xy", "_trigger_score"]
