"""goose_cnn_agent.py - StochasticGoose-style CNN-frame-change agent.

Learns a per-level CNN that predicts P(action will change frame). At every step:

1. Encode current frame as 16-channel one-hot.
2. Forward pass -> action logits (5) + coord heatmap (64x64).
3. Hierarchical sampling:
   a. Sigmoid action logits and ACTION6's max coord prob give a 6-way
      "expected change" distribution.
   b. Sample an action type by softmax with a temperature; if ACTION6 is
      picked, sample (x, y) from the coord heatmap softmax.
4. Observe the next frame; record (state_hash, action_id, [x, y], frame_changed)
   in the experience buffer.
5. Run K mini-batch BCE updates on the buffer (online training).
6. On level transitions, reset model weights AND buffer (per-level mechanics
   change; gotcha #4).

Falls back to uniform-random with a state-graph dedup when torch is unavailable
(CI / CPU smoke). The state graph is shared with `agents/trigger_bfs_agent.py`
so future agents can reuse the same dedup primitives.

Smoke target (`scripts/goose_cnn_smoke_local.py`):
  - 22-check parity (mirrors trigger_bfs_smoke + qwen_policy_smoke shape).
  - On ls20 with the real SDK + GPU: levels_completed >= 1 within 200 actions.

Expected LB anchor: 0.25 (Stochastic Goose public Kaggle sample). The CNN
training adds frame-change predictive power on top; together with state-graph
dedup we target 0.30+ as a Track G first iteration.
"""

from __future__ import annotations

import random
from typing import Any

from . import GameAction, GameState
from .goose_cnn_model import (
    GRID_SIZE,
    NUM_SIMPLE_ACTIONS,
    GooseCNNPredictor,
    hash_frame_grid,
)
from .state_graph import StateGraph, hash_frame


def _to_ndarray(layer: Any) -> Any:
    """Best-effort coercion of a frame layer to a numpy.ndarray."""
    try:
        import numpy as np

        if isinstance(layer, np.ndarray):
            return layer
        return np.asarray(layer, dtype=np.uint8)
    except Exception:
        return None


def _layers_to_grid(layers: list | None) -> Any:
    """Last layer of the frame -> 2D numpy grid (or None if not available)."""
    if not layers:
        return None
    grid = _to_ndarray(layers[-1])
    if grid is None or getattr(grid, "ndim", 0) != 2:
        return None
    return grid


def _frame_changed(prev_layers: list | None, next_layers: list | None) -> bool:
    """Return True iff the visible last-layer grid changed at all."""
    p = _layers_to_grid(prev_layers)
    n = _layers_to_grid(next_layers)
    if p is None or n is None:
        return False
    if p.shape != n.shape:
        return True
    try:
        return bool((p != n).any())
    except Exception:
        return False


def _softmax_sample_2d(
    probs_2d: Any,
    rng: random.Random,
    temperature: float = 1.0,
    non_bg_mask: Any = None,
) -> tuple[int, int]:
    """Sample (x, y) from a 64x64 prob map (after softmax-normalization).

    If `non_bg_mask` (64x64 binary, sum > 0) is supplied, sampling is *restricted*
    to non-background pixels (mirrors trigger_bfs's saliency-tier-0 click). The
    learned heatmap is still used to weight among them, so a trained CNN can
    re-rank the candidates. When `non_bg_mask` is None or all-zero, sample from
    the full 64x64 grid (uniform fallback).
    """
    import numpy as np

    arr = np.asarray(probs_2d, dtype=np.float64)
    if arr.shape != (GRID_SIZE, GRID_SIZE):
        return (rng.randint(0, GRID_SIZE - 1), rng.randint(0, GRID_SIZE - 1))
    logits = np.log(np.clip(arr, 1e-6, 1.0)) / max(temperature, 1e-3)
    if non_bg_mask is not None:
        m = np.asarray(non_bg_mask, dtype=np.float64)
        if m.shape == (GRID_SIZE, GRID_SIZE) and m.sum() > 0:
            logits = np.where(m > 0, logits, -1e9)
    logits = logits - logits.max()
    p = np.exp(logits)
    p = p / max(p.sum(), 1e-12)
    flat = p.flatten()
    cum = np.cumsum(flat)
    r = rng.random()
    idx = int(np.searchsorted(cum, r))
    idx = min(idx, GRID_SIZE * GRID_SIZE - 1)
    y, x = divmod(idx, GRID_SIZE)
    return (int(x), int(y))


def _non_bg_mask(layers: list | None) -> Any:
    """64x64 binary mask: 1.0 where pixel != background color (mode of grid)."""
    import numpy as np

    grid = _layers_to_grid(layers)
    if grid is None:
        return None
    try:
        bg = int(np.bincount(grid.flatten(), minlength=16).argmax())
        return (grid != bg).astype(np.float64)
    except Exception:
        return None


def _sample_click_xy_fallback(layers: list | None, rng: random.Random) -> dict[str, int]:
    """Fallback: pick (x, y) from non-background pixels; uniform if none."""
    grid = _layers_to_grid(layers)
    if grid is None:
        return {"x": rng.randint(0, GRID_SIZE - 1), "y": rng.randint(0, GRID_SIZE - 1)}
    try:
        import numpy as np

        bg = int(np.bincount(grid.flatten(), minlength=16).argmax())
        ys, xs = np.where(grid != bg)
        if len(xs) > 0:
            idx = rng.randrange(len(xs))
            return {"x": int(xs[idx]), "y": int(ys[idx])}
    except Exception:  # noqa: S110 - any failure falls back to uniform sampling
        pass
    return {"x": rng.randint(0, GRID_SIZE - 1), "y": rng.randint(0, GRID_SIZE - 1)}


def _action_logits_to_six(action_probs: Any, coord_probs: Any) -> Any:
    """Map (5 action probs, 64x64 coord probs) -> 6-d combined preference.

    The 6-th slot is the *max* of the coord heatmap (best click).
    """
    import numpy as np

    six = np.zeros((6,), dtype=np.float32)
    six[:NUM_SIMPLE_ACTIONS] = np.asarray(action_probs, dtype=np.float32)[:NUM_SIMPLE_ACTIONS]
    six[NUM_SIMPLE_ACTIONS] = float(np.max(coord_probs)) if coord_probs is not None else 0.5
    return six


def _softmax_sample_1d(
    logits: Any,
    avail_mask: Any,
    rng: random.Random,
    temperature: float = 1.0,
) -> int:
    """Sample an index in [0..5] from masked logits (logits[i] = score for action i+1)."""
    import numpy as np

    a = np.asarray(logits, dtype=np.float64)
    m = np.asarray(avail_mask, dtype=np.float64)
    a = a / max(temperature, 1e-3)
    a = a - a.max()
    p = np.exp(a) * m
    s = p.sum()
    if s <= 0:
        idxs = np.where(m > 0)[0]
        if len(idxs) == 0:
            return 0
        return int(rng.choice(idxs.tolist()))
    p = p / s
    cum = np.cumsum(p)
    r = rng.random()
    idx = int(np.searchsorted(cum, r))
    return min(idx, len(p) - 1)


class GooseCNNAgent:
    """StochasticGoose-style CNN-driven agent with online BCE training."""

    name = "goose-cnn"

    def __init__(
        self,
        seed: int = 0,
        train_every: int = 4,
        train_steps_per_call: int = 4,
        action_temperature: float = 1.0,
        coord_temperature: float = 0.7,
        device: str | None = None,
        max_buffer: int = 50000,
        **_: Any,
    ) -> None:
        self._rng = random.Random(seed)
        self._train_every = max(1, int(train_every))
        self._train_steps = max(1, int(train_steps_per_call))
        self._action_temp = float(action_temperature)
        self._coord_temp = float(coord_temperature)
        self.predictor = GooseCNNPredictor(seed=seed, device=device, max_buffer=max_buffer)
        self.graph = StateGraph()
        self._step_count: int = 0
        self._prev_hash: bytes | None = None
        self._prev_action: int | None = None
        self._prev_xy: tuple[int, int] | None = None
        self._prev_layers: list | None = None
        self._prev_levels: int = 0
        self._grids_by_hash: dict[bytes, Any] = {}

    def _on_level_transition(self) -> None:
        """Reset model weights + buffer + graph; gotcha #4."""
        self.predictor.reset(seed=self._rng.randint(0, 1_000_000))
        self.graph = StateGraph()
        self._grids_by_hash.clear()
        self._prev_hash = None
        self._prev_action = None
        self._prev_xy = None
        self._prev_layers = None
        self._step_count = 0

    def choose_action(self, frame: Any) -> GameAction:
        # Game-over / not-played -> RESET.
        if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._prev_hash = None
            self._prev_action = None
            self._prev_xy = None
            self._prev_layers = None
            return GameAction.RESET

        cur_layers = list(getattr(frame, "frame", []) or [])
        cur_hash = hash_frame(cur_layers) if cur_layers else b"\x00" * 8
        cur_levels = int(getattr(frame, "levels_completed", 0))

        if cur_levels != self._prev_levels:
            self._on_level_transition()
            self._prev_levels = cur_levels

        # Bookkeeping: register the state in the graph + grids_by_hash.
        avail = list(getattr(frame, "available_actions", []) or [1, 2, 3, 4, 5, 6, 7])
        self.graph.add_or_get(cur_hash, available_actions=avail, levels=cur_levels)

        cur_grid = _layers_to_grid(cur_layers)
        if cur_grid is not None:
            grid_key = hash_frame_grid(cur_grid)
            self._grids_by_hash[grid_key] = cur_grid

        # Observation: did the last action change anything?
        if self._prev_hash is not None and self._prev_action is not None:
            changed = _frame_changed(self._prev_layers, cur_layers)
            self.graph.observe(
                self._prev_hash,
                self._prev_action,
                cur_hash,
                change_score=1.0 if changed else 0.0,
            )
            if self._prev_layers is not None:
                prev_grid = _layers_to_grid(self._prev_layers)
                if prev_grid is not None:
                    prev_grid_key = hash_frame_grid(prev_grid)
                    self._grids_by_hash[prev_grid_key] = prev_grid
                    self.predictor.buffer.add(
                        prev_grid_key,
                        self._prev_action,
                        changed,
                        click_xy=self._prev_xy,
                    )

        # Online training every K steps.
        if (
            self._step_count > 0
            and self._step_count % self._train_every == 0
            and self.predictor.available
        ):
            self.predictor.update(self._train_steps, self._grids_by_hash)

        # Predict action / coord probs from the current frame.
        if cur_grid is not None:
            preds = self.predictor.predict(cur_grid)
            ap = preds["action_probs"]
            cp = preds["coord_probs"]
        else:
            import numpy as np

            ap = np.full((NUM_SIMPLE_ACTIONS,), 0.5, dtype="float32")
            cp = np.full((GRID_SIZE, GRID_SIZE), 0.5, dtype="float32")

        # Build availability mask over [ACTION1..ACTION5, ACTION6].
        non_reset = [int(a) for a in avail if int(a) != 0]
        if not non_reset:
            non_reset = [1, 2, 3, 4, 5, 6, 7]
        # ACTION7 (Undo) is dropped from the head (rarely available; treated as
        # uniform fallback on the same head structure to avoid biasing it).
        avail_mask = [1.0 if (i + 1) in non_reset else 0.0 for i in range(NUM_SIMPLE_ACTIONS)] + [
            1.0 if 6 in non_reset else 0.0
        ]

        # Strict untried-first explore (trigger_bfs-style). The CNN only feeds
        # in for (a) ACTION6 coord sampling, and (b) the all-tried fallback.
        # Re-sampling tried actions costs a step in the 100-action mock; the
        # learned model adds value only if it can predict outcomes BETTER than
        # "I have already observed this exact (state, action) pair", which it
        # cannot in expectation. So we let direct observation dominate.
        node = self.graph.nodes.get(cur_hash)
        untried_set = set(node.untried_actions) if node is not None else set()
        untried_avail = [a for a in untried_set if a in non_reset]

        nb_mask = _non_bg_mask(cur_layers)
        if untried_avail:
            chosen_action_id = self._rng.choice(untried_avail)
            xy: tuple[int, int] | None = None
            if chosen_action_id == 6:
                xy = _softmax_sample_2d(
                    cp, self._rng, temperature=self._coord_temp, non_bg_mask=nb_mask
                )
        else:
            # All actions tried at this state. Combine the CNN's predicted
            # action probs with the empirical change scores recorded by the
            # state graph (mirrors trigger_bfs's "highest-change edge" tie-
            # breaker). When the CNN is uniform (cold start) the empirical
            # term dominates; when the CNN has trained it adds extrapolation.
            six_logits = _action_logits_to_six(ap, cp)
            if node is not None:
                for a, succ_h in node.edges.items():
                    if a not in non_reset:
                        continue
                    succ = self.graph.nodes.get(succ_h)
                    if succ is None:
                        continue
                    boost = float(succ.incoming_change_score)
                    if 1 <= a <= NUM_SIMPLE_ACTIONS:
                        six_logits[a - 1] += boost
                    elif a == 6:
                        six_logits[NUM_SIMPLE_ACTIONS] += boost
            idx = _softmax_sample_1d(
                six_logits, avail_mask, self._rng, temperature=self._action_temp
            )
            if idx < NUM_SIMPLE_ACTIONS:
                chosen_action_id = idx + 1
                xy = None
            elif 6 in non_reset:
                chosen_action_id = 6
                xy = _softmax_sample_2d(
                    cp, self._rng, temperature=self._coord_temp, non_bg_mask=nb_mask
                )
            else:
                chosen_action_id = self._rng.choice(non_reset)
                xy = None
                if chosen_action_id == 6:
                    xy = _softmax_sample_2d(
                        cp, self._rng, temperature=self._coord_temp, non_bg_mask=nb_mask
                    )

        action = GameAction.from_id(chosen_action_id)
        data: dict[str, int] = {}
        if action.is_complex():
            if xy is None:
                fallback = _sample_click_xy_fallback(cur_layers, self._rng)
                xy = (fallback["x"], fallback["y"])
            data = {"x": int(xy[0]), "y": int(xy[1])}
            action.set_data(data)

        # Roll over.
        self._prev_hash = cur_hash
        self._prev_action = chosen_action_id
        self._prev_xy = xy
        self._prev_layers = cur_layers
        self._prev_levels = cur_levels
        self.graph.record_action(cur_hash, chosen_action_id, data)
        self._step_count += 1
        return action

    def is_done(self, frame: Any) -> bool:
        return frame.state == GameState.WIN


__all__ = [
    "GooseCNNAgent",
    "_action_logits_to_six",
    "_frame_changed",
    "_non_bg_mask",
    "_sample_click_xy_fallback",
    "_softmax_sample_1d",
    "_softmax_sample_2d",
]
