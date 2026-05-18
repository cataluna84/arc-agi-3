"""trigger_bfs_mcts_agent.py - MCTS over TriggerBFS state-graph (exp011).

Option B of the D17+ MCTS-over-StateGraph experiment. Builds on top of:
  - agents.state_graph   (StateGraph, StateNode, hash_frame)
  - agents.frame_segmenter (segment_frame, identify_status_bars, tiers)
  - agents.trigger_bfs_agent helpers (_trigger_score for change reward)

Algorithm
---------
At every step, the agent maintains MCTS statistics over (action_id, click_bucket)
keys at each state node, separate from but parallel to the StateGraph node
record. Selection uses UCB1 over Q = mean change_score:

    UCB1(s, a) = Q(s, a) + c * sqrt(ln(N(s)) / n(s, a))

with Q = +inf for unvisited (s, a) so brand-new edges are tried first.

There is no rollout; in this CPU-bound deterministic env one env step is one
node visit. "Simulation + backup" simply means executing the chosen action
via the gateway on the next call and propagating the resulting change_score
into the prior state's MCTS stats. Level transitions clear both the
StateGraph and the MCTS stats (gotcha #4).

Click coords for ACTION6 are bucketed: indices 0..K-1 from the segmenter top
candidates (tier 0 then tier 1, area-sorted), then K..K+4 from deterministic
fallback points, then K+5 = "uniform random" via a step-seeded RNG.

Expected LB band: 0.18 - 0.30 (vs trigger-bfs v1 = 0.12, FORGE = 0.24).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import GameAction, GameState
from .state_graph import StateGraph, hash_frame

if TYPE_CHECKING:
    from collections.abc import Iterable

# Top-K segmenter candidates considered (tier 0 + tier 1). Matches Phase-1
# Qwen Path B's _DEFAULT_CANDIDATES upper bound (=14) so the two agents
# explore the same coordinate space at parity.
_MAX_SEGMENT_CANDIDATES = 8

# Deterministic ACTION6 fallback coordinates. Appended after segmenter
# candidates so the agent has SOMETHING to bucket through even when
# frame_segmenter surfaces no usable tier-0..1 segments. Total = 5.
_ACTION6_FALLBACK_COORDS: tuple[tuple[int, int], ...] = (
    (32, 32),  # center
    (16, 16),  # NW quadrant
    (48, 16),  # NE quadrant
    (16, 48),  # SW quadrant
    (48, 48),  # SE quadrant
)

# Total click buckets: K segmenter + 5 fallback + 1 random uniform = 14.
_RANDOM_BUCKET_OFFSET = _MAX_SEGMENT_CANDIDATES + len(_ACTION6_FALLBACK_COORDS)
_MAX_BUCKETS = _RANDOM_BUCKET_OFFSET + 1  # 14


@dataclass
class MCTSStat:
    """Per-(action_id, click_bucket) MCTS counter."""

    n_visits: int = 0
    sum_change_score: float = 0.0

    @property
    def q(self) -> float:
        if self.n_visits == 0:
            return 0.0
        return self.sum_change_score / self.n_visits


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
    """delta_pixels + 5*delta_levels + 2*new_colors. Same formula as
    trigger_bfs_agent._trigger_score (kept inline so we never need to
    import the other agent's private API)."""
    if prev_layers is None or next_layers is None:
        return 0.0
    p = _to_ndarray(prev_layers[-1])
    n = _to_ndarray(next_layers[-1])
    if p is None or n is None or p.shape != n.shape:
        return float(5 * (next_levels - prev_levels))
    delta_pixels = float((p != n).sum())
    new_colors = float(len({int(v) for v in n.flat} - {int(v) for v in p.flat}))
    delta_levels = float(next_levels - prev_levels)
    return delta_pixels + 5.0 * delta_levels + 2.0 * new_colors


def _segment_click_candidates(layers: list | None) -> list[dict]:
    """Return up to ``_MAX_BUCKETS`` ACTION6 click candidates.

    Each candidate is a dict ``{"label": "C<i>", "x": int, "y": int,
    "tier": int, "segment_id": int}``. Tier 0..1 segments come first
    (area-desc, dedup'd by coord), then deterministic fallbacks, then
    a placeholder "uniform random" candidate flagged with segment_id=-1
    whose (x, y) is filled at call time. Wrapped in try/except → empty
    list per gotcha #17.
    """
    results: list[dict] = []
    seen_coords: set[tuple[int, int]] = set()
    try:
        import numpy as np

        from . import frame_segmenter as fs

        if not layers:
            return _fallback_candidates_only()
        layer = layers[-1]
        grid = layer if isinstance(layer, np.ndarray) else np.asarray(layer, dtype=np.uint8)
        if grid.ndim != 2:
            return _fallback_candidates_only()

        label_map, segments = fs.segment_frame(grid)
        _sb_mask, sb_groups = fs.identify_status_bars(label_map, segments)
        sb_ids: set[int] = set()
        for g in sb_groups:
            sb_ids.update(g)
        tiers = fs.frame_segments_to_priority_tiers(segments, status_bar_segment_ids=sb_ids)

        frame_pixels = grid.shape[0] * grid.shape[1]
        half = frame_pixels // 2
        rng = random.Random(0)

        for tier_idx in (0, 1):
            tier_sids = [sid for sid in tiers[tier_idx] if segments[sid].area <= half]
            # Largest area first within tier (deterministic).
            tier_sids.sort(key=lambda sid: -segments[sid].area)
            for sid in tier_sids:
                coord = fs.mask_to_click_coords(label_map, sid, rng=rng)
                if coord is None:
                    continue
                xy = (int(coord[0]), int(coord[1]))
                if xy in seen_coords:
                    continue
                seen_coords.add(xy)
                results.append(
                    {
                        "label": f"C{len(results)}",
                        "x": xy[0],
                        "y": xy[1],
                        "tier": tier_idx,
                        "segment_id": int(sid),
                    }
                )
                if len(results) >= _MAX_SEGMENT_CANDIDATES:
                    break
            if len(results) >= _MAX_SEGMENT_CANDIDATES:
                break
    except Exception:  # noqa: S110 - any failure falls through to fallbacks
        pass

    # Append deterministic fallbacks (5).
    for fx, fy in _ACTION6_FALLBACK_COORDS:
        if (fx, fy) in seen_coords:
            continue
        seen_coords.add((fx, fy))
        results.append(
            {
                "label": f"C{len(results)}",
                "x": int(fx),
                "y": int(fy),
                "tier": 4,
                "segment_id": -1,
            }
        )

    # Append the "uniform random" sentinel (x/y filled at use-time).
    results.append(
        {
            "label": f"C{len(results)}",
            "x": -1,
            "y": -1,
            "tier": 5,
            "segment_id": -1,
        }
    )
    return results[:_MAX_BUCKETS]


def _fallback_candidates_only() -> list[dict]:
    """Fallback path: only deterministic + random-uniform buckets."""
    results: list[dict] = []
    for fx, fy in _ACTION6_FALLBACK_COORDS:
        results.append(
            {
                "label": f"C{len(results)}",
                "x": int(fx),
                "y": int(fy),
                "tier": 4,
                "segment_id": -1,
            }
        )
    results.append(
        {
            "label": f"C{len(results)}",
            "x": -1,
            "y": -1,
            "tier": 5,
            "segment_id": -1,
        }
    )
    return results


def _action_bucket_keys(
    avail: Iterable[int],
    candidates: list[dict] | None,
) -> list[tuple[int, int]]:
    """Yield (action_id, bucket) pairs across non-ACTION6 + ACTION6 buckets.

    Non-ACTION6 actions get bucket = -1. ACTION6 yields one key per click
    bucket index 0..len(candidates)-1 (only if 6 is in avail).
    """
    keys: list[tuple[int, int]] = []
    avail_set = {int(a) for a in avail if int(a) != 0}
    for aid in sorted(avail_set):
        if aid == 6:
            continue
        keys.append((aid, -1))
    if 6 in avail_set and candidates:
        keys.extend((6, bucket) for bucket in range(len(candidates)))
    return keys


def _ucb1_score(
    stat: MCTSStat | None,
    total_visits: int,
    c_uct: float,
) -> float:
    """UCB1 over (action, bucket). +inf for unvisited keys."""
    if stat is None or stat.n_visits == 0:
        return math.inf
    if total_visits <= 0:
        return stat.q
    return stat.q + c_uct * math.sqrt(math.log(max(total_visits, 1)) / stat.n_visits)


class TriggerBFSMCTSAgent:
    """MCTS over the trigger-bfs state graph (exp011)."""

    name = "trigger-bfs-mcts"

    def __init__(self, seed: int = 0, c_uct: float = 2.0, **_: Any) -> None:
        self._rng = random.Random(seed)
        self._seed = int(seed)
        self.c_uct = float(c_uct)
        self.graph = StateGraph()
        # MCTS bookkeeping (parallel to StateGraph nodes).
        self._mcts_stats: dict[bytes, dict[tuple[int, int], MCTSStat]] = {}
        self._mcts_total: dict[bytes, int] = {}
        # Per-call rollover state.
        self._prev_hash: bytes | None = None
        self._prev_action_key: tuple[int, int] | None = None
        self._prev_action_id: int | None = None
        self._prev_layers: list | None = None
        self._prev_levels: int = 0
        self._prev_data: dict = {}
        self._step: int = 0

    # ------------------------------------------------------------------
    # MCTS primitives (kept thin so tests can poke them directly)
    # ------------------------------------------------------------------

    def _stat_for(self, node_hash: bytes, key: tuple[int, int]) -> MCTSStat:
        per_node = self._mcts_stats.setdefault(node_hash, {})
        s = per_node.get(key)
        if s is None:
            s = MCTSStat()
            per_node[key] = s
        return s

    def _mcts_select(
        self,
        node_hash: bytes,
        avail: list[int],
        candidates: list[dict],
    ) -> tuple[int, int]:
        keys = _action_bucket_keys(avail, candidates)
        if not keys:
            # No legal action; defensive fallback (any non-RESET).
            return (1, -1)
        total = self._mcts_total.get(node_hash, 0)
        # Stable shuffle so equal-score ties (notably the +inf bulk start)
        # break in a randomized but seed-deterministic order.
        self._rng.shuffle(keys)
        best_key = keys[0]
        best_score = -math.inf
        per_node = self._mcts_stats.get(node_hash)
        for key in keys:
            stat = per_node.get(key) if per_node else None
            score = _ucb1_score(stat, total, self.c_uct)
            if score > best_score:
                best_score = score
                best_key = key
        return best_key

    def _backup(
        self,
        node_hash: bytes,
        key: tuple[int, int],
        change_score: float,
    ) -> None:
        stat = self._stat_for(node_hash, key)
        stat.n_visits += 1
        stat.sum_change_score += float(change_score)
        self._mcts_total[node_hash] = self._mcts_total.get(node_hash, 0) + 1

    # ------------------------------------------------------------------
    # Agent contract
    # ------------------------------------------------------------------

    def choose_action(self, frame: Any) -> GameAction:
        try:
            return self._choose_action_inner(frame)
        except Exception:  # defensive last-resort (gotcha #17)
            avail = list(getattr(frame, "available_actions", []) or [1, 2, 3, 4, 5])
            non_reset = [int(a) for a in avail if int(a) not in (0, 6)] or [1, 2, 3, 4, 5]
            try:
                return GameAction.from_id(self._rng.choice(non_reset))
            except Exception:
                return GameAction.RESET

    def _choose_action_inner(self, frame: Any) -> GameAction:
        if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._prev_hash = None
            self._prev_action_key = None
            self._prev_action_id = None
            self._prev_layers = None
            return GameAction.RESET

        cur_layers = list(getattr(frame, "frame", []) or [])
        cur_hash = hash_frame(cur_layers) if cur_layers else b"\x00" * 8
        cur_levels = int(getattr(frame, "levels_completed", 0))

        # Level guard (gotcha #4 + #19): wipe state graph + MCTS stats.
        # Also drop the cross-level prev_* so we don't backup into stale keys.
        if cur_levels != self._prev_levels and cur_levels >= 0:
            self.graph.reset()
            self.graph.current_levels = cur_levels
            self._mcts_stats.clear()
            self._mcts_total.clear()
            self._prev_hash = None
            self._prev_action_key = None
            self._prev_action_id = None
            self._prev_layers = None

        avail = list(getattr(frame, "available_actions", []) or [1, 2, 3, 4, 5, 6, 7])
        node = self.graph.add_or_get(cur_hash, available_actions=avail, levels=cur_levels)

        # If we have a recorded prev action, observe + backup the transition.
        if self._prev_hash is not None and self._prev_action_key is not None:
            change_score = _trigger_score(
                self._prev_layers, cur_layers, self._prev_levels, cur_levels
            )
            prev_aid = self._prev_action_key[0]
            self.graph.observe(self._prev_hash, prev_aid, cur_hash, change_score)
            self._backup(self._prev_hash, self._prev_action_key, change_score)

        # Build candidates for ACTION6 (lazy: empty list if 6 not avail).
        candidates: list[dict] = (
            _segment_click_candidates(cur_layers) if 6 in {int(a) for a in avail} else []
        )

        # Selection.
        action_id, bucket = self._mcts_select(cur_hash, avail, candidates)
        action = GameAction.from_id(int(action_id))

        data: dict = {}
        if action.is_complex():
            data = self._coords_for_bucket(bucket, candidates)
            action.set_data(data)

        # Roll over.
        self._prev_hash = cur_hash
        self._prev_action_key = (int(action_id), int(bucket))
        self._prev_action_id = int(action_id)
        self._prev_data = data
        self._prev_layers = cur_layers
        self._prev_levels = cur_levels
        self._step += 1
        self.graph.record_action(cur_hash, int(action_id), data)
        # Ensure the chosen action is no longer "untried" at this node
        # (the StateGraph normally drops it on observe(), but we also want
        # immediate drops so MCTS reasons over the right untried set).
        if int(action_id) in node.untried_actions:
            node.untried_actions.discard(int(action_id))
        return action

    def _coords_for_bucket(self, bucket: int, candidates: list[dict]) -> dict[str, int]:
        """Resolve bucket -> {"x", "y"}. Handles random-uniform sentinel."""
        if 0 <= bucket < len(candidates):
            c = candidates[bucket]
            x = int(c.get("x", -1))
            y = int(c.get("y", -1))
            tier = int(c.get("tier", -1))
            if tier == 5 or x < 0 or y < 0:
                # Random-uniform: deterministic per (seed, step).
                r = random.Random(self._seed * 1_000_003 + self._step)
                return {"x": r.randint(0, 63), "y": r.randint(0, 63)}
            return {"x": max(0, min(63, x)), "y": max(0, min(63, y))}
        # Out-of-range bucket -> center fallback.
        return {"x": 32, "y": 32}

    def is_done(self, frame: Any) -> bool:
        return frame.state == GameState.WIN


__all__ = [
    "MCTSStat",
    "TriggerBFSMCTSAgent",
    "_action_bucket_keys",
    "_segment_click_candidates",
    "_trigger_score",
    "_ucb1_score",
]
