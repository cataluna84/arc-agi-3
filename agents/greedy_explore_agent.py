"""Greedy-explore agent — picks actions that historically caused frame change.

Tracks per-action "did the frame change" empirical probability across the run
and picks the action with highest empirical change rate, with epsilon-greedy
fallback. ACTION6 click coords are sampled uniformly from [0, 63]^2.

Expected LB: between random (~0.18) and Just-Explore (0.19); useful primarily
as a sanity check that the local_runner harness works end-to-end with a
non-trivial agent that consumes observations.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Any

from . import GameAction, GameState


def _hash_frame(frame_layers: Any) -> str:
    """Hash a list of 64x64 layers (numpy arrays or list-of-list-of-int)."""
    if frame_layers is None:
        return ""
    h = hashlib.blake2b(digest_size=16)
    for layer in frame_layers:
        # numpy arrays expose tobytes(); plain lists need to be flattened
        tobytes = getattr(layer, "tobytes", None)
        if callable(tobytes):
            h.update(tobytes())
        else:
            for row in layer:
                h.update(bytes(int(v) % 256 for v in row))
    return h.hexdigest()


class GreedyExploreAgent:
    name = "greedy-explore"

    def __init__(self, seed: int = 0, epsilon: float = 0.2) -> None:
        self._rng = random.Random(seed)
        self._epsilon = epsilon
        # action_id -> [n_changes, n_total]
        self._stats: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        self._prev_hash: str = ""
        self._prev_action: int = -1

    def choose_action(self, frame: Any) -> GameAction:
        # Update stats from the *previous* (action, prev_hash, current_hash) triple.
        cur_hash = _hash_frame(getattr(frame, "frame", None))
        if self._prev_action >= 0 and self._prev_hash:
            self._stats[self._prev_action][1] += 1
            if cur_hash != self._prev_hash:
                self._stats[self._prev_action][0] += 1

        # If game hasn't started or just ended, send RESET.
        if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._prev_hash = cur_hash
            self._prev_action = int(GameAction.RESET)
            return GameAction.RESET

        avail = [int(a) for a in (frame.available_actions or []) if int(a) != 0]
        if not avail:
            avail = [1, 2, 3, 4, 5, 6, 7]

        # Epsilon-greedy
        if self._rng.random() < self._epsilon:
            chosen = self._rng.choice(avail)
        else:
            scored: list[tuple[float, int]] = []
            for aid in avail:
                ch, tot = self._stats[aid]
                rate = ch / tot if tot > 0 else 0.5  # optimistic prior
                scored.append((rate, aid))
            self._rng.shuffle(scored)  # break ties at random
            scored.sort(key=lambda t: t[0], reverse=True)
            chosen = scored[0][1]

        action = GameAction.from_id(chosen)
        if action.is_complex():
            action.set_data(
                {
                    "x": self._rng.randint(0, 63),
                    "y": self._rng.randint(0, 63),
                }
            )
        self._prev_hash = cur_hash
        self._prev_action = chosen
        return action

    def is_done(self, frame: Any) -> bool:
        return frame.state == GameState.WIN
