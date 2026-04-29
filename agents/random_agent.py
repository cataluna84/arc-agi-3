"""Uniform-random agent — expected LB ~0.18 (per Random Agent public notebook).

Picks a random action from the env's `available_actions` list. Treats
NOT_PLAYED / GAME_OVER as a signal to send RESET. ACTION6 (complex / click)
gets random (x, y) in [0, 63] x [0, 63].
"""

from __future__ import annotations

import random
from typing import Any

from . import GameAction, GameState


class RandomAgent:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def choose_action(self, frame: Any) -> GameAction:
        # If game hasn't started or just ended, send RESET
        if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        # Otherwise pick a random action from what the env says is allowed.
        # Fall back to all non-RESET actions if available_actions is empty.
        avail = [int(a) for a in (frame.available_actions or []) if int(a) != 0]
        if not avail:
            avail = [1, 2, 3, 4, 5, 6, 7]
        action = GameAction.from_id(self._rng.choice(avail))

        # Complex actions (ACTION6) need (x, y) coords on the 64x64 grid.
        if action.is_complex():
            action.set_data(
                {
                    "x": self._rng.randint(0, 63),
                    "y": self._rng.randint(0, 63),
                }
            )
        return action

    def is_done(self, frame: Any) -> bool:
        # Stop only on WIN; let the runner decide what to do on GAME_OVER
        # (the agent will issue RESET on the next call and try again).
        return frame.state == GameState.WIN
