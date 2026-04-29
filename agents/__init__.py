"""Agent implementations for ARC-AGI-3.

Contract (matches the real arc_agi 0.9.x / arcengine SDK):

    class MyAgent:
        name: str = "<short-id-for-logs>"

        def __init__(self, seed: int = 0) -> None: ...

        def choose_action(self, frame) -> GameAction:
            '''Return a GameAction enum (not a string, not a raw int).
            For complex actions (ACTION6), call action.set_data({"x":..,"y":..})
            before returning.'''
            ...

        def is_done(self, frame) -> bool:                  # optional
            '''Return True to terminate the episode early
            (the runner also stops on state in {WIN, GAME_OVER}).'''
            ...

`frame` quacks like `arcengine.FrameDataRaw`:
    .game_id           : str
    .state             : GameState   # NOT_PLAYED | NOT_FINISHED | WIN | GAME_OVER
    .levels_completed  : int
    .win_levels        : int
    .available_actions : list[int]   # subset of {0..7}; 0 = RESET
    .frame             : list[np.ndarray]  # one 64x64 array per layer
    .guid              : str | None
    .full_reset        : bool

In offline mock mode (no SDK install) the runner constructs `MockFrame`
objects with the same attribute names so agents work unchanged.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# When the SDK is installed (post `scripts/install_arc_agi_sdk.py`), use the
# real enums. Otherwise we expose tiny stand-ins so agents and the runner can
# still be imported and unit-tested offline.
try:
    from arcengine import GameAction, GameState  # type: ignore

    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

    from enum import IntEnum

    class GameAction(IntEnum):  # type: ignore[no-redef]
        RESET = 0
        ACTION1 = 1
        ACTION2 = 2
        ACTION3 = 3
        ACTION4 = 4
        ACTION5 = 5
        ACTION6 = 6
        ACTION7 = 7

        def is_simple(self) -> bool:
            return self.value in (0, 1, 2, 3, 4, 5, 7)

        def is_complex(self) -> bool:
            return self.value == 6

        def set_data(self, data: dict) -> None:  # mock no-op
            self._data = data  # type: ignore[attr-defined]

        @classmethod
        def from_id(cls, idx: int) -> GameAction:
            return cls(idx)

    class GameState(enum.StrEnum):  # type: ignore[no-redef]
        NOT_PLAYED = "NOT_PLAYED"
        NOT_FINISHED = "NOT_FINISHED"
        WIN = "WIN"
        GAME_OVER = "GAME_OVER"


@dataclass
class MockFrame:
    """Duck-typed FrameDataRaw stand-in used by the local_runner mock backend."""

    game_id: str = ""
    state: Any = GameState.NOT_FINISHED  # GameState
    levels_completed: int = 0
    win_levels: int = 1
    available_actions: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])
    frame: list[Any] = field(default_factory=list)  # list of 64x64 grids (lists or ndarrays)
    guid: str | None = None
    full_reset: bool = False


@runtime_checkable
class Agent(Protocol):
    """Minimal duck-typed agent interface used by experiments/local_runner.py."""

    name: str

    def choose_action(self, frame: Any) -> GameAction: ...


__all__ = [
    "SDK_AVAILABLE",
    "Agent",
    "GameAction",
    "GameState",
    "MockFrame",
]
