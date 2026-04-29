"""Local stand-in for the official `agents.agent.Agent` base class.

Ash's notebook (and the bundled ARC-AGI-3-Agents harness) imports
`from agents.agent import Agent`. On Kaggle that import resolves to the
official harness package; locally we provide this minimal stub so the
verbatim Ash source can be imported and instantiated by our local_runner
without dragging in the entire harness.

We deliberately match the surface area of the upstream Agent class
(github.com/arcprize/ARC-AGI-3-Agents agents/agent.py) so that subclasses
written against either copy work without modification.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class Agent:
    """Minimal base for ARC-AGI-3 agents (mirrors the official harness API).

    Subclasses must override `choose_action(frames, latest_frame) -> GameAction`
    and `is_done(frames, latest_frame) -> bool`. Local invocation:

        agent = MyAgent(card_id="", game_id="ls20",
                        agent_name="ash", ROOT_URL="",
                        record=False, arc_env=env)
    """

    MAX_ACTIONS: int = 80  # to avoid infinite loops; subclasses may override

    action_counter: int = 0
    timer: float = 0.0
    agent_name: str
    card_id: str
    game_id: str
    guid: str
    frames: list

    headers: dict
    arc_env: Any  # arc_agi.EnvironmentWrapper at runtime

    def __init__(
        self,
        card_id: str = "",
        game_id: str = "",
        agent_name: str = "",
        ROOT_URL: str = "",
        record: bool = False,
        arc_env: Any | None = None,
        tags: list | None = None,
        **kwargs: Any,  # tolerate any extra args the upstream version may add
    ) -> None:
        self.ROOT_URL = ROOT_URL
        self.card_id = card_id
        self.game_id = game_id
        self.guid = ""
        self.agent_name = agent_name or self.__class__.__name__.lower()
        self.tags = tags or []
        self.frames = []  # subclasses populate this; the official harness
        # initializes with [FrameData(levels_completed=0)]
        # but Ash's MyAgent doesn't depend on that detail.
        self._cleanup = True
        # Note: do NOT set self.recorder here. Upstream Ash code does
        #     `if hasattr(self, "recorder") and not self.is_playback: ...`
        # If we set recorder = None, hasattr is True and the code crashes.
        # The recorder attribute is only set by start_recording() when
        # record=True (which we don't use in local mode).
        self.headers = {}
        self.arc_env = arc_env

    @property
    def state(self):
        return self.frames[-1].state if self.frames else None

    @property
    def levels_completed(self) -> int:
        return getattr(self.frames[-1], "levels_completed", 0) if self.frames else 0

    @property
    def seconds(self) -> float:
        if not self.timer:
            return 0.0
        return (time.time() - self.timer) * 100 // 1 / 100

    @property
    def fps(self) -> float:
        if self.action_counter == 0 or not self.seconds:
            return 0.0
        return round(self.action_counter / max(self.seconds, 0.1), 2)

    @property
    def is_playback(self) -> bool:
        return False

    @property
    def name(self) -> str:
        n = self.__class__.__name__.lower()
        return f"{self.game_id}.{n}" if self.game_id else n

    def append_frame(self, frame: Any) -> None:
        """Append a frame; Ash's MyAgent overrides this to cap the buffer."""
        self.frames.append(frame)
        guid = getattr(frame, "guid", None)
        if guid:
            self.guid = guid

    def cleanup(self, scorecard: Any | None = None) -> None:
        """Called once at end-of-game; the local_runner doesn't need it."""
        del scorecard

    # --- abstract surface ---------------------------------------------------

    def is_done(self, frames: list, latest_frame: Any) -> bool:  # pragma: no cover
        raise NotImplementedError

    def choose_action(self, frames: list, latest_frame: Any):  # pragma: no cover
        raise NotImplementedError
