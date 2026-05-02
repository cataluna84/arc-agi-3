"""qwen_policy_agent.py - Option A reboot of exp004 (POSTMORTEM 4.1-4.5).

Five compounding fixes applied vs original `agents/qwen_agent.py`:

  F1 (cause 4.1, deterministic decode): sampling temperature 0.7 + nucleus 0.9
     via QwenBackbone, *plus* StateGraph dedup -- when re-entering an already-
     visited state, the LLM's reply is filtered down to its untried actions.
  F2 (cause 4.2, vision prefill): vLLM with prefix caching (system prompt is
     identical across turns -> ~10x speedup) + frame upscale 8 -> 4
     (256 -> 64 patch tokens).
  F3 (cause 4.3, no state memory): wraps `agents.state_graph.StateGraph`,
     so every action is informed by visit_count / untried_actions / edges
     accumulated since the last level transition.
  F4 (cause 4.4, no change feedback): `ChangeLog` annotates each history
     entry with `changed (+N px)` or `LEVEL UP +k`. The model can therefore
     learn (in-context) which actions actually move the world in this game.
  F5 (cause 4.5, no ACTION6 path): if the model picks ACTION6, the parser
     extracts coords; if absent, falls back to bg-aware sampling
     (`_sample_click_xy` from `trigger_bfs_agent`).

Local-only smoke (no GPU, no model load):
    .venv/bin/python scripts/qwen_policy_smoke_local.py

Kaggle dev kernel use (vLLM bundled separately):
    QWEN_RUNTIME=vllm \\
    QWEN_MODEL_PATH=/kaggle/input/datasets/cataluna84/qwen3-6-35b-a3b-bf16 \\
    .venv/bin/python experiments/local_runner.py \\
        --agent agents.qwen_policy_agent:QwenPolicyAgent \\
        --use-sdk --games ls20,vc33,ft09 --max-actions 200
"""

from __future__ import annotations

import os
import random
from typing import Any

from . import GameAction, GameState
from .qwen_backbone import (
    ChangeLog,
    QwenBackbone,
    parse_action_id,
    parse_coords,
    render_frame_image,
)
from .state_graph import StateGraph, hash_frame
from .trigger_bfs_agent import _sample_click_xy, _trigger_score


def build_policy_messages(
    frame: Any,
    history: ChangeLog,
    image: Any,
    untried_actions: list[int],
    visit_count: int,
) -> list[dict]:
    """Build the chat-template messages for Option A (policy) mode."""
    avail = [int(a) for a in (getattr(frame, "available_actions", None) or [])]
    avail_str = ", ".join(f"ACTION{a}" for a in avail) if avail else "ACTION1..ACTION7"
    untried_str = (
        ", ".join(f"ACTION{a}" for a in sorted(untried_actions))
        if untried_actions
        else "(all tried at this state)"
    )
    state_str = getattr(getattr(frame, "state", None), "name", "?")
    levels = getattr(frame, "levels_completed", 0)
    win_levels = getattr(frame, "win_levels", 1)

    system = (
        "You are an ARC-AGI-3 game agent. Reply with ONE action only, NO "
        "explanation. Format: 'ACTIONn' on one line (e.g. ACTION3). For "
        "ACTION6 add '(x, y)' both integers in [0, 63]. PREFER untried "
        "actions in the current state. Avoid actions that recently caused "
        "no-change."
    )
    user_text = (
        f"state={state_str} level={levels}/{win_levels} visit_count={visit_count}\n"
        f"Available: {avail_str}\n"
        f"Untried at this state: {untried_str}\n"
        f"Recent history (oldest->newest):\n{history.render()}\n"
        "Output ONE action only."
    )
    user_content_parts: list[dict] = []
    if image is not None:
        user_content_parts.append({"type": "image", "image": image})
    user_content_parts.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content_parts},
    ]


def _layer0(layers: list | None) -> Any:
    return layers[0] if layers else None


def _grid_pixel_diff(prev_layers: list | None, next_layers: list | None) -> int:
    """Return the count of changed cells between previous and current frame."""
    p = _layer0(prev_layers)
    n = _layer0(next_layers)
    if p is None or n is None:
        return 0
    try:
        import numpy as np

        pa = np.asarray(p, dtype=np.uint8)
        na = np.asarray(n, dtype=np.uint8)
        if pa.shape != na.shape:
            return 0
        return int((pa != na).sum())
    except ImportError:
        return sum(
            1
            for ra, rb in zip(p, n, strict=False)
            for va, vb in zip(ra, rb, strict=False)
            if va != vb
        )


class QwenPolicyAgent:
    """Option A: Qwen-as-policy with all 5 POSTMORTEM fixes applied."""

    name = "qwen_policy"

    def __init__(
        self,
        seed: int = 0,
        backbone: QwenBackbone | None = None,
        history_len: int | None = None,
        **_: Any,
    ) -> None:
        self._rng = random.Random(seed)
        self.backbone = backbone if backbone is not None else QwenBackbone()
        self.history = ChangeLog(
            capacity=history_len
            if history_len is not None
            else int(os.environ.get("QWEN_HISTORY_LEN") or 8)
        )
        self.graph = StateGraph()
        self._prev_hash: bytes | None = None
        self._prev_action: int | None = None
        self._prev_layers: list | None = None
        self._prev_levels: int = 0

    def _is_terminal(self, frame: Any) -> bool:
        return getattr(frame, "state", None) in (GameState.NOT_PLAYED, GameState.GAME_OVER)

    def choose_action(self, frame: Any) -> GameAction:
        if self._is_terminal(frame):
            self._prev_hash = None
            self._prev_action = None
            self._prev_layers = None
            return GameAction.RESET

        cur_layers = list(getattr(frame, "frame", []) or [])
        cur_hash = hash_frame(cur_layers) if cur_layers else b"\x00" * 8
        cur_levels = int(getattr(frame, "levels_completed", 0))

        self.graph.maybe_reset_for_level(cur_levels)
        change_score = _trigger_score(self._prev_layers, cur_layers, self._prev_levels, cur_levels)

        avail = [int(a) for a in (getattr(frame, "available_actions", []) or [])]
        non_reset_avail = [a for a in avail if a != 0] or [1, 2, 3, 4, 5, 6, 7]
        node = self.graph.add_or_get(cur_hash, available_actions=avail, levels=cur_levels)
        if self._prev_hash is not None and self._prev_action is not None:
            self.graph.observe(self._prev_hash, self._prev_action, cur_hash, change_score)

        # Update change-feedback log for the prior action's outcome.
        if self._prev_action is not None and self._prev_layers is not None:
            dpx = _grid_pixel_diff(self._prev_layers, cur_layers)
            dlevels = cur_levels - self._prev_levels
            self.history.add(
                f"ACTION{self._prev_action}",
                changed=(dpx > 0 or dlevels > 0),
                dpx=dpx,
                dlevels=dlevels,
            )

        # Build prompt.
        image = render_frame_image(_layer0(cur_layers), upscale=self.backbone.frame_upscale)
        untried = sorted(node.untried_actions & set(non_reset_avail))
        messages = build_policy_messages(frame, self.history, image, untried, node.visit_count)

        try:
            reply = self.backbone.generate(messages, image=image)
        except Exception as exc:
            self._log_fallback(exc)
            reply = ""

        chosen_id = parse_action_id(reply, available_ids=non_reset_avail)
        if untried and chosen_id not in untried:
            chosen_id = self._rng.choice(untried)
        elif chosen_id not in non_reset_avail:
            chosen_id = self._rng.choice(non_reset_avail)

        action = GameAction.from_id(chosen_id)
        data: dict[str, int] = {}
        if action.is_complex():
            coords = parse_coords(reply)
            if coords is None:
                data = _sample_click_xy(cur_layers, self._rng)
            else:
                data = {"x": coords[0], "y": coords[1]}
            action.set_data(data)

        self.graph.record_action(cur_hash, chosen_id, data)
        self._prev_hash = cur_hash
        self._prev_action = chosen_id
        self._prev_layers = cur_layers
        self._prev_levels = cur_levels
        return action

    def is_done(self, frame: Any) -> bool:
        return getattr(frame, "state", None) == GameState.WIN

    def _log_fallback(self, exc: Exception) -> None:
        if os.environ.get("QWEN_DEBUG_PROMPTS") != "1":
            return
        try:
            with open("/tmp/qwen_trace.log", "a") as fh:
                fh.write(f"[QwenPolicyAgent] backbone fault: {type(exc).__name__}: {exc}\n")
        except OSError:
            pass


__all__ = ["QwenPolicyAgent", "build_policy_messages"]
