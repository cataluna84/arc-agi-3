#!/usr/bin/env python3
"""qwen_agent_smoke_local.py - GPU-free smoke test for QwenAgent's pure-Python parts.

Verifies:
    1. `agents.qwen_agent.build_prompt(frame, history)` produces a sensible
       chat-message structure with image + text grid + available_actions.
    2. `agents.qwen_agent.parse_action(text, available_ids)` correctly handles
       a battery of model-reply shapes (canonical, noisy, missing, malformed,
       coords, ALL CAPS / lower / mixed).
    3. End-to-end frame -> prompt -> (synthetic reply) -> action loop on the
       offline `MockGame` without any model weights or torch imports.

Run from repo root:
    .venv/bin/python scripts/qwen_agent_smoke_local.py

Exits non-zero on any check failure; suitable for CI / make smoke.
"""

from __future__ import annotations

import os
import random
import sys

# Ensure repo root is on sys.path so `import agents` works regardless of CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agents import GameAction, GameState, MockFrame  # noqa: E402
from agents.qwen_agent import build_prompt, parse_action  # noqa: E402


def _make_synthetic_frame(seed: int = 0) -> MockFrame:
    """64x64 grid with a small coloured square in the centre, palette 0..15."""
    rng = random.Random(seed)
    grid = [[0] * 64 for _ in range(64)]
    # plant a non-trivial pattern: a 6x6 square of colour 7 around (28..34, 28..34)
    for y in range(28, 34):
        for x in range(28, 34):
            grid[y][x] = 7
    # scatter a handful of stray cells so the text grid isn't all zeros
    for _ in range(20):
        y = rng.randint(0, 63)
        x = rng.randint(0, 63)
        grid[y][x] = rng.randint(1, 15)
    return MockFrame(
        game_id="ls20-mock",
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        win_levels=3,
        available_actions=[1, 2, 3, 4, 6],
        frame=[grid],
    )


def _check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}{(' - ' + detail) if detail else ''}")
    if not ok:
        _check.failures += 1  # type: ignore[attr-defined]


_check.failures = 0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Suite 1: build_prompt structure
# ---------------------------------------------------------------------------


def suite_build_prompt() -> None:
    print("\n[suite] build_prompt")
    frame = _make_synthetic_frame(seed=1)
    history = [("ACTION1", True), ("ACTION3", False), ("ACTION2", True)]
    messages, image = build_prompt(frame, history)

    _check(
        "messages is a 2-element list (system + user)",
        isinstance(messages, list) and len(messages) == 2,
        f"got len={len(messages) if isinstance(messages, list) else type(messages)}",
    )
    _check("system role correct", messages[0].get("role") == "system")
    _check("user role correct", messages[1].get("role") == "user")

    user_parts = messages[1].get("content", [])
    _check("user content is a list of parts", isinstance(user_parts, list) and len(user_parts) >= 1)

    # If PIL is installed we expect an image part; otherwise text-only is fine.
    has_image_part = any(p.get("type") == "image" for p in user_parts)
    text_part = next((p for p in user_parts if p.get("type") == "text"), None)
    _check("at least one text part", text_part is not None)

    if has_image_part:
        _check("image is non-empty PIL.Image", image is not None)
        _check(
            "image is at least 512x512 (8x upscale)",
            image is not None and image.size[0] >= 512 and image.size[1] >= 512,
            detail=f"size={getattr(image, 'size', None)}",
        )
    else:
        print("  [info] PIL not installed; text-only prompt verified")

    text = (text_part or {}).get("text", "")
    _check("text mentions available actions", "Available:" in text)
    _check(
        "text contains the actual ACTION1/2/3/4/6 list",
        all(f"ACTION{i}" in text for i in (1, 2, 3, 4, 6)),
    )
    _check(
        "text includes recent history lines",
        "ACTION1: changed" in text and "ACTION3: no-change" in text,
    )
    sys_text = messages[0]["content"].lower()
    _check(
        "system asks for terse single-action reply",
        "actionn" in sys_text and "one action only" in sys_text,
    )


# ---------------------------------------------------------------------------
# Suite 2: parse_action robustness
# ---------------------------------------------------------------------------


def suite_parse_action() -> None:
    print("\n[suite] parse_action")
    avail = [1, 2, 3, 4, 6]

    cases: list[tuple] = [
        # (label, model_reply, available_ids, expected_id, expected_coords)
        ("canonical end-of-line", "I will increment the counter.\nACTION1", avail, 1, None),
        ("colon syntax", "Action: 3 - move down", avail, 3, None),
        ("snake_case", "Pick action_4 because of the diagonal", avail, 4, None),
        ("RESET keyword (avail includes 0)", "Game over - sending RESET", [0, 1, 2, 3], 0, None),
        (
            "RESET ignored when 0 not in avail (env not in reset state)",
            "Game over - sending RESET",
            avail,
            1,
            None,
        ),
        (
            "multiple mentions, picks first available",
            "I considered ACTION5 then ACTION2 - going with the latter",
            avail,
            2,
            None,
        ),
        ("ACTION6 with coords", "Click at ACTION6 (12, 34) to plant pattern", avail, 6, (12, 34)),
        ("ACTION6 with x= y= syntax", "ACTION6 x=40, y=15", avail, 6, (40, 15)),
        (
            "filtered: model picks ACTION7 not in avail -> deterministic fallback",
            "ACTION7",
            avail,
            1,
            None,
        ),
        ("noise + lowercase", "i think action 6 with (5, 5) makes sense.", avail, 6, (5, 5)),
        (
            "garbage -> deterministic fallback",
            "this reply contains no action token at all",
            avail,
            1,
            None,
        ),
        (
            "ACTION6 missing coords -> coords None",
            "ACTION6 - just click somewhere reasonable",
            avail,
            6,
            None,
        ),
    ]

    for label, reply, av, exp_id, exp_coords in cases:
        action, coords = parse_action(reply, av)
        ok = (int(action.value) == exp_id) and (coords == exp_coords)
        _check(
            label,
            ok,
            detail=f"got id={action.value} coords={coords} expected id={exp_id} coords={exp_coords}",
        )


# ---------------------------------------------------------------------------
# Suite 3: end-to-end loop with synthetic replies (no model)
# ---------------------------------------------------------------------------


def suite_end_to_end() -> None:
    print("\n[suite] end-to-end (synthetic replies, no torch)")
    # We monkey-patch QwenAgent's heavy methods so we can drive the loop without
    # loading any model. This validates that the agent's bookkeeping (history,
    # frame-change detection, state machine) is sound.

    from agents.qwen_agent import QwenAgent

    class FakeQwen(QwenAgent):
        def _ensure_model_loaded(self) -> None:  # type: ignore[override]
            self._loaded = True
            self._processor = None
            self._model = None

            class _T:  # tiny torch stub
                @staticmethod
                def inference_mode():
                    class _CM:
                        def __enter__(self_):
                            return None

                        def __exit__(self_, *a):
                            return False

                    return _CM()

            self._torch = _T()

        def choose_action(self, frame):
            # Pre-empt the parent: deterministic synthetic reply driven by step counter.
            # We still exercise build_prompt + parse_action.
            from agents.qwen_agent import build_prompt, parse_action  # local

            state = getattr(frame, "state", None)
            if state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                self._record_history(frame, "RESET")
                return GameAction.RESET
            self._update_history_from_observation(frame)
            _messages, _ = build_prompt(frame, list(self._history))
            avail = [int(a) for a in (frame.available_actions or [])]
            # Cycle through the available actions deterministically
            step = getattr(self, "_step_idx", 0)
            chosen = avail[step % len(avail)] if avail else 1
            self._step_idx = step + 1
            reply = f"ACTION{chosen} (32, 32)"
            action, coords = parse_action(reply, avail)
            if action.is_complex():
                x, y = coords if coords is not None else (32, 32)
                action.set_data({"x": x, "y": y})
            layers = getattr(frame, "frame", None) or []
            grid = layers[0] if layers else None
            self._prev_grid_hash = self._grid_hash(grid)
            self._prev_action_name = action.name
            return action

    # Drive against the offline MockGame to exercise the env -> agent pipe
    sys.path.insert(0, _REPO_ROOT)
    from experiments.local_runner import MockGame

    env = MockGame("ls20-mock", seed=0)
    env.reset()

    agent = FakeQwen()
    frame = env.observation_space
    actions_taken = 0
    valid_action_count = 0
    for _step in range(30):
        if frame.state in (GameState.WIN, GameState.GAME_OVER):
            break
        action = agent.choose_action(frame)
        if int(action.value) in (frame.available_actions or []):
            valid_action_count += 1
        data = getattr(action, "_data", {}) or {}
        frame = env.step(action, data=data, reasoning=None)
        actions_taken += 1
        if frame is None:
            break

    _check(
        "e2e: agent emitted at least 5 actions",
        actions_taken >= 5,
        detail=f"actions_taken={actions_taken}",
    )
    _check(
        "e2e: every emitted action was in available_actions",
        valid_action_count == actions_taken,
        detail=f"valid={valid_action_count}/{actions_taken}",
    )
    _check(
        "e2e: history buffer populated within cap",
        0 < len(agent._history) <= agent.history_len,
        detail=f"history_len={len(agent._history)}/{agent.history_len}",
    )


# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 72)
    print("Qwen agent local smoke (no GPU, no model load, no torch)")
    print("=" * 72)
    suite_build_prompt()
    suite_parse_action()
    suite_end_to_end()
    print("\n" + "-" * 72)
    n_fail = _check.failures  # type: ignore[attr-defined]
    if n_fail:
        print(f"FAILED ({n_fail} check(s) failed)")
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
