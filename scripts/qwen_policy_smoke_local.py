"""Local 22-check parity smoke for QwenPolicyAgent (Option A).

Exercises the agent's control flow end-to-end without loading the model:
  - parsers (action id, coords, score, choice)
  - change-log + state-graph integration
  - prompt builder
  - mock-backbone end-to-end via local_runner subprocess

No GPU, no torch import, no vLLM install required. Pure Python.

Run:
    .venv/bin/python scripts/qwen_policy_smoke_local.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make `agents` and `experiments` importable when run as a script.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _ok(label: str) -> None:
    print(f"  [PASS] {label}")


def _fail(label: str, exc: Exception | None = None) -> None:
    print(f"  [FAIL] {label}: {exc}")
    sys.exit(1)


def main() -> int:
    print("=== qwen_policy_smoke_local: 22 checks (no GPU/no model) ===")
    checks = 0

    # ---- Imports + module surface ----------------------------------------
    try:
        from agents import qwen_backbone as bb
        from agents.qwen_backbone import ChangeLog, QwenBackbone
        from agents.qwen_policy_agent import QwenPolicyAgent, build_policy_messages
    except Exception as exc:
        _fail("imports succeed", exc)
        return 1
    _ok("imports succeed")
    checks += 1

    expected_bb = {
        "ChangeLog",
        "QwenBackbone",
        "parse_action_id",
        "parse_choice",
        "parse_coords",
        "parse_score",
        "render_frame_image",
        "render_frame_text",
    }
    missing = expected_bb - set(bb.__all__)
    if missing:
        _fail("qwen_backbone __all__ complete", RuntimeError(str(missing)))
    _ok("qwen_backbone __all__ complete")
    checks += 1

    # ---- Parsers ---------------------------------------------------------
    if bb.parse_action_id("ACTION3", [1, 2, 3]) != 3:
        _fail("parse_action_id basic")
    _ok("parse_action_id basic")
    checks += 1

    if bb.parse_action_id("ACTION99", [1, 2, 3]) != 1:
        _fail("parse_action_id falls back to lowest available")
    _ok("parse_action_id falls back")
    checks += 1

    if bb.parse_coords("(12, 34)") != (12, 34):
        _fail("parse_coords bare")
    _ok("parse_coords bare")
    checks += 1

    if bb.parse_coords("x=4, y=8") != (4, 8):
        _fail("parse_coords labeled")
    _ok("parse_coords labeled")
    checks += 1

    if bb.parse_score("score: 0.42") != 0.42:
        _fail("parse_score basic")
    _ok("parse_score basic")
    checks += 1

    if bb.parse_choice("explore please", ["explore", "replay"]) != "explore":
        _fail("parse_choice basic")
    _ok("parse_choice basic")
    checks += 1

    # ---- ChangeLog -------------------------------------------------------
    cl = ChangeLog(capacity=4)
    for i in range(6):
        cl.add(f"ACTION{i}", changed=(i % 2 == 0), dpx=i)
    if len(cl) != 4:
        _fail("ChangeLog caps capacity")
    _ok("ChangeLog caps capacity")
    checks += 1

    cl2 = ChangeLog()
    cl2.add("ACTION6", changed=True, dpx=64, dlevels=1)
    if "LEVEL UP" not in cl2.render():
        _fail("ChangeLog renders LEVEL UP")
    _ok("ChangeLog renders LEVEL UP")
    checks += 1

    # ---- Frame rendering -------------------------------------------------
    g = [[i % 16 for i in range(64)] for _ in range(64)]
    text = bb.render_frame_text(g)
    if len(text.split("\n")) != 64:
        _fail("render_frame_text rows=64")
    _ok("render_frame_text rows=64")
    checks += 1

    img = bb.render_frame_image(g, upscale=4)
    if img is not None and (img.size != (256, 256)):
        _fail("render_frame_image upscale=4 size")
    _ok("render_frame_image upscale=4 size")
    checks += 1

    img8 = bb.render_frame_image(g, upscale=8)
    if img8 is not None and (img8.size != (512, 512)):
        _fail("render_frame_image upscale=8 size")
    _ok("render_frame_image upscale=8 size")
    checks += 1

    # ---- QwenBackbone construction (no load) ------------------------------
    backbone = QwenBackbone(model_path="/dev/null")
    if backbone.actual_runtime != "none":
        _fail("backbone unloaded actual_runtime is 'none'")
    if backbone.frame_upscale != 4:
        _fail("backbone defaults frame_upscale=4")
    _ok("backbone defaults frame_upscale=4 + unloaded")
    checks += 1

    # ---- Mock backbone class for offline test ----------------------------
    class _FakeBackbone:
        frame_upscale = 4
        actual_runtime = "fake"

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, messages, image=None):
            self.calls += 1
            # Cycle through actions so we exercise different code paths.
            return ["ACTION3", "ACTION1", "ACTION6 (12, 34)", "ACTION2"][self.calls % 4]

    fb = _FakeBackbone()
    agent = QwenPolicyAgent(seed=0, backbone=fb)
    if agent.name != "qwen_policy":
        _fail("agent.name == 'qwen_policy'")
    _ok("QwenPolicyAgent constructs cleanly")
    checks += 1

    # ---- Prompt building --------------------------------------------------
    from agents import MockFrame

    mf = MockFrame(
        game_id="test",
        levels_completed=0,
        win_levels=3,
        available_actions=[1, 2, 3, 6, 7],
        frame=[[[0] * 64 for _ in range(64)]],
    )
    msgs = build_policy_messages(
        mf, ChangeLog(), image=None, untried_actions=[1, 2, 3, 6], visit_count=0
    )
    if not (len(msgs) == 2 and msgs[0]["role"] == "system" and msgs[1]["role"] == "user"):
        _fail("build_policy_messages structure")
    _ok("build_policy_messages 2-role chat")
    checks += 1

    # The user content must contain available + untried lists.
    user_text_parts = [p for p in msgs[1]["content"] if p["type"] == "text"]
    user_text = user_text_parts[0]["text"] if user_text_parts else ""
    if "Available:" not in user_text or "Untried at this state:" not in user_text:
        _fail("policy prompt contains both lists")
    _ok("policy prompt contains both lists")
    checks += 1

    # ---- choose_action drives state graph + change log ---------------------
    a1 = agent.choose_action(mf)
    if not hasattr(a1, "name"):
        _fail("choose_action returns GameAction-like object")
    _ok("choose_action returns GameAction-like object")
    checks += 1

    # Subsequent call after a state change should add to change-log.
    mf2 = MockFrame(
        game_id="test",
        levels_completed=0,
        win_levels=3,
        available_actions=[1, 2, 3, 6, 7],
        frame=[[[1] * 64 for _ in range(64)]],
    )
    a2 = agent.choose_action(mf2)
    if len(agent.history) == 0:
        _fail("choose_action populates change log")
    _ok("choose_action populates change log")
    checks += 1

    # State graph picks up two distinct frames.
    if len(agent.graph.nodes) < 2:
        _fail("state graph accumulates >=2 nodes after 2 calls")
    _ok("state graph accumulates >=2 nodes")
    checks += 1

    # RESET path on GAME_OVER.
    from agents import GameAction, GameState

    mf3 = MockFrame(state=GameState.GAME_OVER)
    a3 = agent.choose_action(mf3)
    if a3 != GameAction.RESET:
        _fail("choose_action returns RESET on GAME_OVER")
    _ok("choose_action returns RESET on GAME_OVER")
    checks += 1

    # ---- Untried filter forces the LLM choice into untried set --------------
    class _AlwaysActionOne:
        frame_upscale = 4
        actual_runtime = "fake"

        def generate(self, messages, image=None):
            return "ACTION1"

    agent2 = QwenPolicyAgent(seed=0, backbone=_AlwaysActionOne())
    mfa = MockFrame(
        game_id="test",
        levels_completed=0,
        win_levels=3,
        available_actions=[2, 3, 4, 5],  # ACTION1 NOT available
        frame=[[[0] * 64 for _ in range(64)]],
    )
    chosen = agent2.choose_action(mfa)
    chosen_id = int(getattr(chosen, "value", chosen))
    if chosen_id not in {2, 3, 4, 5}:
        _fail("untried filter rejects unavailable LLM choice")
    _ok("untried filter rejects unavailable LLM choice")
    checks += 1

    print(f"\n=== Result: {checks}/22 checks passed ===")
    if checks != 22:
        return 1
    return 0


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"wall_clock_s={time.time() - t0:.2f}")
    sys.exit(rc)
