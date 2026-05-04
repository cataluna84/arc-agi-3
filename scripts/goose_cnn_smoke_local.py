"""goose_cnn_smoke_local.py - 22-check parity smoke for GooseCNNAgent.

No GPU, no SDK required. Exercises the CPU/no-torch fallback path of the
GooseCNNPredictor and the agent's control flow on the offline mock env.

Usage:
    .venv/bin/python scripts/goose_cnn_smoke_local.py
Exit code 0 = all checks pass; non-zero = report failure with traceback.
"""

from __future__ import annotations

import random as _random
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _check(passed: bool, label: str, ctx: str = "") -> bool:
    sigil = "PASS" if passed else "FAIL"
    print(f"  [{sigil}] {label}{(' -- ' + ctx) if ctx else ''}")
    return passed


def main() -> int:
    print("=== GooseCNNAgent smoke (22 checks) ===")
    n, ok = 0, 0

    # 1 import model module
    try:
        import agents.goose_cnn_model as gcm  # noqa: F401

        n += 1
        ok += _check(True, "import agents.goose_cnn_model")
    except Exception:
        n += 1
        traceback.print_exc()
        return 1

    # 2 import agent module
    try:
        import agents.goose_cnn_agent as gca  # noqa: F401

        n += 1
        ok += _check(True, "import agents.goose_cnn_agent")
    except Exception:
        n += 1
        traceback.print_exc()
        return 1

    import numpy as np

    from agents.goose_cnn_agent import (
        _action_logits_to_six,
        _frame_changed,
        _softmax_sample_1d,
        _softmax_sample_2d,
    )
    from agents.goose_cnn_model import (
        GRID_SIZE,
        NUM_SIMPLE_ACTIONS,
        ExperienceBuffer,
        GooseCNNPredictor,
        hash_frame_grid,
    )

    # 3 hash determinism
    rng = np.random.default_rng(0)
    g = rng.integers(0, 16, size=(GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    n += 1
    ok += _check(hash_frame_grid(g) == hash_frame_grid(g.copy()), "hash_frame_grid deterministic")

    # 4 hash size
    n += 1
    ok += _check(len(hash_frame_grid(g)) == 8, "hash_frame_grid is 8 bytes")

    # 5 hash distinctness
    g2 = rng.integers(0, 16, size=(GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    n += 1
    ok += _check(hash_frame_grid(g) != hash_frame_grid(g2), "hash_frame_grid distinguishes grids")

    # 6 buffer add + dedup
    buf = ExperienceBuffer(max_size=4)
    buf.add(b"a" * 8, 1, True)
    buf.add(b"a" * 8, 1, False)  # update in place
    n += 1
    ok += _check(len(buf) == 1, "ExperienceBuffer dedups identical key", ctx=str(len(buf)))

    # 7 buffer eviction
    buf.add(b"b" * 8, 2, True)
    buf.add(b"c" * 8, 6, True, click_xy=(1, 2))
    buf.add(b"d" * 8, 3, True)
    buf.add(b"e" * 8, 4, True)  # forces eviction
    n += 1
    ok += _check(len(buf) == 4, "ExperienceBuffer respects max_size", ctx=str(len(buf)))

    # 8 buffer sample shape
    samples = buf.sample(2, _random.Random(0))
    n += 1
    ok += _check(len(samples) == 2 and len(samples[0]) == 5, "buffer.sample returns 5-tuples")

    # 9 predictor builds (or falls back) without crash
    pred = GooseCNNPredictor(seed=0, device="cpu")
    out = pred.predict(g)
    n += 1
    ok += _check(
        out["action_probs"].shape == (NUM_SIMPLE_ACTIONS,),
        "predictor.action_probs shape",
        ctx=str(out["action_probs"].shape),
    )

    # 10 coord shape
    n += 1
    ok += _check(out["coord_probs"].shape == (GRID_SIZE, GRID_SIZE), "predictor.coord_probs shape")

    # 11 prob range
    n += 1
    ok += _check(
        bool((out["action_probs"] >= 0).all())
        and bool((out["action_probs"] <= 1).all())
        and bool((out["coord_probs"] >= 0).all())
        and bool((out["coord_probs"] <= 1).all()),
        "predictor outputs in [0, 1]",
    )

    # 12 softmax_sample_1d respects mask
    logits = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mask = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    rng_py = _random.Random(0)
    sampled_set = {_softmax_sample_1d(logits, mask, rng_py) for _ in range(20)}
    n += 1
    ok += _check(sampled_set == {1}, "softmax_sample_1d respects mask")

    # 13 softmax_sample_2d in bounds
    cp = np.full((GRID_SIZE, GRID_SIZE), 0.5, dtype=np.float32)
    in_bounds = True
    for _ in range(20):
        x, y = _softmax_sample_2d(cp, _random.Random(0))
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            in_bounds = False
            break
    n += 1
    ok += _check(in_bounds, "softmax_sample_2d in bounds")

    # 14 _action_logits_to_six combines properly
    six = _action_logits_to_six(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), cp)
    n += 1
    ok += _check(six.shape == (6,) and abs(float(six[5]) - 0.5) < 1e-5, "logits_to_six shape+max")

    # 15 _frame_changed
    a = [g.copy()]
    b = [g.copy()]
    c = [g2.copy()]
    n += 1
    ok += _check(
        not _frame_changed(a, b) and _frame_changed(a, c),
        "frame_changed identity vs diff",
    )

    # 16 agent constructs
    from agents import GameAction, GameState, MockFrame
    from agents.goose_cnn_agent import GooseCNNAgent

    agent = GooseCNNAgent(seed=0, device="cpu")
    n += 1
    ok += _check(agent.name == "goose-cnn", "agent.name is 'goose-cnn'")

    # 17 NOT_PLAYED -> RESET
    fr = MockFrame(state=GameState.NOT_PLAYED, available_actions=[1, 2, 6])
    a0 = agent.choose_action(fr)
    n += 1
    ok += _check(a0 == GameAction.RESET, "NOT_PLAYED -> RESET")

    # 18 valid action choice
    fr2 = MockFrame(
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        available_actions=[1, 2, 3, 6],
        frame=[g],
    )
    a1 = agent.choose_action(fr2)
    a1_int = int(getattr(a1, "value", a1))
    n += 1
    ok += _check(isinstance(a1, GameAction) and a1_int in {1, 2, 3, 6}, "valid action choice")

    # 19 ACTION6 carries x/y in [0, 63]
    fr3 = MockFrame(
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        available_actions=[6],
        frame=[g],
    )
    a2 = agent.choose_action(fr3)
    a2_int = int(getattr(a2, "value", a2))
    raw_data = getattr(a2, "_data", None) or getattr(a2, "action_data", None)
    if isinstance(raw_data, dict):
        dx, dy = raw_data.get("x", -1), raw_data.get("y", -1)
    else:
        dx = getattr(raw_data, "x", -1) if raw_data is not None else -1
        dy = getattr(raw_data, "y", -1) if raw_data is not None else -1
    n += 1
    ok += _check(
        a2_int == 6 and 0 <= dx < GRID_SIZE and 0 <= dy < GRID_SIZE,
        "ACTION6 carries x/y in [0, 63]",
        ctx=f"x={dx} y={dy}",
    )

    # 20 level transition wipes buffer
    pre_n = len(agent.predictor.buffer)
    fr4 = MockFrame(
        state=GameState.NOT_FINISHED, levels_completed=1, available_actions=[1, 2], frame=[g]
    )
    agent.choose_action(fr4)
    n += 1
    ok += _check(
        len(agent.predictor.buffer) == 0,
        "level transition wipes ExperienceBuffer",
        ctx=f"pre={pre_n} post={len(agent.predictor.buffer)}",
    )

    # 21 mock end-to-end via local_runner -- expect >=1 win
    import json
    import subprocess

    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "local_runner.py"),
        "--agent",
        "agents.goose_cnn_agent:GooseCNNAgent",
        "--games",
        "ls20-mock",
        "--max-actions",
        "300",
        "--seed",
        "0",
    ]
    try:
        out_b = subprocess.check_output(cmd, cwd=str(REPO_ROOT), stderr=subprocess.STDOUT)
        text = out_b.decode("utf-8", errors="replace")
        last = text.rfind("[local_runner] totals:")
        totals = json.loads(text[last:].split(":", 1)[1])
        n += 1
        ok += _check(
            totals["wins"] >= 1,
            "mock end-to-end: >=1 win on ls20-mock in 300 actions",
            ctx=str(totals),
        )
    except Exception as e:
        n += 1
        ok += _check(False, "mock end-to-end run failed", ctx=repr(e))

    # 22 is_done semantics
    n += 1
    ok += _check(
        agent.is_done(MockFrame(state=GameState.WIN, frame=[g])) is True
        and agent.is_done(MockFrame(state=GameState.NOT_FINISHED, frame=[g])) is False,
        "is_done WIN-only",
    )

    print(f"\n=== Result: {ok}/{n} checks passed ===")
    return 0 if ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
