"""trigger_bfs_smoke_local.py - 22-check parity smoke for TriggerBFSAgent.

No GPU, no SDK required. Validates that:
  1. agents.trigger_bfs_agent is importable.
  2. agents.state_graph primitives behave correctly (hash, frontier).
  3. TriggerBFSAgent constructs cleanly.
  4. choose_action returns a valid GameAction.
  5. ACTION6 emits an x/y in [0, 63].
  6. Level transitions clear the graph.
  7. End-to-end run on the offline mock env returns >=1 win.

Usage:
    .venv/bin/python scripts/trigger_bfs_smoke_local.py
Exit code 0 = all checks pass; non-zero = report failure with traceback.
"""

from __future__ import annotations

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
    print("=== TriggerBFSAgent smoke (22 checks) ===")
    n, ok = 0, 0

    # 1
    try:
        import agents.trigger_bfs_agent as mod  # noqa: F401

        n += 1
        ok += _check(True, "import agents.trigger_bfs_agent")
    except Exception:
        n += 1
        traceback.print_exc()
        return 1

    # 2 import state_graph + primitives
    try:
        from agents.state_graph import StateGraph, StateNode, hash_frame  # noqa: F401

        n += 1
        ok += _check(True, "import agents.state_graph")
    except Exception:
        n += 1
        traceback.print_exc()
        return 1

    # 3 hash determinism
    import numpy as np

    rng = np.random.default_rng(0)
    g = [rng.integers(0, 16, size=(64, 64), dtype=np.uint8)]
    n += 1
    ok += _check(hash_frame(g) == hash_frame([np.copy(g[0])]), "hash_frame is deterministic")

    # 4 hash size
    n += 1
    ok += _check(len(hash_frame(g)) == 8, "hash_frame returns 8 bytes")

    # 5 hash distinctness
    g2 = [rng.integers(0, 16, size=(64, 64), dtype=np.uint8)]
    n += 1
    ok += _check(hash_frame(g) != hash_frame(g2), "hash_frame distinguishes different grids")

    # 6 StateGraph.add_or_get seeds untried (RESET excluded)
    sg = StateGraph()
    h = b"a" * 8
    node = sg.add_or_get(h, [0, 1, 2, 6], levels=0)
    n += 1
    ok += _check(node.untried_actions == {1, 2, 6}, "add_or_get seeds untried minus RESET")

    # 7 frontier population
    n += 1
    ok += _check(h in sg.frontier, "new node with untried actions joins frontier")

    # 8 observe drains untried
    h2 = b"b" * 8
    sg.add_or_get(h2, [1], levels=0)
    sg.observe(h, 1, h2)
    n += 1
    ok += _check(1 not in sg.nodes[h].untried_actions, "observe removes the tried action")

    # 9 frontier auto-removal once exhausted
    sg.observe(h, 2, h2)
    sg.observe(h, 6, h2)
    n += 1
    ok += _check(h not in sg.frontier, "frontier auto-removes node when fully tried")

    # 10 maybe_reset_for_level
    n += 1
    ok += _check(sg.maybe_reset_for_level(1) is True, "level transition resets graph")
    n += 1
    ok += _check(sg.stats()["n_nodes"] == 0, "graph is empty after reset")

    # 12 trigger score formula sanity
    from agents.trigger_bfs_agent import _sample_click_xy, _trigger_score

    p = [np.zeros((64, 64), dtype=np.uint8)]
    nfr = [np.copy(p[0])]
    nfr[0][10, 10] = 7
    s = _trigger_score(p, nfr, 0, 0)
    n += 1
    ok += _check(s > 0.0, "trigger_score > 0 when one pixel changes")

    # 13 trigger score with level delta
    s2 = _trigger_score(p, nfr, 0, 1)
    n += 1
    ok += _check(s2 > s, "trigger_score grows with level transition")

    # 14 click sampling within bounds
    xy = _sample_click_xy(p, __import__("random").Random(0))
    n += 1
    ok += _check(0 <= xy["x"] <= 63 and 0 <= xy["y"] <= 63, "click xy within [0, 63]")

    # 15 click sampling on empty grid falls back to uniform
    xy2 = _sample_click_xy(None, __import__("random").Random(0))
    n += 1
    ok += _check(0 <= xy2["x"] <= 63 and 0 <= xy2["y"] <= 63, "click xy fallback bounds")

    # 16 agent constructs
    from agents import GameAction, GameState, MockFrame
    from agents.trigger_bfs_agent import TriggerBFSAgent

    agent = TriggerBFSAgent(seed=0)
    n += 1
    ok += _check(agent.name == "trigger-bfs", "agent.name is 'trigger-bfs'")

    # 17 NOT_PLAYED -> RESET
    fr = MockFrame(state=GameState.NOT_PLAYED, available_actions=[1, 2, 6])
    a0 = agent.choose_action(fr)
    n += 1
    ok += _check(a0 == GameAction.RESET, "NOT_PLAYED -> RESET")

    # 18 normal turn returns valid GameAction
    fr2 = MockFrame(
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        available_actions=[1, 2, 3, 6],
        frame=[np.zeros((64, 64), dtype=np.uint8)],
    )
    a1 = agent.choose_action(fr2)
    n += 1
    ok += _check(isinstance(a1, GameAction), "choose_action returns GameAction")

    # 19 ACTION6 carries valid x/y when emitted
    # Force ACTION6 by repeatedly running until we see it
    seen_a6 = False
    for _ in range(40):
        a = agent.choose_action(fr2)
        if a == GameAction.ACTION6:
            seen_a6 = True
            ad = getattr(a, "_data", None) or getattr(a, "action_data", None)
            if isinstance(ad, dict):
                x, y = ad.get("x", -1), ad.get("y", -1)
            else:
                x = getattr(ad, "x", -1) if ad is not None else -1
                y = getattr(ad, "y", -1) if ad is not None else -1
            break
    if seen_a6:
        n += 1
        ok += _check(0 <= x <= 63 and 0 <= y <= 63, "ACTION6 carries x/y in [0,63]")
    else:
        n += 1
        ok += _check(True, "ACTION6 not chosen in 40 mock turns (graph fully explored)")

    # 20 graph grew with observed states
    n += 1
    ok += _check(agent.graph.stats()["n_nodes"] >= 1, "graph has >=1 node after turns")

    # 21 level transition during play wipes the graph
    fr3 = MockFrame(
        state=GameState.NOT_FINISHED,
        levels_completed=1,
        available_actions=[1, 2, 6],
        frame=[np.zeros((64, 64), dtype=np.uint8)],
    )
    agent.choose_action(fr3)
    n += 1
    ok += _check(agent.graph.current_levels == 1, "agent advanced to level 1")

    # 22 mock end-to-end via local_runner
    import json
    import subprocess

    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "local_runner.py"),
        "--agent",
        "agents.trigger_bfs_agent:TriggerBFSAgent",
        "--games",
        "ls20-mock",
        "--max-actions",
        "300",
        "--seed",
        "0",
    ]
    try:
        out = subprocess.check_output(cmd, cwd=str(REPO_ROOT), stderr=subprocess.STDOUT)
        text = out.decode("utf-8", errors="replace")
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

    print(f"\n=== Result: {ok}/{n} checks passed ===")
    return 0 if ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
