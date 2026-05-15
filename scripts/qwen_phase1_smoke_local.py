#!/usr/bin/env python3
"""qwen_phase1_smoke_local.py - CPU-only smoke for Phase-1 QwenAgent features.

Extends `scripts/qwen_agent_smoke_local.py` with checks for the Phase-1
additions:
    - _parse_json_first round-trips valid + malformed JSON
    - _segment_action6_candidates returns ≤8 candidates, tier-0 first,
      handles empty grids
    - _apply_guards snaps unavailable actions, fills ACTION6 from candidates
      by `why="Cn"` label, defaults to (32,32) when no candidates
    - State-graph maybe_reset_for_level fires on level transitions
    - Fallback path: scripted exception in _choose_action_inner returns
      a valid GameAction via TriggerBFS
    - Outcome history truncates at history_len

Run from repo root:
    .venv/bin/python scripts/qwen_phase1_smoke_local.py

Exits non-zero on any check failure.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}{(' - ' + detail) if detail else ''}")
    if not ok:
        _check.failures += 1  # type: ignore[attr-defined]


_check.failures = 0  # type: ignore[attr-defined]


def _get_xy(action) -> tuple:
    """Extract (x, y) from action; works for both arcengine.GameAction and mock IntEnum.

    arcengine stores click coords on `action.action_data` (a typed ComplexAction).
    The mock IntEnum stores them on `action._data` (dict).
    """
    if not action.is_complex():
        return (None, None)
    ad = getattr(action, "action_data", None)
    if ad is not None:
        try:
            return (int(ad.x), int(ad.y))
        except (AttributeError, TypeError, ValueError):
            pass
    d = getattr(action, "_data", None)
    if isinstance(d, dict):
        return (d.get("x"), d.get("y"))
    return (None, None)


def _make_grid_with_blob(blob_color: int = 7, top_left: tuple[int, int] = (28, 28)):
    grid = [[0] * 64 for _ in range(64)]
    y0, x0 = top_left
    for y in range(y0, y0 + 6):
        for x in range(x0, x0 + 6):
            grid[y][x] = blob_color
    return grid


def _make_mock_frame(
    *,
    available=None,
    levels: int = 0,
    state_name: str = "NOT_FINISHED",
    grid=None,
):
    from agents import GameState, MockFrame

    state_map = {
        "NOT_FINISHED": GameState.NOT_FINISHED,
        "NOT_PLAYED": GameState.NOT_PLAYED,
        "WIN": GameState.WIN,
        "GAME_OVER": GameState.GAME_OVER,
    }
    return MockFrame(
        game_id="ls20-mock",
        state=state_map.get(state_name, GameState.NOT_FINISHED),
        levels_completed=levels,
        win_levels=3,
        available_actions=list(available) if available is not None else [1, 2, 3, 4, 6],
        frame=[grid if grid is not None else _make_grid_with_blob()],
    )


# ---------------------------------------------------------------------------
# Suite 1: _parse_json_first
# ---------------------------------------------------------------------------


def suite_parse_json_first() -> None:
    print("\n[suite] _parse_json_first")
    from agents.qwen_agent import _parse_json_first

    d = _parse_json_first('{"action":"ACTION3","x":null,"y":null,"why":"x"}')
    _check("valid JSON returns dict", isinstance(d, dict))
    _check("valid JSON action field", d is not None and d.get("action") == "ACTION3")

    d = _parse_json_first('Here you go: {"action":"ACTION6","x":12,"y":8,"why":"C0"} done')
    _check("embedded JSON parses", d is not None and d.get("x") == 12)

    _check(
        "malformed JSON returns None",
        _parse_json_first('{"action":"ACTION3" oops broken}') is None,
    )
    _check("no JSON returns None", _parse_json_first("just ACTION1 prose") is None)
    _check("empty string returns None", _parse_json_first("") is None)
    _check("None input returns None", _parse_json_first(None) is None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Suite 2: _segment_action6_candidates
# ---------------------------------------------------------------------------


def suite_segment_action6_candidates() -> None:
    print("\n[suite] _segment_action6_candidates")
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        print("  [info] numpy not installed; skipping segmenter suite")
        return

    from agents.qwen_agent import QwenAgent

    agent = QwenAgent()

    grid = _make_grid_with_blob(blob_color=7, top_left=(28, 28))
    cands = agent._segment_action6_candidates([grid])
    _check("blob grid returns ≥1 candidate", len(cands) >= 1, detail=f"len={len(cands)}")
    _check("candidate has required keys", all({"label", "x", "y", "tier"} <= set(c) for c in cands))
    _check("candidate coords in [0,63]", all(0 <= c["x"] < 64 and 0 <= c["y"] < 64 for c in cands))
    _check("first candidate label is C0", cands[0]["label"] == "C0")
    _check("first candidate tier ≤ 1", cands[0]["tier"] <= 1, detail=f"tier={cands[0]['tier']}")

    _check("empty layers returns []", agent._segment_action6_candidates([]) == [])
    _check("None layers returns []", agent._segment_action6_candidates(None) == [])  # type: ignore[arg-type]

    multi_grid = [[0] * 64 for _ in range(64)]
    for i in range(12):
        y0 = (i * 5) % 60
        x0 = (i * 7) % 60
        for y in range(y0, min(64, y0 + 3)):
            for x in range(x0, min(64, x0 + 3)):
                multi_grid[y][x] = 6 + (i % 10)
    cands = agent._segment_action6_candidates([multi_grid])
    _check("candidates capped at 8 with many blobs", len(cands) <= 8, detail=f"len={len(cands)}")


# ---------------------------------------------------------------------------
# Suite 3: _apply_guards
# ---------------------------------------------------------------------------


def suite_apply_guards() -> None:
    print("\n[suite] _apply_guards")
    from agents.qwen_agent import QwenAgent

    agent = QwenAgent()
    frame_simple = _make_mock_frame(available=[1, 2, 3])
    frame_with_6 = _make_mock_frame(available=[1, 2, 6])

    # Case A: parsed action out of avail → snaps to lowest non-RESET available.
    a = agent._apply_guards({"action": "ACTION9"}, frame_simple, candidates=[])
    _check("snap unavailable to lowest avail", int(a.value) == 1)

    # Case B: parsed ACTION6 with explicit x,y → uses them.
    a = agent._apply_guards(
        {"action": "ACTION6", "x": 5, "y": 7, "why": "x"},
        frame_with_6,
        candidates=[],
    )
    x, y = _get_xy(a)
    _check(
        "ACTION6 explicit (x,y) is honored",
        int(a.value) == 6 and x == 5 and y == 7,
        detail=f"got value={a.value} xy=({x},{y})",
    )

    # Case C: ACTION6 with why="C2" and 4 candidates → uses candidates[2].
    cands = [
        {"label": "C0", "x": 10, "y": 10, "tier": 0},
        {"label": "C1", "x": 20, "y": 20, "tier": 0},
        {"label": "C2", "x": 30, "y": 40, "tier": 0},
        {"label": "C3", "x": 50, "y": 50, "tier": 1},
    ]
    a = agent._apply_guards(
        {"action": "ACTION6", "x": None, "y": None, "why": "C2"},
        frame_with_6,
        candidates=cands,
    )
    x, y = _get_xy(a)
    _check(
        "ACTION6 why=C2 uses candidates[2]",
        int(a.value) == 6 and x == 30 and y == 40,
        detail=f"got value={a.value} xy=({x},{y})",
    )

    # Case D: ACTION6 with no coords and 1 candidate → uses candidates[0].
    a = agent._apply_guards(
        {"action": "ACTION6"},
        frame_with_6,
        candidates=[{"label": "C0", "x": 12, "y": 8, "tier": 0}],
    )
    x, y = _get_xy(a)
    _check(
        "ACTION6 no-coords-with-candidates uses C0",
        int(a.value) == 6 and x == 12 and y == 8,
        detail=f"got xy=({x},{y})",
    )

    # Case E: ACTION6 with no coords and no candidates → (32, 32).
    a = agent._apply_guards({"action": "ACTION6"}, frame_with_6, candidates=[])
    x, y = _get_xy(a)
    _check(
        "ACTION6 no-coords-no-candidates defaults to (32,32)",
        int(a.value) == 6 and x == 32 and y == 32,
        detail=f"got xy=({x},{y})",
    )

    # Case F: malformed action (no number) → snap to lowest available.
    a = agent._apply_guards({"action": "garbage"}, frame_simple, candidates=[])
    _check("malformed action snaps to ACTION1", int(a.value) == 1)


# ---------------------------------------------------------------------------
# Suite 4: state-graph integration via PhaseModelStub (model-free agent)
# ---------------------------------------------------------------------------


def _make_phase1_model_stub(reply_factory):
    """Subclass QwenAgent and replace _choose_action_inner with a model-free
    variant: same state-graph integration as the real path, but the model call
    is replaced by `reply_factory(step_idx, frame) -> str`."""
    from agents import GameAction, GameState
    from agents.qwen_agent import QwenAgent, _parse_json_first
    from agents.state_graph import hash_frame
    from agents.trigger_bfs_agent import _trigger_score

    class PhaseModelStub(QwenAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._step_idx = 0
            self._reply_factory = reply_factory
            self._loaded = True  # skip _ensure_model_loaded

        def _ensure_model_loaded(self):  # type: ignore[override]
            pass

        def _choose_action_inner(self, frame):  # type: ignore[override]
            state = getattr(frame, "state", None)
            if state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                self._record_history(frame, "RESET")
                return GameAction.RESET

            self._update_history_from_observation(frame)
            cur_layers = list(getattr(frame, "frame", []) or [])
            cur_hash = hash_frame(cur_layers) if cur_layers else b"\x00" * 8
            cur_levels = int(getattr(frame, "levels_completed", 0))

            if cur_levels >= 0 and cur_levels != self._prev_levels:
                self._state_graph.maybe_reset_for_level(cur_levels)
                self._outcome_history.clear()

            avail = list(getattr(frame, "available_actions", []) or [1, 2, 3, 4, 5, 6, 7])
            node = self._state_graph.add_or_get(
                cur_hash, available_actions=avail, levels=cur_levels
            )

            if self._prev_hash is not None and self._prev_action_id is not None:
                change_score = _trigger_score(
                    self._prev_layers, cur_layers, self._prev_levels, cur_levels
                )
                self._state_graph.observe(
                    self._prev_hash, self._prev_action_id, cur_hash, change_score
                )
                level_delta = cur_levels - self._prev_levels
                self._outcome_history.append(
                    (
                        self._prev_action_name or f"ACTION{self._prev_action_id}",
                        float(change_score),
                        int(level_delta),
                    )
                )

            avail_set = {int(a) for a in avail}
            candidates = self._segment_action6_candidates(cur_layers) if 6 in avail_set else []

            reply = self._reply_factory(self._step_idx, frame)
            self._step_idx += 1
            parsed = _parse_json_first(reply)
            if parsed is None:
                self._parse_failure_count += 1
                parsed = {"action": reply}

            action = self._apply_guards(parsed, frame, candidates, node)

            layers = getattr(frame, "frame", None) or []
            grid = layers[0] if layers else None
            self._prev_grid_hash = self._grid_hash(grid)
            self._prev_action_name = action.name
            self._prev_action_id = int(action.value)
            self._prev_layers = cur_layers
            self._prev_levels = cur_levels
            self._prev_hash = cur_hash
            data = getattr(action, "_data", {}) or {}
            self._state_graph.record_action(cur_hash, int(action.value), data)
            return action

    return PhaseModelStub


def suite_state_graph_integration() -> None:
    print("\n[suite] state_graph integration (model-stubbed)")
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        print("  [info] numpy not installed; skipping state-graph integration suite")
        return

    def rotation_reply(step_idx, frame):
        avail = list(frame.available_actions or [1])
        a = avail[step_idx % len(avail)]
        return f'{{"action":"ACTION{a}","x":null,"y":null,"why":"r{step_idx}"}}'

    stub_cls = _make_phase1_model_stub(rotation_reply)
    agent = stub_cls()
    frame = _make_mock_frame(available=[1, 2, 3])

    for _ in range(3):
        agent.choose_action(frame)

    _check(
        "state graph has ≥1 node after 3 same-state actions",
        len(agent._state_graph.nodes) >= 1,
        detail=f"nodes={len(agent._state_graph.nodes)}",
    )
    _check(
        "action_history records 3 actions",
        len(agent._state_graph.action_history) == 3,
        detail=f"len={len(agent._state_graph.action_history)}",
    )

    frame_l1 = _make_mock_frame(available=[1, 2, 3], levels=1)
    agent.choose_action(frame_l1)

    _check(
        "level transition resets graph (≤2 nodes after L0→L1)",
        len(agent._state_graph.nodes) <= 2,
        detail=f"nodes={len(agent._state_graph.nodes)}",
    )
    _check(
        "outcome_history cleared on level transition",
        len(agent._outcome_history) <= 1,
        detail=f"outcome_history={len(agent._outcome_history)}",
    )


# ---------------------------------------------------------------------------
# Suite 5: fallback path
# ---------------------------------------------------------------------------


def suite_fallback_path() -> None:
    print("\n[suite] fallback path (raises in _choose_action_inner)")
    from agents.qwen_agent import QwenAgent

    class ExplodingQwen(QwenAgent):
        def _choose_action_inner(self, frame):
            raise RuntimeError("synthetic OOM")

    agent = ExplodingQwen()
    frame = _make_mock_frame(available=[1, 2, 3, 6])

    initial_fb = agent._fallback_count
    action = agent.choose_action(frame)

    _check(
        "fallback returns a GameAction",
        hasattr(action, "value"),
        detail=f"got {type(action).__name__}",
    )
    _check(
        "fallback returned action is in available_actions",
        int(action.value) in [1, 2, 3, 6, 0],
        detail=f"action_id={action.value}",
    )
    _check(
        "fallback counter incremented",
        agent._fallback_count == initial_fb + 1,
        detail=f"fb_count={agent._fallback_count}",
    )


# ---------------------------------------------------------------------------
# Suite 6: history truncation
# ---------------------------------------------------------------------------


def suite_history_truncation() -> None:
    print("\n[suite] history truncation")
    from agents.qwen_agent import QwenAgent

    agent = QwenAgent(history_len=4)
    _check(
        "outcome_history maxlen respects history_len",
        agent._outcome_history.maxlen == 4,
    )
    for i in range(10):
        agent._outcome_history.append((f"ACTION{i % 5 + 1}", float(i), 0))
    _check(
        "outcome_history truncates to 4",
        len(agent._outcome_history) == 4,
        detail=f"len={len(agent._outcome_history)}",
    )


# ---------------------------------------------------------------------------
# Suite 7: reset_counters
# ---------------------------------------------------------------------------


def suite_reset_counters() -> None:
    print("\n[suite] reset_counters")
    from agents.qwen_agent import QwenAgent

    agent = QwenAgent()
    agent._fallback_count = 7
    agent._parse_failure_count = 3
    agent.reset_counters()
    _check("fallback_count reset to 0", agent._fallback_count == 0)
    _check("parse_failure_count reset to 0", agent._parse_failure_count == 0)


# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 72)
    print("Qwen Phase-1 local smoke (no GPU, no model load, no torch)")
    print("=" * 72)
    suite_parse_json_first()
    suite_segment_action6_candidates()
    suite_apply_guards()
    suite_state_graph_integration()
    suite_fallback_path()
    suite_history_truncation()
    suite_reset_counters()
    print("\n" + "-" * 72)
    n_fail = _check.failures  # type: ignore[attr-defined]
    if n_fail:
        print(f"FAILED ({n_fail} check(s) failed)")
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
