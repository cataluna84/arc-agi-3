"""Phase-1 unit tests for QwenAgent.

Covers state-graph integration, ACTION6 candidate generation, guard
snapping, fallback wiring, level transitions, and history truncation.

The model itself is never loaded; tests work against the agent's pure-Python
surface using a model-free stub for the few places where the parent agent
calls into the LLM.
"""

from __future__ import annotations

import pytest


def _get_xy(action):
    """Extract (x, y) from an action: works for arcengine and mock IntEnum."""
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


def _make_grid_with_blob(blob_color: int = 7, top_left=(28, 28), size: int = 6):
    grid = [[0] * 64 for _ in range(64)]
    y0, x0 = top_left
    for y in range(y0, min(64, y0 + size)):
        for x in range(x0, min(64, x0 + size)):
            grid[y][x] = blob_color
    return grid


def _make_frame(*, available=None, levels: int = 0, grid=None):
    from agents import GameState, MockFrame

    return MockFrame(
        game_id="ls20-mock",
        state=GameState.NOT_FINISHED,
        levels_completed=levels,
        win_levels=3,
        available_actions=list(available) if available is not None else [1, 2, 3, 4, 6],
        frame=[grid if grid is not None else _make_grid_with_blob()],
    )


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def test_segment_action6_candidates_returns_at_most_k():
    pytest.importorskip("numpy")
    from agents.qwen_agent import QwenAgent

    agent = QwenAgent()
    grid = [[0] * 64 for _ in range(64)]
    # 12 small blobs of different colors → expect candidates capped at 8.
    for i in range(12):
        y0 = (i * 5) % 60
        x0 = (i * 7) % 60
        for y in range(y0, min(64, y0 + 3)):
            for x in range(x0, min(64, x0 + 3)):
                grid[y][x] = 6 + (i % 10)
    cands = agent._segment_action6_candidates([grid])
    assert 0 < len(cands) <= 8
    # Labels are C0..C7 in order.
    for idx, c in enumerate(cands):
        assert c["label"] == f"C{idx}"


def test_segment_action6_candidates_empty_grid():
    from agents.qwen_agent import QwenAgent

    agent = QwenAgent()
    assert agent._segment_action6_candidates([]) == []
    assert agent._segment_action6_candidates(None) == []  # type: ignore[arg-type]


def test_segment_action6_candidates_coords_within_bounds():
    pytest.importorskip("numpy")
    from agents.qwen_agent import QwenAgent

    agent = QwenAgent()
    grid = _make_grid_with_blob(blob_color=7, top_left=(20, 20), size=4)
    cands = agent._segment_action6_candidates([grid])
    assert len(cands) >= 1
    for c in cands:
        assert 0 <= c["x"] < 64
        assert 0 <= c["y"] < 64
        assert 0 <= c["tier"] <= 3


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_apply_guards_snaps_unavailable_action():
    from agents.qwen_agent import QwenAgent

    agent = QwenAgent()
    frame = _make_frame(available=[1, 2, 3])
    action = agent._apply_guards({"action": "ACTION9"}, frame, candidates=[])
    assert int(action.value) == 1  # lowest non-RESET available


def test_apply_guards_fills_action6_from_candidate():
    from agents.qwen_agent import QwenAgent

    agent = QwenAgent()
    frame = _make_frame(available=[1, 2, 6])
    candidates = [{"label": "C0", "x": 10, "y": 20, "tier": 0}]
    action = agent._apply_guards({"action": "ACTION6"}, frame, candidates=candidates)
    assert int(action.value) == 6
    x, y = _get_xy(action)
    assert x == 10
    assert y == 20


def test_apply_guards_respects_why_label():
    from agents.qwen_agent import QwenAgent

    agent = QwenAgent()
    frame = _make_frame(available=[1, 6])
    candidates = [
        {"label": "C0", "x": 1, "y": 1, "tier": 0},
        {"label": "C1", "x": 2, "y": 2, "tier": 0},
        {"label": "C2", "x": 30, "y": 40, "tier": 0},
        {"label": "C3", "x": 50, "y": 50, "tier": 1},
    ]
    action = agent._apply_guards(
        {"action": "ACTION6", "why": "C2"},
        frame,
        candidates=candidates,
    )
    x, y = _get_xy(action)
    assert int(action.value) == 6
    assert x == 30
    assert y == 40


def test_apply_guards_action6_explicit_xy_honored():
    from agents.qwen_agent import QwenAgent

    agent = QwenAgent()
    frame = _make_frame(available=[1, 6])
    action = agent._apply_guards(
        {"action": "ACTION6", "x": 5, "y": 7, "why": "x"},
        frame,
        candidates=[],
    )
    x, y = _get_xy(action)
    assert int(action.value) == 6
    assert x == 5
    assert y == 7


def test_apply_guards_skips_known_no_change():
    """If action `chosen` has a recorded edge to a successor with
    visit_count >= 1 and incoming_change_score == 0, rotate to an untried."""
    from agents.qwen_agent import QwenAgent
    from agents.state_graph import StateNode

    agent = QwenAgent()
    frame = _make_frame(available=[1, 2, 3])

    # Build a synthetic node where ACTION1 leads to a known-no-change successor.
    cur_hash = b"\x01" * 8
    succ_hash = b"\x02" * 8
    node = StateNode(
        state_hash=cur_hash,
        visit_count=1,
        untried_actions={2, 3},
        edges={1: succ_hash},
        incoming_change_score=0.0,
        last_levels=0,
    )
    succ = StateNode(
        state_hash=succ_hash,
        visit_count=1,
        untried_actions=set(),
        edges={},
        incoming_change_score=0.0,
        last_levels=0,
    )
    agent._state_graph.nodes[cur_hash] = node
    agent._state_graph.nodes[succ_hash] = succ

    # Model picks ACTION1, but ACTION1 has known no-change → rotate to ACTION2.
    action = agent._apply_guards({"action": "ACTION1"}, frame, candidates=[], node=node)
    assert int(action.value) == 2  # smallest untried in avail


# ---------------------------------------------------------------------------
# Fallback wiring
# ---------------------------------------------------------------------------


def test_fallback_wires_trigger_bfs():
    """If _choose_action_inner raises, the outer choose_action returns a valid
    action via the TriggerBFS fallback."""
    from agents.qwen_agent import QwenAgent

    class ExplodingQwen(QwenAgent):
        def _choose_action_inner(self, frame):
            raise RuntimeError("synthetic OOM")

    agent = ExplodingQwen()
    frame = _make_frame(available=[1, 2, 3, 6])
    initial = agent._fallback_count
    action = agent.choose_action(frame)
    # Must return some GameAction object.
    assert hasattr(action, "value")
    # Must be in available_actions (RESET=0 also acceptable as last-ditch).
    assert int(action.value) in {0, 1, 2, 3, 6}
    assert agent._fallback_count == initial + 1


# ---------------------------------------------------------------------------
# Level transition
# ---------------------------------------------------------------------------


def test_level_transition_resets_state_graph():
    """When levels_completed changes (>= 0), the state graph and outcome
    history are cleared."""
    from agents.qwen_agent import QwenAgent
    from agents.state_graph import hash_frame

    agent = QwenAgent()
    # Pre-populate the graph and outcome history.
    h = hash_frame([[[1] * 64 for _ in range(64)]])
    agent._state_graph.add_or_get(h, available_actions=[1, 2, 3], levels=0)
    agent._state_graph.record_action(h, 1, {})
    agent._outcome_history.append(("ACTION1", 10.0, 0))
    agent._prev_levels = 0

    # Simulate the level-transition guard (gotcha #19).
    cur_levels = 1
    if cur_levels >= 0 and cur_levels != agent._prev_levels:
        agent._state_graph.maybe_reset_for_level(cur_levels)
        agent._outcome_history.clear()

    assert len(agent._state_graph.nodes) == 0
    assert len(agent._outcome_history) == 0


# ---------------------------------------------------------------------------
# History truncation
# ---------------------------------------------------------------------------


def test_outcome_history_truncates_to_history_len():
    from agents.qwen_agent import QwenAgent

    agent = QwenAgent(history_len=4)
    for i in range(10):
        agent._outcome_history.append((f"ACTION{i % 5 + 1}", float(i), 0))
    assert len(agent._outcome_history) == 4


# ---------------------------------------------------------------------------
# reset_counters
# ---------------------------------------------------------------------------


def test_reset_counters_zeroes_smoke_counters():
    from agents.qwen_agent import QwenAgent

    agent = QwenAgent()
    agent._fallback_count = 11
    agent._parse_failure_count = 5
    agent.reset_counters()
    assert agent._fallback_count == 0
    assert agent._parse_failure_count == 0
