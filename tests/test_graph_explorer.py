from __future__ import annotations

import random

import numpy as np

from agents import GameState, MockFrame
from agents.graph_explorer import ActionKey, FrameAnalysis, GraphExplorer
from agents.graph_explorer_agent import GraphExplorerAgent


def _frame_with_objects(*objects: tuple[int, int, int, int]) -> np.ndarray:
    grid = np.zeros((64, 64), dtype=np.uint8)
    for x, y, size, color in objects:
        grid[y : y + size, x : x + size] = color
    return grid


def test_candidate_seeding_excludes_reset_and_prioritizes_simple_actions() -> None:
    explorer = GraphExplorer(random.Random(0))
    grid = _frame_with_objects((10, 10, 3, 7))
    analysis = explorer.analyze_layers([grid])

    node = explorer.add_or_get_state(analysis.state_hash, [0, 1, 6], 0, analysis)

    assert ActionKey(1) in node.candidates
    assert all(key.action_id != 0 for key in node.candidates)
    assert node.candidates[ActionKey(1)].priority == 0
    assert any(key.action_id == 6 and key.target for key in node.candidates)


def test_action6_segments_are_tracked_as_separate_candidates() -> None:
    explorer = GraphExplorer(random.Random(0))
    grid = _frame_with_objects((10, 10, 3, 7), (30, 30, 3, 8))
    analysis = explorer.analyze_layers([grid])
    node = explorer.add_or_get_state(analysis.state_hash, [6], 0, analysis)

    click_keys = [key for key in node.candidates if key.action_id == 6]

    assert len(click_keys) >= 2
    assert len({key.target for key in click_keys}) == len(click_keys)


def test_observe_marks_only_the_selected_action6_candidate() -> None:
    explorer = GraphExplorer(random.Random(0))
    grid = _frame_with_objects((10, 10, 3, 7), (30, 30, 3, 8))
    analysis = explorer.analyze_layers([grid])
    node = explorer.add_or_get_state(analysis.state_hash, [6], 0, analysis)
    click_keys = sorted(key for key in node.candidates if key.action_id == 6)

    explorer.observe(analysis.state_hash, click_keys[0], b"next-state", 1.0)

    assert click_keys[0] not in node.untested
    assert click_keys[1] in node.untested


def test_threshold_increments_to_next_priority_group() -> None:
    explorer = GraphExplorer(random.Random(0))
    grid = _frame_with_objects((10, 10, 3, 1))
    analysis = explorer.analyze_layers([grid])
    explorer.add_or_get_state(analysis.state_hash, [6], 0, analysis)

    candidate = explorer.next_action(analysis.state_hash, [6])

    assert candidate.action_id == 6
    assert candidate.priority == 1
    assert explorer.current_priority == 1


def test_shortest_path_returns_first_edge_toward_frontier() -> None:
    explorer = GraphExplorer(random.Random(0))
    state_a = b"a" * 16
    state_b = b"b" * 16
    node_a = explorer.add_or_get_state(
        state_a,
        [1],
        0,
        FrameAnalysis(state_hash=state_a),
    )
    explorer.add_or_get_state(
        state_b,
        [2],
        0,
        FrameAnalysis(state_hash=state_b),
    )
    key = ActionKey(1)
    explorer.observe(state_a, key, state_b, 1.0)

    candidate = explorer.next_action(state_a, [1])

    assert candidate.key == key
    assert key not in node_a.untested


def test_masked_hash_is_stable_for_status_bar_color_changes() -> None:
    explorer = GraphExplorer(random.Random(0))
    first = np.zeros((64, 64), dtype=np.uint8)
    second = first.copy()
    first[0, :] = 5
    second[0, :] = 6

    h1 = explorer.analyze_layers([first]).state_hash
    h2 = explorer.analyze_layers([second]).state_hash

    assert h1 == h2


def test_graph_explorer_agent_returns_legal_action_with_click_data() -> None:
    grid = _frame_with_objects((10, 10, 3, 7))
    frame = MockFrame(
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        available_actions=[6],
        frame=[grid],
    )
    agent = GraphExplorerAgent(seed=0)

    action = agent.choose_action(frame)

    action_id = int(action.value) if hasattr(action, "value") else int(action)
    assert action_id == 6
    data = getattr(action, "action_data", None) or getattr(action, "_data", None)
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    assert 0 <= int(data["x"]) < 64
    assert 0 <= int(data["y"]) < 64
