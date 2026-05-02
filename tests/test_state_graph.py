"""Unit tests for agents/state_graph.py.

Covers (per SPEC_4WEEKS.md §1.4):
1. hash_frame determinism (same input -> same digest)
2. hash_frame distinctness (different input -> different digest)
3. add_or_get seeds untried actions from available_actions (RESET excluded)
4. observe() removes the action from untried set + drops empty frontier nodes
5. maybe_reset_for_level clears all state on level transition
"""

from __future__ import annotations

import numpy as np

from agents.state_graph import StateGraph, hash_frame


def _grid(seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 16, size=(64, 64), dtype=np.uint8)]


def test_hash_determinism():
    g = _grid(42)
    h1 = hash_frame(g)
    h2 = hash_frame([np.copy(g[0])])
    assert h1 == h2
    assert isinstance(h1, bytes)
    assert len(h1) == 8


def test_hash_distinctness():
    g1 = _grid(0)
    g2 = _grid(1)
    assert hash_frame(g1) != hash_frame(g2)


def test_add_or_get_seeds_untried():
    sg = StateGraph()
    h = b"\x00" * 8
    node = sg.add_or_get(h, available_actions=[0, 1, 2, 6], levels=0)
    # RESET (id 0) is excluded; remaining {1, 2, 6}
    assert node.untried_actions == {1, 2, 6}
    assert h in sg.frontier


def test_observe_removes_untried_and_drains_frontier():
    sg = StateGraph()
    h_a = b"a" * 8
    h_b = b"b" * 8
    sg.add_or_get(h_a, available_actions=[1], levels=0)  # only one action
    sg.add_or_get(h_b, available_actions=[1, 2], levels=0)
    sg.observe(h_a, 1, h_b, change_score=1.0)
    # Now h_a has no untried -> dropped from frontier
    assert h_a not in sg.frontier
    assert sg.nodes[h_a].edges == {1: h_b}
    assert sg.nodes[h_b].visit_count == 1
    assert sg.nodes[h_b].incoming_change_score == 1.0


def test_level_transition_resets_graph():
    sg = StateGraph()
    h = b"x" * 8
    sg.add_or_get(h, available_actions=[1, 2], levels=0)
    sg.record_action(h, 1, {})
    assert sg.stats()["n_nodes"] == 1
    reset = sg.maybe_reset_for_level(1)
    assert reset is True
    assert sg.stats()["n_nodes"] == 0
    assert sg.stats()["n_actions_recorded"] == 0
    assert sg.current_levels == 1
    # No-op if already at this level
    assert sg.maybe_reset_for_level(1) is False
