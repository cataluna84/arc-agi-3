"""Unit tests for agents/trigger_bfs_mcts_agent.py (exp011).

Coverage:
1. UCB1 returns +inf for unvisited keys.
2. UCB1 prefers higher Q given equal visits.
3. _action_bucket_keys with mixed avail + K candidates.
4. _action_bucket_keys with ACTION6 only.
5. _segment_click_candidates length cap (<=14).
6. _segment_click_candidates includes fallbacks even when segmenter is empty.
7. _segment_click_candidates handles empty/None layers safely.
8. _backup increments visits + accumulates score (+ updates total).
9. Level transition resets the MCTS stats AND state graph.
10. End-to-end: 5 calls on a MockFrame do not crash; returned actions are valid.
"""

from __future__ import annotations

import math

import numpy as np

from agents import GameAction, GameState, MockFrame
from agents.trigger_bfs_mcts_agent import (
    MCTSStat,
    TriggerBFSMCTSAgent,
    _action_bucket_keys,
    _segment_click_candidates,
    _ucb1_score,
)


def test_ucb1_returns_inf_for_unvisited():
    assert _ucb1_score(None, total_visits=10, c_uct=2.0) == math.inf
    s = MCTSStat(n_visits=0, sum_change_score=0.0)
    assert _ucb1_score(s, total_visits=10, c_uct=2.0) == math.inf


def test_ucb1_prefers_higher_q_at_equal_visits():
    s_lo = MCTSStat(n_visits=5, sum_change_score=1.0)  # Q = 0.2
    s_hi = MCTSStat(n_visits=5, sum_change_score=10.0)  # Q = 2.0
    score_lo = _ucb1_score(s_lo, total_visits=20, c_uct=2.0)
    score_hi = _ucb1_score(s_hi, total_visits=20, c_uct=2.0)
    assert score_hi > score_lo


def test_action_bucket_keys_mixed():
    # avail = [1, 2, 3, 6] with K=3 candidates -> 3 simple + 3 click = 6 keys.
    candidates = [
        {"label": "C0", "x": 10, "y": 20, "tier": 0, "segment_id": 0},
        {"label": "C1", "x": 30, "y": 40, "tier": 0, "segment_id": 1},
        {"label": "C2", "x": 32, "y": 32, "tier": 4, "segment_id": -1},
    ]
    keys = _action_bucket_keys([1, 2, 3, 6], candidates)
    assert len(keys) == 6
    assert (1, -1) in keys
    assert (2, -1) in keys
    assert (3, -1) in keys
    assert (6, 0) in keys
    assert (6, 1) in keys
    assert (6, 2) in keys


def test_action_bucket_keys_action6_only():
    candidates = [{"label": "C0", "x": 0, "y": 0, "tier": 4, "segment_id": -1}]
    keys = _action_bucket_keys([6], candidates)
    assert keys == [(6, 0)]


def test_action_bucket_keys_excludes_reset():
    # RESET (0) must never appear in the MCTS key set.
    candidates: list[dict] = []
    keys = _action_bucket_keys([0, 1, 2], candidates)
    assert all(k[0] != 0 for k in keys)
    assert (1, -1) in keys
    assert (2, -1) in keys


def test_segment_click_candidates_caps_at_max():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 16, size=(64, 64), dtype=np.uint8)
    cands = _segment_click_candidates([frame])
    assert len(cands) <= 14
    # The trailing entry is the random-uniform sentinel (tier == 5).
    assert any(c["tier"] == 5 for c in cands)


def test_segment_click_candidates_includes_fallbacks_when_empty():
    # All-zero grid -> segmenter likely surfaces no salient tier-0/1
    # segments; the fallback path must still yield deterministic clicks.
    flat = np.zeros((64, 64), dtype=np.uint8)
    cands = _segment_click_candidates([flat])
    # At least one fallback (tier == 4) + the random sentinel (tier == 5).
    tiers = {c["tier"] for c in cands}
    assert 4 in tiers
    assert 5 in tiers
    # (32, 32) center is the first deterministic fallback.
    assert any(c["x"] == 32 and c["y"] == 32 and c["tier"] == 4 for c in cands)


def test_segment_click_candidates_empty_layers_safe():
    # Both None and [] must not raise; both return only the safe-fallback
    # set (5 deterministic + 1 random sentinel).
    a = _segment_click_candidates(None)
    b = _segment_click_candidates([])
    assert len(a) == 6
    assert len(b) == 6
    assert {c["tier"] for c in a} == {4, 5}


def test_backup_increments_visits_and_total():
    agent = TriggerBFSMCTSAgent(seed=0)
    h = b"node-a" * 1
    key = (1, -1)
    agent._backup(h, key, 3.0)
    agent._backup(h, key, 5.0)
    stat = agent._mcts_stats[h][key]
    assert stat.n_visits == 2
    assert stat.sum_change_score == 8.0
    assert math.isclose(stat.q, 4.0)
    assert agent._mcts_total[h] == 2


def test_level_transition_resets_mcts():
    agent = TriggerBFSMCTSAgent(seed=0)
    g0 = np.zeros((64, 64), dtype=np.uint8)
    fr0 = MockFrame(
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        available_actions=[1, 2, 3],
        frame=[g0],
    )
    # Run 3 calls -> at least 1 prev_action_key + some backups happen.
    for _ in range(3):
        agent.choose_action(fr0)
    assert agent._mcts_stats  # non-empty
    # Now level-up: a new frame at levels_completed=1 must wipe MCTS state.
    g1 = np.copy(g0)
    g1[5, 5] = 9
    fr1 = MockFrame(
        state=GameState.NOT_FINISHED,
        levels_completed=1,
        available_actions=[1, 2, 3],
        frame=[g1],
    )
    agent.choose_action(fr1)
    # After the level transition reset, only the fresh post-reset node is
    # in the stats (the backup for the *previous-level* action key was
    # skipped because we cleared first).
    assert agent.graph.current_levels == 1
    # The MCTS stats dict was cleared inside the level-transition branch
    # before any new backup, so it stays empty until the NEXT call's
    # observe+backup phase.
    assert len(agent._mcts_stats) == 0
    assert len(agent._mcts_total) == 0


def test_end_to_end_5_calls_no_crash():
    agent = TriggerBFSMCTSAgent(seed=0)
    g = np.zeros((64, 64), dtype=np.uint8)
    g[10:15, 20:25] = 7  # something for the segmenter to bite on
    fr = MockFrame(
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        available_actions=[1, 2, 3, 4, 5, 6, 7],
        frame=[g],
    )
    for _ in range(5):
        a = agent.choose_action(fr)
        assert isinstance(a, GameAction)
        assert int(a.value) in (1, 2, 3, 4, 5, 6, 7)
        if a == GameAction.ACTION6:
            ad = getattr(a, "_data", None) or getattr(a, "action_data", None)
            if isinstance(ad, dict):
                x, y = ad.get("x", -1), ad.get("y", -1)
            else:
                x = getattr(ad, "x", -1) if ad is not None else -1
                y = getattr(ad, "y", -1) if ad is not None else -1
            assert 0 <= int(x) <= 63
            assert 0 <= int(y) <= 63


def test_not_played_returns_reset():
    agent = TriggerBFSMCTSAgent(seed=0)
    fr = MockFrame(state=GameState.NOT_PLAYED, available_actions=[1, 2, 6])
    a = agent.choose_action(fr)
    assert a == GameAction.RESET


def test_agent_name():
    agent = TriggerBFSMCTSAgent(seed=0)
    assert agent.name == "trigger-bfs-mcts"
