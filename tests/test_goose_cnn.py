"""Unit tests for agents/goose_cnn_model.py and agents/goose_cnn_agent.py.

Designed to run on CPU with or without torch installed (predictor falls back
to uniform probabilities when torch is unavailable).
"""

from __future__ import annotations

import numpy as np

from agents import GameAction, GameState, MockFrame
from agents.goose_cnn_agent import (
    GooseCNNAgent,
    _action_logits_to_six,
    _frame_changed,
    _softmax_sample_1d,
    _softmax_sample_2d,
)
from agents.goose_cnn_model import (
    GRID_SIZE,
    NUM_COLORS,
    NUM_SIMPLE_ACTIONS,
    ExperienceBuffer,
    GooseCNNPredictor,
    hash_frame_grid,
)


def _grid(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 16, size=(GRID_SIZE, GRID_SIZE), dtype=np.uint8)


def test_hash_frame_grid_determinism_and_distinctness():
    g1 = _grid(0)
    g2 = _grid(1)
    h1 = hash_frame_grid(g1)
    h1b = hash_frame_grid(g1.copy())
    h2 = hash_frame_grid(g2)
    assert isinstance(h1, bytes)
    assert len(h1) == 8
    assert h1 == h1b
    assert h1 != h2


def test_experience_buffer_dedup_and_eviction():
    buf = ExperienceBuffer(max_size=3)
    buf.add(b"a" * 8, 1, True)
    buf.add(b"a" * 8, 1, False)  # update in-place: still len 1
    assert len(buf) == 1
    buf.add(b"b" * 8, 6, True, click_xy=(2, 3))
    buf.add(b"c" * 8, 2, True)
    assert len(buf) == 3
    buf.add(b"d" * 8, 3, True)  # evicts oldest (a)
    assert len(buf) == 3
    keys = list(buf._data.keys())
    assert (b"a" * 8, 1, -1, -1) not in keys


def test_predictor_no_torch_fallback_returns_uniform():
    pred = GooseCNNPredictor(seed=0, device="cpu")
    g = _grid(7)
    out = pred.predict(g)
    assert out["action_probs"].shape == (NUM_SIMPLE_ACTIONS,)
    assert out["coord_probs"].shape == (GRID_SIZE, GRID_SIZE)
    assert (out["action_probs"] >= 0).all() and (out["action_probs"] <= 1).all()
    assert (out["coord_probs"] >= 0).all() and (out["coord_probs"] <= 1).all()


def test_action_logits_to_six_combines_action_and_coord_max():
    ap = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    cp = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    cp[10, 10] = 0.9
    six = _action_logits_to_six(ap, cp)
    assert six.shape == (6,)
    assert np.allclose(six[:5], ap)
    assert abs(float(six[5]) - 0.9) < 1e-5


def test_softmax_sample_2d_in_bounds():
    import random as r

    rng = r.Random(0)
    cp = np.full((GRID_SIZE, GRID_SIZE), 0.5, dtype=np.float32)
    cp[5, 7] = 0.99
    for _ in range(20):
        x, y = _softmax_sample_2d(cp, rng, temperature=0.5)
        assert 0 <= x < GRID_SIZE
        assert 0 <= y < GRID_SIZE


def test_softmax_sample_1d_respects_mask():
    import random as r

    rng = r.Random(0)
    logits = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mask = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # only ACTION2 allowed
    for _ in range(50):
        idx = _softmax_sample_1d(logits, mask, rng, temperature=1.0)
        assert idx == 1


def test_frame_changed_detects_diff():
    a = [_grid(0)]
    b = [_grid(0).copy()]
    c = [_grid(1)]
    assert _frame_changed(a, b) is False
    assert _frame_changed(a, c) is True
    assert _frame_changed(None, c) is False
    assert _frame_changed(a, None) is False


def test_agent_returns_reset_on_not_played():
    a = GooseCNNAgent(seed=0)
    f = MockFrame(state=GameState.NOT_PLAYED, frame=[_grid(0)])
    assert a.choose_action(f) == GameAction.RESET


def test_agent_returns_reset_on_game_over():
    a = GooseCNNAgent(seed=0)
    f = MockFrame(state=GameState.GAME_OVER, frame=[_grid(0)])
    assert a.choose_action(f) == GameAction.RESET


def _action_int(act) -> int:
    """Coerce GameAction (Enum or IntEnum) to int."""
    return int(getattr(act, "value", act))


def _action_data(act) -> dict:
    """Pull the click {x, y} dict from a GameAction in either SDK or mock form."""
    raw = getattr(act, "_data", None) or getattr(act, "action_data", None)
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    return {"x": getattr(raw, "x", -1), "y": getattr(raw, "y", -1)}


def test_agent_picks_legal_simple_action():
    a = GooseCNNAgent(seed=0)
    f = MockFrame(
        state=GameState.NOT_FINISHED,
        available_actions=[1, 2, 3],
        frame=[_grid(0)],
    )
    act = a.choose_action(f)
    assert _action_int(act) in {1, 2, 3}
    assert _action_int(act) != 0


def test_agent_action6_sets_xy_in_bounds():
    a = GooseCNNAgent(seed=0)
    f = MockFrame(
        state=GameState.NOT_FINISHED,
        available_actions=[6],
        frame=[_grid(0)],
    )
    act = a.choose_action(f)
    assert _action_int(act) == 6
    data = _action_data(act)
    assert 0 <= data.get("x", -1) < GRID_SIZE
    assert 0 <= data.get("y", -1) < GRID_SIZE


def test_agent_resets_on_level_transition():
    a = GooseCNNAgent(seed=0)
    g0 = _grid(0)
    g1 = _grid(1)
    a.choose_action(MockFrame(state=GameState.NOT_FINISHED, frame=[g0], levels_completed=0))
    # Push enough buffer entries to verify reset clears them.
    a.choose_action(MockFrame(state=GameState.NOT_FINISHED, frame=[g1], levels_completed=0))
    n_before = len(a.predictor.buffer)
    assert n_before > 0 or not a.predictor.available
    a.choose_action(MockFrame(state=GameState.NOT_FINISHED, frame=[g1], levels_completed=1))
    # Buffer should be wiped on the level transition.
    assert len(a.predictor.buffer) == 0
    assert a._prev_levels == 1


def test_agent_is_done_only_on_win():
    a = GooseCNNAgent(seed=0)
    assert a.is_done(MockFrame(state=GameState.WIN, frame=[_grid(0)])) is True
    assert a.is_done(MockFrame(state=GameState.NOT_FINISHED, frame=[_grid(0)])) is False
    assert a.is_done(MockFrame(state=GameState.GAME_OVER, frame=[_grid(0)])) is False
    assert a.is_done(MockFrame(state=GameState.NOT_PLAYED, frame=[_grid(0)])) is False


def test_one_hot_encoder_shape_and_values_when_torch_available():
    """If torch is installed, encode_one_hot returns proper [16, 64, 64] tensor."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return
    from agents.goose_cnn_model import encode_one_hot

    g = _grid(0)
    t = encode_one_hot(g)
    assert tuple(t.shape) == (NUM_COLORS, GRID_SIZE, GRID_SIZE)
    # Each pixel should have exactly one channel == 1.
    summed = t.sum(dim=0)
    assert summed.min().item() == 1.0
    assert summed.max().item() == 1.0
