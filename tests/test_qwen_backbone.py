"""Unit tests for agents.qwen_backbone (parsers, change-log, frame helpers)."""

from __future__ import annotations

from agents.qwen_backbone import (
    ChangeLog,
    parse_action_id,
    parse_choice,
    parse_coords,
    parse_score,
    render_frame_text,
)


def test_parse_action_id_action3():
    assert parse_action_id("I will use ACTION3.", [1, 2, 3, 4, 5, 6, 7]) == 3


def test_parse_action_id_lowercase_with_separator():
    assert parse_action_id("action_5 looks promising", [1, 2, 3, 4, 5, 6, 7]) == 5


def test_parse_action_id_json_form():
    assert parse_action_id('{"action": 6, "x": 12, "y": 34}', [1, 2, 3, 4, 5, 6, 7]) == 6


def test_parse_action_id_filters_unavailable():
    # Model wants ACTION3 but it's not available; falls back to first available.
    assert parse_action_id("ACTION3", [1, 2, 4]) == 1


def test_parse_action_id_returns_lowest_available_when_unparseable():
    assert parse_action_id("unparseable garbage", [3, 5, 7]) == 3


def test_parse_action_id_empty_text_falls_back():
    assert parse_action_id("", [2, 4]) == 2


def test_parse_coords_labeled():
    assert parse_coords("x=12, y=34") == (12, 34)


def test_parse_coords_bare():
    assert parse_coords("ACTION6 (12, 34)") == (12, 34)


def test_parse_coords_clamped_to_64():
    assert parse_coords("x=128, y=200") == (0, 8)


def test_parse_coords_none_when_absent():
    assert parse_coords("just some text") is None


def test_parse_score_first_float():
    assert parse_score("score: 0.42") == 0.42


def test_parse_score_negative():
    assert parse_score("estimate -0.7 confidence") == -0.7


def test_parse_score_no_number_returns_zero():
    assert parse_score("nothing here") == 0.0


def test_parse_choice_first_match_wins():
    assert parse_choice("I prefer EXPLORE then maybe REPLAY", ["explore", "replay"]) == "explore"


def test_parse_choice_returns_none_if_no_match():
    assert parse_choice("neither word here", ["explore", "replay"]) is None


def test_change_log_caps_capacity():
    cl = ChangeLog(capacity=3)
    for i in range(5):
        cl.add(f"ACTION{i}", changed=False)
    assert len(cl) == 3
    rendered = cl.render()
    assert "ACTION2" in rendered and "ACTION3" in rendered and "ACTION4" in rendered
    assert "ACTION0" not in rendered and "ACTION1" not in rendered


def test_change_log_renders_change_dpx():
    cl = ChangeLog(capacity=3)
    cl.add("ACTION1", changed=True, dpx=14)
    rendered = cl.render()
    assert "ACTION1" in rendered and "+14 px" in rendered


def test_change_log_renders_level_up():
    cl = ChangeLog(capacity=3)
    cl.add("ACTION6", changed=True, dpx=200, dlevels=1)
    rendered = cl.render()
    assert "LEVEL UP" in rendered and "+1" in rendered


def test_change_log_empty_render():
    cl = ChangeLog()
    assert "no actions yet" in cl.render()


def test_render_frame_text_64x64():
    grid = [[i % 16 for i in range(64)] for _ in range(64)]
    out = render_frame_text(grid)
    rows = out.split("\n")
    assert len(rows) == 64
    assert all(len(r) == 64 for r in rows)
    # Every char must be a hex digit
    assert all(c in "0123456789abcdef" for r in rows for c in r)


def test_render_frame_text_none_returns_empty():
    assert render_frame_text(None) == ""
