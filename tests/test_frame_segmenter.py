"""Unit tests for agents/frame_segmenter.py.

Covers (per SPEC_4WEEKS.md §2.3 D10):
1. segment_frame correctly flood-fills a known 2-blob grid (4-connectivity)
2. is_rectangle is True for axis-aligned blobs, False for L-shape
3. twin detection on a row of 4 same-shape blobs
4. identify_status_bars detects a top horizontal line (rule a)
5. identify_status_bars detects 4 vertical-edge dots (rule b)
6. frame_segments_to_priority_tiers stratifies correctly across all 5 tiers
7. hash_masked_frame is deterministic + shape-aware + ignores masked pixels
8. mask_to_click_coords samples uniformly inside the chosen segment only
9. salient_pixels_in_segment returns only tier-0 pixels in a known layout
"""

from __future__ import annotations

import random

import numpy as np

from agents.frame_segmenter import (
    FRAME_SIZE,
    MAXIMAL_WIDTH,
    MINIMAL_WIDTH,
    Segment,
    frame_segments_to_priority_tiers,
    hash_masked_frame,
    identify_status_bars,
    mask_to_click_coords,
    salient_pixels_in_segment,
    segment_frame,
)


def _zeros(size: int = 16) -> np.ndarray:
    return np.zeros((size, size), dtype=np.uint8)


def test_segment_frame_two_blobs():
    frame = _zeros(16)
    # Two non-adjacent rectangles of different colors.
    frame[2:5, 2:5] = 8  # red 3x3 at top-left
    frame[10:13, 10:13] = 11  # yellow 3x3 at bottom-right
    label_map, segments = segment_frame(frame)
    # 3 segments: background (color 0) + the 2 blobs.
    assert len(segments) == 3
    colors = sorted(s.color for s in segments)
    assert colors == [0, 8, 11]
    # The 3x3 blobs are rectangles.
    blob_segs = [s for s in segments if s.color in (8, 11)]
    assert all(s.area == 9 for s in blob_segs)
    assert all(s.is_rectangle for s in blob_segs)
    # label_map must agree with the segment index.
    for sid, seg in enumerate(segments):
        assert (label_map == sid).sum() == seg.area


def test_segment_frame_l_shape_not_rectangle():
    frame = _zeros(8)
    # An L-shaped blob (color 9) of area 5 with bbox 3x3 -> not a rectangle.
    frame[1, 1:4] = 9
    frame[2:4, 1] = 9
    _, segments = segment_frame(frame)
    nine = next(s for s in segments if s.color == 9)
    assert nine.area == 5
    assert nine.bounding_box == (1, 1, 3, 3)
    assert nine.is_rectangle is False


def test_twin_detection_row_of_four():
    frame = _zeros(16)
    # Four 1x1 dots, same color, same shape -> all 4 are mutually twins.
    for x in (2, 5, 8, 11):
        frame[14, x] = 12
    _, segments = segment_frame(frame)
    dots = [s for s in segments if s.color == 12]
    assert len(dots) == 4
    # Each dot has 3 twins (the other three).
    assert all(s.number_of_twins == 3 for s in dots)


def test_status_bar_line_top_edge():
    frame = _zeros(FRAME_SIZE)
    # 1-px horizontal line on the very top edge, full width.
    frame[1, 0:FRAME_SIZE] = 14
    label_map, segments = segment_frame(frame)
    mask, bars = identify_status_bars(label_map, segments)
    assert mask.shape == (FRAME_SIZE, FRAME_SIZE)
    # The line is flagged.
    assert mask[1, 0]
    assert mask[1, FRAME_SIZE - 1]
    # And background pixels far from the edge are not.
    assert not mask[30, 30]
    assert len(bars) >= 1


def test_status_bar_dots_left_edge():
    frame = _zeros(FRAME_SIZE)
    # 4 vertical dots on the left edge -> dot status bar (rule b).
    for y in (5, 15, 25, 35):
        frame[y, 0] = 8
    label_map, segments = segment_frame(frame)
    mask, bars = identify_status_bars(label_map, segments)
    # All four dots are masked.
    assert mask[5, 0] and mask[15, 0] and mask[25, 0] and mask[35, 0]
    # And we found one dot-group with 4 segments.
    dot_groups = [g for g in bars if len(g) == 4]
    assert len(dot_groups) == 1


def test_priority_tiers_full_stratification():
    # Construct synthetic Segments directly (covers all 5 tiers cleanly).
    segs = [
        Segment((0, 0, 4, 4), color=11, area=25, is_rectangle=True),  # tier 0: salient + medium
        Segment(
            (10, 10, 14, 14), color=2, area=25, is_rectangle=True
        ),  # tier 1: medium non-salient
        Segment(
            (20, 20, 22, 60), color=9, area=120, is_rectangle=True
        ),  # tier 2: salient + extreme (height>32 -> not medium)
        Segment(
            (30, 30, 30, 30), color=3, area=1, is_rectangle=True
        ),  # tier 3: tiny non-salient (width 1 -> below MIN)
        Segment(
            (0, 60, 63, 63), color=0, area=200, is_rectangle=True
        ),  # tier 4: status bar (forced via sb id below)
    ]
    sb_ids = {4}
    tiers = frame_segments_to_priority_tiers(segs, status_bar_segment_ids=sb_ids)
    assert tiers[0] == {0}
    assert tiers[1] == {1}
    assert tiers[2] == {2}
    assert tiers[3] == {3}
    assert tiers[4] == {4}


def test_priority_tiers_constants_consistent():
    # Sanity-check the constants did not drift from the paper.
    assert MINIMAL_WIDTH == 2
    assert MAXIMAL_WIDTH == 32
    assert FRAME_SIZE == 64


def test_hash_masked_frame_deterministic_and_mask_aware():
    a = _zeros(FRAME_SIZE)
    a[10:15, 10:15] = 7
    b = a.copy()
    h_a = hash_masked_frame(a)
    h_b = hash_masked_frame(b)
    assert h_a == h_b
    assert len(h_a) == 16
    # Now alter a single pixel in the top-left corner; if we mask that
    # corner, the hashes still match. Without the mask, they differ.
    c = a.copy()
    c[0, 0] = 11
    assert hash_masked_frame(a) != hash_masked_frame(c)
    mask = np.zeros_like(a, dtype=bool)
    mask[0, 0] = True
    assert hash_masked_frame(a, mask) == hash_masked_frame(c, mask)


def test_hash_masked_frame_shape_aware():
    flat = np.zeros((1, 4096), dtype=np.uint8)
    square = np.zeros((64, 64), dtype=np.uint8)
    assert hash_masked_frame(flat) != hash_masked_frame(square)


def test_mask_to_click_coords_in_segment_only():
    frame = _zeros(FRAME_SIZE)
    frame[20:25, 30:35] = 8  # 5x5 red blob at (x=30..34, y=20..24)
    label_map, segments = segment_frame(frame)
    # find segment id of the red blob
    red_sid = next(sid for sid, s in enumerate(segments) if s.color == 8)
    rng = random.Random(123)
    for _ in range(200):
        coord = mask_to_click_coords(label_map, red_sid, rng=rng)
        assert coord is not None
        x, y = coord
        assert 30 <= x <= 34
        assert 20 <= y <= 24


def test_salient_pixels_tier_zero():
    frame = _zeros(FRAME_SIZE)
    frame[5:10, 5:10] = 11  # tier 0 (salient + 5x5 medium)
    frame[40:42, 40:42] = 3  # tier 3 (non-salient + 2x2 medium? no - 2x2 IS medium since MIN=2)
    label_map, segments = segment_frame(frame)
    tiers = frame_segments_to_priority_tiers(segments)
    pixels = salient_pixels_in_segment(label_map, segments, tier=0, tiers=tiers)
    # All returned pixels must belong to the yellow blob.
    assert pixels.shape[1] == 2
    assert len(pixels) == 25
    for y, x in pixels:
        assert 5 <= y < 10
        assert 5 <= x < 10
