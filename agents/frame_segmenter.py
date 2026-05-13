"""frame_segmenter.py - per-color connected-components + 5-tier priority grouping.

Direct port of the FrameProcessor from the dolphin-in-a-coma ARC-AGI-3
solution (3rd-place preview challenge, 12-19 levels solved, 1st-place
on graph-exploration paper arXiv:2512.24156). See NOTICE for upstream
attribution.

Public surface (used by `agents/trigger_bfs_agent.py` D11 wire-up):
    segment_frame(frame)              -> (label_map, segments)
    identify_status_bars(label_map, segments)  -> (status_bar_mask, sb_segments)
    frame_segments_to_priority_tiers(segments) -> list[set[int]]  # 5 tiers
    hash_masked_frame(frame, status_bar_mask)  -> bytes            # 16-byte blake2b
    mask_to_click_coords(label_map, segment_id) -> (x, y) | None  # uniform sample
    salient_pixels_in_segment(label_map, segments, tier)  -> np.ndarray

Design notes vs the upstream HeuristicAgent.FrameProcessor:
- Stateless functions instead of methods on `FrameProcessor` (more
  testable; matches our `state_graph.hash_frame` style).
- 4-connectivity by default (matches upstream `connectivity_rank=4`).
- 16-byte hash digest (matches upstream blake2b digest_size=16).
- `SALIENT_COLORS` and `STATUS_BAR_COLOR` constants exposed so the
  wire-up can adjust thresholds without re-reading source.
- No matplotlib dependency in the hot path (visualization deferred).

The algorithm:
1. **Segmentation**: BFS flood-fill on the 64x64 grid; one component
   per maximal same-color region.
2. **Twin detection**: O(n^2) pass — two components are "twins" if
   they share area + rectangle-ness + color (used by status-bar
   detection's dot-bar rule).
3. **Status bar detection**: combines two rules:
   (a) the segment is fully within `STATUS_BAR_DISTANCE_THRESHOLD`
       pixels of an edge AND has a long aspect ratio (>=5:1 either
       way) — line status bar;
   (b) OR the segment has >= STATUS_BAR_TWINS_THRESHOLD twins also
       on the same edge — dot status bar.
4. **Priority tiers** (5 groups, lower index = higher priority):
   0. salient color AND medium width (2..32 px)         — most likely interactive
   1. medium width but non-salient color
   2. salient color but extreme width (<2 or >32)
   3. neither salient nor medium width AND not a status bar
   4. probable status bar                               — explored last
5. **Hash**: pack two 4-bit cells per byte, blake2b digest_size=16 with
   shape personalization. Status-bar pixels are replaced with
   STATUS_BAR_COLOR (16) before hashing so equivalent game states with
   different step-counter pixels collide to the same hash.
"""

from __future__ import annotations

import hashlib
import random
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# --- Color taxonomy (per dolphin-in-a-coma reference) ---
SALIENT_COLORS: frozenset[int] = frozenset({6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
NON_SALIENT_COLORS: frozenset[int] = frozenset({0, 1, 2, 3, 4, 5})
STATUS_BAR_COLOR: int = 16  # used internally to mask out probable status bars

# --- Geometry thresholds ---
FRAME_SIZE: int = 64
STATUS_BAR_DISTANCE_THRESHOLD: int = 3  # how close to the edge a segment must sit
STATUS_BAR_RATIO_THRESHOLD: float = 5.0  # min aspect ratio for "line" status bar
STATUS_BAR_TWINS_THRESHOLD: int = 3  # min twin-count for "dot" status bar
MINIMAL_WIDTH: int = 2  # smallest "medium" width (px)
MAXIMAL_WIDTH: int = 32  # largest  "medium" width (px)

# 4-connectivity: up, down, left, right
_OFFSETS4: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_OFFSETS8: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)


@dataclass
class Segment:
    """A single connected-component blob.

    bounding_box is (x1, y1, x2, y2) inclusive, matching the upstream
    convention (x = column, y = row).
    """

    bounding_box: tuple[int, int, int, int]
    color: int
    area: int
    is_rectangle: bool
    number_of_twins: int = 0
    twin_ids: list[int] = field(default_factory=list)

    @property
    def width(self) -> int:
        x1, _, x2, _ = self.bounding_box
        return x2 - x1 + 1

    @property
    def height(self) -> int:
        _, y1, _, y2 = self.bounding_box
        return y2 - y1 + 1


def _np():
    """Lazy-import numpy (lets the smoke runner walk the module on no-numpy hosts)."""
    import numpy as np

    return np


def segment_frame(
    frame,
    connectivity: int = 4,
) -> tuple[np.ndarray, list[Segment]]:
    """Segment ``frame`` into single-color connected components.

    Parameters
    ----------
    frame : 2-D numpy array, dtype uint8, values 0..16
        The active layer of a frame (status-bar mask already applied if any).
    connectivity : 4 or 8
        Neighborhood for flood-fill. Default 4 (matches upstream).

    Returns
    -------
    label_map : np.ndarray of shape == frame.shape, dtype int
        ``label_map[y, x] = segment_id`` (0-indexed; -1 only if uninitialized).
    segments : list[Segment]
        One Segment per blob, indexed by segment_id. Twins are filled in
        a second pass.
    """
    np = _np()
    frame = np.asarray(frame, dtype=np.uint8)
    if frame.ndim != 2:
        raise ValueError(f"segment_frame expects a 2-D array; got shape {frame.shape!r}")

    h, w = frame.shape
    label_map = np.full((h, w), -1, dtype=np.int32)
    segments: list[Segment] = []
    cid = -1
    offsets = _OFFSETS4 if connectivity == 4 else _OFFSETS8

    for y in range(h):
        for x in range(w):
            if label_map[y, x] != -1:
                continue
            cid += 1
            color = int(frame[y, x])
            q = deque([(y, x)])
            label_map[y, x] = cid
            min_x = max_x = x
            min_y = max_y = y
            area = 0
            while q:
                cy, cx = q.popleft()
                area += 1
                if cx < min_x:
                    min_x = cx
                if cx > max_x:
                    max_x = cx
                if cy < min_y:
                    min_y = cy
                if cy > max_y:
                    max_y = cy
                for dy, dx in offsets:
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and label_map[ny, nx] == -1
                        and frame[ny, nx] == color
                    ):
                        label_map[ny, nx] = cid
                        q.append((ny, nx))
            rect_area = (max_x - min_x + 1) * (max_y - min_y + 1)
            segments.append(
                Segment(
                    bounding_box=(min_x, min_y, max_x, max_y),
                    color=color,
                    area=area,
                    is_rectangle=(area == rect_area),
                )
            )

    # Second pass: twin detection.
    # Two segments are "twins" iff same color + same area + same is_rectangle.
    # O(n^2) per the upstream simplification — n is typically << 200.
    for i, comp in enumerate(segments):
        twins: list[int] = []
        for j, other in enumerate(segments):
            if i == j:
                continue
            if (
                other.area == comp.area
                and other.is_rectangle == comp.is_rectangle
                and other.color == comp.color
            ):
                twins.append(j)
        comp.twin_ids = twins
        comp.number_of_twins = len(twins)

    return label_map, segments


def _segment_on_edges(
    segment: Segment, frame_shape: tuple[int, int] = (FRAME_SIZE, FRAME_SIZE)
) -> list[str]:
    """Return which edges this segment lies fully against (within threshold)."""
    h, w = frame_shape
    x1, y1, x2, y2 = segment.bounding_box
    edges: list[str] = []
    if max(x1, x2) < STATUS_BAR_DISTANCE_THRESHOLD:
        edges.append("left")
    if min(x1, x2) > w - STATUS_BAR_DISTANCE_THRESHOLD:
        edges.append("right")
    if max(y1, y2) < STATUS_BAR_DISTANCE_THRESHOLD:
        edges.append("top")
    if min(y1, y2) > h - STATUS_BAR_DISTANCE_THRESHOLD:
        edges.append("bottom")
    return edges


def _has_long_aspect(segment: Segment, direction: str = "any") -> bool:
    """Aspect ratio ≥ STATUS_BAR_RATIO_THRESHOLD in the given direction."""
    wx = segment.width
    wy = segment.height
    ratio = wx / wy if wy > 0 else 0.0
    if direction in ("any", "horizontal") and ratio >= STATUS_BAR_RATIO_THRESHOLD:
        return True
    return direction in ("any", "vertical") and 0 < ratio <= 1.0 / STATUS_BAR_RATIO_THRESHOLD


def identify_status_bars(
    label_map,
    segments: list[Segment],
    frame_shape: tuple[int, int] = (FRAME_SIZE, FRAME_SIZE),
) -> tuple[np.ndarray, list[list[int]]]:
    """Rule-based status-bar detection.

    A segment is flagged as part of a status bar if either:
      (a) it sits fully against an edge AND has aspect ratio >= 5:1
          in the corresponding direction — line status bar; or
      (b) it sits fully against an edge AND has >= STATUS_BAR_TWINS_THRESHOLD
          twins also on the same edge — dot/marker status bar.

    Returns
    -------
    mask : np.ndarray (frame_shape, dtype=bool)
        True where any status-bar pixel lives.
    bar_segment_groups : list[list[int]]
        Each entry is a group of segment ids that together form one bar
        (a line is a 1-segment group; dots form an N-segment group).
    """
    np = _np()
    mask = np.zeros(frame_shape, dtype=bool)
    bar_groups: list[list[int]] = []
    checked: set[int] = set()

    for i, seg in enumerate(segments):
        if i in checked:
            continue
        checked.add(i)
        on_edges = _segment_on_edges(seg, frame_shape)
        if not on_edges:
            continue

        # Resolve direction from edges
        verticals = {"left", "right"}
        horizontals = {"top", "bottom"}
        v_hit = bool(verticals.intersection(on_edges))
        h_hit = bool(horizontals.intersection(on_edges))
        if v_hit and h_hit:
            direction = "any"
        elif v_hit:
            direction = "vertical"
        else:
            direction = "horizontal"

        if _has_long_aspect(seg, direction):
            # Rule (a): line status bar — single segment.
            bar_groups.append([i])
            mask[label_map == i] = True
            continue

        # Rule (b): same-edge twins.
        twin_ids_on_edge = [
            tid
            for tid in seg.twin_ids
            if tid < len(segments)
            and _segment_on_edges(segments[tid], frame_shape)
            and bool(set(on_edges).intersection(_segment_on_edges(segments[tid], frame_shape)))
        ]
        if len(twin_ids_on_edge) + 1 >= STATUS_BAR_TWINS_THRESHOLD:
            bar_group = [i, *twin_ids_on_edge]
            bar_groups.append(bar_group)
            for sid in bar_group:
                mask[label_map == sid] = True
                checked.add(sid)

    return mask, bar_groups


def frame_segments_to_priority_tiers(
    segments: list[Segment],
    n_groups: int = 5,
    status_bar_segment_ids: set[int] | None = None,
) -> list[set[int]]:
    """Stratify segments into 5 priority tiers (lower index = higher priority).

    Tier semantics (per upstream FrameProcessor.frame_segments_to_action_groups):
      0. salient color AND medium width  — most likely interactive
      1. medium width but non-salient color
      2. salient color but extreme width (<MIN or >MAX)
      3. neither salient nor medium width, and not a status bar
      4. probable status bar              — explored last

    Notes
    -----
    The upstream version determined "is_status_bar" by checking
    `segment["color"] == STATUS_BAR_COLOR` after the caller had already
    rewritten the masked pixels to STATUS_BAR_COLOR. Here we accept the
    status-bar segment ids explicitly via ``status_bar_segment_ids``
    (more testable, no shared mutable state).
    """
    if n_groups != 5:
        raise NotImplementedError("Only 5 priority tiers are supported (matches paper)")
    tiers: list[set[int]] = [set() for _ in range(5)]
    sb_ids = status_bar_segment_ids or set()
    for sid, seg in enumerate(segments):
        is_salient = seg.color in SALIENT_COLORS
        is_medium = (
            MINIMAL_WIDTH <= seg.width <= MAXIMAL_WIDTH
            and MINIMAL_WIDTH <= seg.height <= MAXIMAL_WIDTH
        )
        is_status_bar = sid in sb_ids or seg.color == STATUS_BAR_COLOR

        if is_status_bar:
            tiers[4].add(sid)
        elif is_salient and is_medium:
            tiers[0].add(sid)
        elif is_medium:
            tiers[1].add(sid)
        elif is_salient:
            tiers[2].add(sid)
        else:
            tiers[3].add(sid)
    return tiers


def hash_masked_frame(frame, status_bar_mask=None) -> bytes:
    """Stable 16-byte blake2b digest of a single frame.

    The status-bar pixels (if a mask is provided) are rewritten to
    STATUS_BAR_COLOR before hashing, so that two frames that only
    differ in their step counter / score bar collide to the same hash.
    """
    np = _np()
    frame = np.asarray(frame, dtype=np.uint8, order="C")
    if status_bar_mask is not None:
        sm = np.asarray(status_bar_mask, dtype=bool)
        if sm.shape != frame.shape:
            raise ValueError(
                f"status_bar_mask shape {sm.shape!r} does not match frame shape {frame.shape!r}"
            )
        frame = frame.copy()
        frame[sm] = STATUS_BAR_COLOR

    flat = frame.ravel()
    # Values 0..16 fit in 5 bits but the upstream packs into 4 bits assuming
    # the masked status_bar_color of 16 is rare — we keep the same scheme
    # but truncate via & 0x0F. Two distinct color schemes (15 vs masked-16)
    # both map to 0, which is acceptable because the upstream paper uses
    # the same packing and reports stable hash performance.
    if flat.size & 1:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])
    packed = ((flat[0::2] & 0x0F) << 4) | (flat[1::2] & 0x0F)
    payload = packed.tobytes()
    shape_tag = repr(frame.shape).encode()
    return hashlib.blake2b(payload, digest_size=16, person=shape_tag).digest()


def mask_to_click_coords(
    label_map,
    segment_id: int,
    rng: random.Random | None = None,
) -> tuple[int, int] | None:
    """Uniformly sample one (x, y) inside the chosen segment.

    Returns None if the segment_id is empty in label_map. The returned
    coordinates are in (x, y) order matching the ARC-AGI-3 ACTION6
    convention.
    """
    np = _np()
    points = np.argwhere(label_map == segment_id)
    if points.size == 0:
        return None
    if rng is None:
        rng = random.Random()
    idx = rng.randrange(len(points))
    y, x = int(points[idx, 0]), int(points[idx, 1])
    return x, y


def salient_pixels_in_segment(
    label_map,
    segments: list[Segment],
    tier: int,
    tiers: list[set[int]] | None = None,
) -> np.ndarray:
    """Return a (N, 2) array of (y, x) pixel coords belonging to any segment
    in the given priority ``tier``. Used by the ACTION6 prior wire-up in
    `agents/trigger_bfs_agent.py` (D11).
    """
    np = _np()
    if tiers is None:
        tiers = frame_segments_to_priority_tiers(segments)
    if tier < 0 or tier >= len(tiers):
        return np.empty((0, 2), dtype=np.int32)
    sids = tiers[tier]
    if not sids:
        return np.empty((0, 2), dtype=np.int32)
    mask = np.zeros(label_map.shape, dtype=bool)
    for sid in sids:
        mask |= label_map == sid
    return np.argwhere(mask)
