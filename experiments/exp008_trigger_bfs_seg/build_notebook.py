"""Build trigger_bfs_seg_comp.ipynb by inlining the source-of-truth agents.

Run:
    uv run python experiments/exp008_trigger_bfs_seg/build_notebook.py

Produces:
    experiments/exp008_trigger_bfs_seg/comp_kernel/trigger_bfs_seg_comp.ipynb

The notebook follows the same 5-cell pattern as exp007 Goose CNN v2
(which scored 0.17 = "runs end-to-end on comp rerun"):
  0. pip install --no-index from competition wheels
  1. %%writefile /kaggle/working/my_agent.py with the entire inlined agent
  2. markdown
  3. comp rerun guard (curl gateway + cp ARC-AGI-3-Agents + run main.py)
  4. dummy fallback submission for dev mode
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

INLINED_AGENT = r'''# =====================================================================
# Trigger-BFS + Frame-Segmenter agent (exp008, D10+D11 of SPEC_4WEEKS).
# Inlined Kaggle copy of:
#   - agents/state_graph.py     (StateGraph, hash_frame, StateNode)
#   - agents/frame_segmenter.py (per-color CC + 5-tier saliency + status bars)
#   - agents/trigger_bfs_agent.py (TriggerBFSAgent wired to segmenter)
# Port of dolphin-in-a-coma/arc-agi-3-just-explore's FrameProcessor
# (arXiv:2512.24156 Rudakov 2026; 3rd-place ARC-AGI-3 preview challenge).
# No torch, no matplotlib. Defensive try/except around the whole
# choose_action (gotcha #17 in .factory/rules/gotchas.md).
# =====================================================================
from __future__ import annotations

import contextlib
import hashlib
import logging
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agents.agent import Agent
from arcengine import GameAction, GameState

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# State graph (mirror of agents/state_graph.py)
# --------------------------------------------------------------------


def _hash_frame(frame_layers) -> bytes:
    h = hashlib.blake2b(digest_size=8)
    if frame_layers is None:
        return h.digest()
    for layer in frame_layers:
        tobytes = getattr(layer, "tobytes", None)
        if callable(tobytes):
            h.update(tobytes())
        else:
            for row in layer:
                h.update(bytes(int(v) % 256 for v in row))
    return h.digest()


@dataclass
class StateNode:
    state_hash: bytes
    visit_count: int = 0
    untried_actions: set = field(default_factory=set)
    edges: dict = field(default_factory=dict)
    last_score: int = 0
    last_levels: int = 0
    incoming_change_score: float = 0.0


class StateGraph:
    def __init__(self):
        self.nodes: dict = {}
        self.frontier: deque = deque()
        self.action_history: list = []
        self.current_levels: int = 0

    def reset(self):
        self.nodes.clear()
        self.frontier.clear()
        self.action_history.clear()

    def maybe_reset_for_level(self, levels: int) -> bool:
        if levels >= 0 and levels != self.current_levels:
            self.reset()
            self.current_levels = levels
            return True
        return False

    def add_or_get(self, state_hash, available_actions, levels, score=0):
        node = self.nodes.get(state_hash)
        if node is None:
            untried = {int(a) for a in (available_actions or []) if int(a) != 0}
            node = StateNode(
                state_hash=state_hash,
                untried_actions=untried,
                last_levels=levels,
                last_score=score,
            )
            self.nodes[state_hash] = node
            if untried:
                self.frontier.append(state_hash)
        return node

    def observe(self, prev_hash, action_id, next_hash, change_score=0.0):
        if prev_hash is not None and prev_hash in self.nodes:
            node = self.nodes[prev_hash]
            node.edges[action_id] = next_hash
            node.untried_actions.discard(action_id)
            if not node.untried_actions:
                with contextlib.suppress(ValueError):
                    self.frontier.remove(prev_hash)
        if next_hash in self.nodes:
            self.nodes[next_hash].visit_count += 1
            self.nodes[next_hash].incoming_change_score = change_score

    def record_action(self, state_hash, action_id, data):
        self.action_history.append((state_hash, action_id, dict(data)))


# --------------------------------------------------------------------
# Frame segmenter (mirror of agents/frame_segmenter.py)
# --------------------------------------------------------------------

SALIENT_COLORS = frozenset({6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
NON_SALIENT_COLORS = frozenset({0, 1, 2, 3, 4, 5})
STATUS_BAR_COLOR = 16
FRAME_SIZE = 64
STATUS_BAR_DISTANCE_THRESHOLD = 3
STATUS_BAR_RATIO_THRESHOLD = 5.0
STATUS_BAR_TWINS_THRESHOLD = 3
MINIMAL_WIDTH = 2
MAXIMAL_WIDTH = 32

_OFFSETS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class Segment:
    bounding_box: tuple
    color: int
    area: int
    is_rectangle: bool
    number_of_twins: int = 0
    twin_ids: list = field(default_factory=list)

    @property
    def width(self) -> int:
        x1, _, x2, _ = self.bounding_box
        return x2 - x1 + 1

    @property
    def height(self) -> int:
        _, y1, _, y2 = self.bounding_box
        return y2 - y1 + 1


def segment_frame(frame):
    frame = np.asarray(frame, dtype=np.uint8)
    h, w = frame.shape
    label_map = np.full((h, w), -1, dtype=np.int32)
    segments: list = []
    cid = -1
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
                if cx < min_x: min_x = cx
                if cx > max_x: max_x = cx
                if cy < min_y: min_y = cy
                if cy > max_y: max_y = cy
                for dy, dx in _OFFSETS4:
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < h and 0 <= nx < w
                            and label_map[ny, nx] == -1
                            and frame[ny, nx] == color):
                        label_map[ny, nx] = cid
                        q.append((ny, nx))
            rect_area = (max_x - min_x + 1) * (max_y - min_y + 1)
            segments.append(Segment(
                bounding_box=(min_x, min_y, max_x, max_y),
                color=color, area=area,
                is_rectangle=(area == rect_area),
            ))
    for i, comp in enumerate(segments):
        twins = []
        for j, other in enumerate(segments):
            if i == j: continue
            if (other.area == comp.area
                    and other.is_rectangle == comp.is_rectangle
                    and other.color == comp.color):
                twins.append(j)
        comp.twin_ids = twins
        comp.number_of_twins = len(twins)
    return label_map, segments


def _segment_on_edges(segment, frame_shape=(FRAME_SIZE, FRAME_SIZE)):
    h, w = frame_shape
    x1, y1, x2, y2 = segment.bounding_box
    edges = []
    if max(x1, x2) < STATUS_BAR_DISTANCE_THRESHOLD: edges.append("left")
    if min(x1, x2) > w - STATUS_BAR_DISTANCE_THRESHOLD: edges.append("right")
    if max(y1, y2) < STATUS_BAR_DISTANCE_THRESHOLD: edges.append("top")
    if min(y1, y2) > h - STATUS_BAR_DISTANCE_THRESHOLD: edges.append("bottom")
    return edges


def _has_long_aspect(segment, direction="any"):
    wx = segment.width
    wy = segment.height
    ratio = wx / wy if wy > 0 else 0.0
    if direction in ("any", "horizontal") and ratio >= STATUS_BAR_RATIO_THRESHOLD:
        return True
    return direction in ("any", "vertical") and 0 < ratio <= 1.0 / STATUS_BAR_RATIO_THRESHOLD


def identify_status_bars(label_map, segments, frame_shape=(FRAME_SIZE, FRAME_SIZE)):
    mask = np.zeros(frame_shape, dtype=bool)
    bar_groups = []
    checked = set()
    for i, seg in enumerate(segments):
        if i in checked: continue
        checked.add(i)
        on_edges = _segment_on_edges(seg, frame_shape)
        if not on_edges: continue
        v_hit = bool({"left", "right"}.intersection(on_edges))
        h_hit = bool({"top", "bottom"}.intersection(on_edges))
        direction = "any" if (v_hit and h_hit) else ("vertical" if v_hit else "horizontal")
        if _has_long_aspect(seg, direction):
            bar_groups.append([i])
            mask[label_map == i] = True
            continue
        twin_ids_on_edge = [
            tid for tid in seg.twin_ids
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


def frame_segments_to_priority_tiers(segments, n_groups=5, status_bar_segment_ids=None):
    tiers = [set() for _ in range(5)]
    sb_ids = status_bar_segment_ids or set()
    for sid, seg in enumerate(segments):
        is_salient = seg.color in SALIENT_COLORS
        is_medium = MINIMAL_WIDTH <= seg.width <= MAXIMAL_WIDTH and MINIMAL_WIDTH <= seg.height <= MAXIMAL_WIDTH
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


def mask_to_click_coords(label_map, segment_id, rng=None):
    points = np.argwhere(label_map == segment_id)
    if points.size == 0:
        return None
    if rng is None:
        rng = random.Random()
    idx = rng.randrange(len(points))
    y, x = int(points[idx, 0]), int(points[idx, 1])
    return x, y


# --------------------------------------------------------------------
# Trigger-BFS agent (mirror of agents/trigger_bfs_agent.py)
# --------------------------------------------------------------------


def _to_ndarray(layer):
    try:
        if isinstance(layer, np.ndarray):
            return layer
        return np.asarray(layer, dtype=np.uint8)
    except Exception:
        return None


def _trigger_score(prev_layers, next_layers, prev_levels, next_levels):
    if prev_layers is None or next_layers is None:
        return 0.0
    p = _to_ndarray(prev_layers[-1])
    n = _to_ndarray(next_layers[-1])
    if p is None or n is None or p.shape != n.shape:
        return float(5 * (next_levels - prev_levels))
    delta_pixels = float((p != n).sum())
    new_colors = float(len({int(v) for v in n.flat} - {int(v) for v in p.flat}))
    delta_levels = float(next_levels - prev_levels)
    return delta_pixels + 5.0 * delta_levels + 2.0 * new_colors


def _sample_click_xy(layers, rng):
    """Pick (x, y) for ACTION6 using the frame-segmenter saliency tiers."""
    try:
        if not layers:
            raise ValueError("no layers")
        grid = _to_ndarray(layers[-1])
        if grid is None or grid.ndim != 2:
            raise ValueError("bad grid")
        label_map, segments = segment_frame(grid)
        _sb_mask, sb_groups = identify_status_bars(label_map, segments)
        sb_ids = set()
        for g in sb_groups:
            sb_ids.update(g)
        tiers = frame_segments_to_priority_tiers(segments, status_bar_segment_ids=sb_ids)
        frame_pixels = grid.shape[0] * grid.shape[1]
        half = frame_pixels // 2
        for tier_idx in range(4):
            tier_sids = [sid for sid in tiers[tier_idx] if segments[sid].area <= half]
            if not tier_sids:
                continue
            chosen_sid = rng.choice(tier_sids)
            coord = mask_to_click_coords(label_map, chosen_sid, rng=rng)
            if coord is not None:
                return {"x": coord[0], "y": coord[1]}
        bg = int(np.bincount(grid.flatten(), minlength=16).argmax())
        ys, xs = np.where(grid != bg)
        if len(xs) > 0:
            idx = rng.randrange(len(xs))
            return {"x": int(xs[idx]), "y": int(ys[idx])}
    except Exception:
        pass
    return {"x": rng.randint(0, 63), "y": rng.randint(0, 63)}


class MyAgent(Agent):
    """Trigger-BFS + frame-segmenter ACTION6 prior."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rng = random.Random(0)
        self.graph = StateGraph()
        self._prev_hash = None
        self._prev_action = None
        self._prev_data: dict = {}
        self._prev_layers = None
        self._prev_levels = 0

    def is_done(self, frames, latest_frame) -> bool:
        return latest_frame.state == GameState.WIN

    def choose_action(self, frames, latest_frame):
        # v2 pattern: wrap the whole body in try/except (gotcha #17).
        try:
            return self._choose_action_inner(frames, latest_frame)
        except Exception as exc:
            logger.warning("choose_action: graceful fallback after %r", exc)
            avail_raw = list(getattr(latest_frame, "available_actions", []) or [1, 2, 3, 4, 5])
            non_reset_fb = [int(a) for a in avail_raw if int(a) != 0 and int(a) != 6]
            if not non_reset_fb:
                non_reset_fb = [1, 2, 3, 4, 5]
            try:
                return GameAction.from_id(self._rng.choice(non_reset_fb))
            except Exception:
                return GameAction.RESET

    def _choose_action_inner(self, frames, latest_frame):
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._prev_hash = None
            self._prev_action = None
            self._prev_layers = None
            return GameAction.RESET
        cur_layers = list(getattr(latest_frame, "frame", []) or [])
        cur_hash = _hash_frame(cur_layers) if cur_layers else b"\x00" * 8
        cur_levels = int(getattr(latest_frame, "levels_completed", 0))
        self.graph.maybe_reset_for_level(cur_levels)
        change_score = _trigger_score(self._prev_layers, cur_layers, self._prev_levels, cur_levels)
        avail = list(getattr(latest_frame, "available_actions", []) or [1, 2, 3, 4, 5, 6, 7])
        node = self.graph.add_or_get(cur_hash, available_actions=avail, levels=cur_levels)
        if self._prev_hash is not None and self._prev_action is not None:
            self.graph.observe(self._prev_hash, self._prev_action, cur_hash, change_score)
        non_reset_avail = [int(a) for a in avail if int(a) != 0]
        if not non_reset_avail:
            non_reset_avail = [1, 2, 3, 4, 5, 6, 7]
        untried_avail = [a for a in node.untried_actions if a in non_reset_avail]
        if untried_avail:
            chosen = self._rng.choice(untried_avail)
        else:
            scored = []
            for a, succ_h in node.edges.items():
                if a not in non_reset_avail:
                    continue
                succ = self.graph.nodes.get(succ_h)
                if succ is None:
                    continue
                s = succ.incoming_change_score + 0.05 * len(succ.untried_actions)
                scored.append((s, a))
            if scored and max(s for s, _ in scored) > 0.0:
                top = max(scored)[0]
                top_actions = [a for s, a in scored if s == top]
                chosen = self._rng.choice(top_actions)
            else:
                chosen = self._rng.choice(non_reset_avail)
        action = GameAction.from_id(chosen)
        data = {}
        if action.is_complex():
            data = _sample_click_xy(cur_layers, self._rng)
            action.set_data(data)
        self._prev_hash = cur_hash
        self._prev_action = chosen
        self._prev_data = data
        self._prev_layers = cur_layers
        self._prev_levels = cur_levels
        self.graph.record_action(cur_hash, chosen, data)
        return action
'''


def make_cell(cell_type, source, lang_version="3.12.0"):
    if isinstance(source, str):
        lines = source.splitlines(keepends=True)
    else:
        lines = source
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def main():
    cells = [
        make_cell(
            "code",
            [
                "!pip install --no-index --find-links \\\n",
                "    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n",
                "    arc-agi python-dotenv",
            ],
        ),
        make_cell(
            "code",
            ["%%writefile /kaggle/working/my_agent.py\n"] + INLINED_AGENT.splitlines(keepends=True),
        ),
        make_cell(
            "markdown", ["this only runs if you submit to the competition, not when you do tests"]
        ),
        make_cell(
            "code",
            [
                "import os\n",
                "if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n",
                "    !curl --fail --retry 999 --retry-all-errors --retry-delay 5 --retry-max-time 600 http://gateway:8001/api/games\n",
                "    !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents /kaggle/working/ARC-AGI-3-Agents\n",
                "    !cp /kaggle/working/my_agent.py /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py\n",
                "    with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py','w') as f:\n",
                '        f.write("""from typing import Type\n',
                "from dotenv import load_dotenv\n",
                "from .agent import Agent, Playback\n",
                "from .swarm import Swarm\n",
                "from .templates.random_agent import Random\n",
                "from .templates.my_agent import MyAgent\n",
                "load_dotenv()\n",
                'AVAILABLE_AGENTS: dict[str, Type[Agent]] = {"random": Random, "myagent": MyAgent}\n',
                '""")\n',
                "    with open('/kaggle/working/ARC-AGI-3-Agents/.env','w') as f:\n",
                '        f.write("""SCHEME=http\n',
                "HOST=gateway\n",
                "PORT=8001\n",
                "ARC_API_KEY=test-key-123\n",
                "ARC_BASE_URL=http://gateway:8001/\n",
                "OPERATION_MODE=online\n",
                "RECORDINGS_DIR=/kaggle/working/server_recording\n",
                '""")\n',
                "    !cd /kaggle/working/ARC-AGI-3-Agents && MPLBACKEND=agg python main.py --agent myagent",
            ],
        ),
        make_cell(
            "code",
            [
                "import os\n",
                "if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n",
                "    import pandas as pd\n",
                "    submission = pd.DataFrame(data=[['1_0','1',True,1]],columns=['row_id','game_id','end_of_game','score'])\n",
                "    submission.to_parquet('/kaggle/working/submission.parquet',index=False)",
            ],
        ),
        make_cell("markdown", ["This is a dummy submission fallback, important to keep"]),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    out_path = REPO / "experiments/exp008_trigger_bfs_seg/comp_kernel/trigger_bfs_seg_comp.ipynb"
    with out_path.open("w") as f:
        json.dump(notebook, f, indent=1)
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
