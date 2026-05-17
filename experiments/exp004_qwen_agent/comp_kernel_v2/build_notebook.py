"""Build qwen_phase1_comp.ipynb for exp004 Phase 1 (D17).

Run:
    uv run python experiments/exp004_qwen_agent/comp_kernel_v2/build_notebook.py

Produces:
    experiments/exp004_qwen_agent/comp_kernel_v2/qwen_phase1_comp.ipynb

Cell layout (7 cells), mirroring exp008's pattern + rtx6000_probe overlay:
  0. pip install --no-index from competition wheels (arc-agi + python-dotenv)
  1. Offline overlay install for Pillow 12.2.0 + transformers 5.7.0 (gotchas
     #13, #14): install via --target /tmp/_pillow_pkg + /tmp/_transformers_pkg,
     prepend to sys.path, purge stale modules.
  2. %%writefile /kaggle/working/my_agent.py — inlined Phase-1 agent:
     state_graph + frame_segmenter + trigger_bfs (fallback) + Phase-1
     QwenAgent + MyAgent shim.
  3. markdown
  4. competition-rerun guard: copy ARC-AGI-3-Agents harness, set Qwen env
     vars, run main.py --agent myagent
  5. dummy submission fallback for dev mode
  6. markdown
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent

# ---------------------------------------------------------------------------
# Inlined agent source (state_graph + frame_segmenter + trigger_bfs + Qwen)
# ---------------------------------------------------------------------------

INLINED_HEADER = r"""# =====================================================================
# Phase-1 "Direct Policy++" Qwen agent (exp004 D17, 2026-05-15).
# Inlined for Kaggle (notebook-visible code per gotcha #21):
#   - agents/state_graph.py     (StateGraph, hash_frame, StateNode)
#   - agents/frame_segmenter.py (per-color CC + 5-tier saliency + status bars)
#   - agents/trigger_bfs_agent.py (TriggerBFSAgent fallback wrapper)
#   - agents/qwen_agent.py      (Phase-1 patched: state-graph + outcome
#                                memory + ACTION6 candidates + JSON guard)
#   - MyAgent shim delegating to QwenAgent
# Defensive try/except around the whole choose_action (gotcha #17).
# Mandatory enable_thinking=False on apply_chat_template (gotcha #22).
# =====================================================================
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from agents.agent import Agent
except ImportError:
    # Dev kernel: ARC-AGI-3-Agents harness not mounted yet; provide a minimal stub.
    class Agent:
        def __init__(self, *args, **kwargs):
            pass

from arcengine import GameAction, GameState

logger = logging.getLogger(__name__)
"""

INLINED_STATE_GRAPH = r"""
# --------------------------------------------------------------------
# State graph
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
"""

INLINED_SEGMENTER = r"""
# --------------------------------------------------------------------
# Frame segmenter
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


# D17 ACTION6-spam fix: hash PRIMARY visual layer with status-bar pixels
# masked (drops drift in overlay/score layers + best-effort bar masking).
# Used by QwenAgent only; TriggerBFS fallback keeps unmasked _hash_frame.
def _hash_frame_masked(frame_layers) -> bytes:
    h = hashlib.blake2b(digest_size=8)
    if not frame_layers:
        return h.digest()
    layer = frame_layers[0]
    try:
        grid = layer if isinstance(layer, np.ndarray) else np.asarray(layer, dtype=np.uint8)
        if grid.ndim != 2:
            h.update(grid.tobytes())
            return h.digest()
        label_map, segments = segment_frame(grid)
        sb_mask, _ = identify_status_bars(label_map, segments)
        if sb_mask is not None and sb_mask.any():
            masked = grid.copy()
            masked[sb_mask] = STATUS_BAR_COLOR
            h.update(masked.tobytes())
        else:
            h.update(grid.tobytes())
        return h.digest()
    except Exception:
        h = hashlib.blake2b(digest_size=8)
        tobytes = getattr(layer, "tobytes", None)
        if callable(tobytes):
            h.update(tobytes())
        else:
            for row in layer:
                h.update(bytes(int(v) % 256 for v in row))
        return h.digest()
"""

INLINED_TRIGGER_BFS = r'''
# --------------------------------------------------------------------
# Trigger-BFS fallback agent (used by QwenAgent on catastrophic failure)
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
    """Pick (x, y) for ACTION6 using saliency tiers (matches D11 trigger-bfs)."""
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


class TriggerBFSAgent:
    """Fallback agent used by QwenAgent on catastrophic failure."""

    name = "trigger-bfs-fallback"

    def __init__(self, seed=0, **_):
        self._rng = random.Random(seed)
        self.graph = StateGraph()
        self._prev_hash = None
        self._prev_action = None
        self._prev_data: dict = {}
        self._prev_layers = None
        self._prev_levels = 0

    def choose_action(self, frame):
        if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._prev_hash = None
            self._prev_action = None
            self._prev_layers = None
            return GameAction.RESET
        cur_layers = list(getattr(frame, "frame", []) or [])
        cur_hash = _hash_frame(cur_layers) if cur_layers else b"\x00" * 8
        cur_levels = int(getattr(frame, "levels_completed", 0))
        self.graph.maybe_reset_for_level(cur_levels)
        change_score = _trigger_score(self._prev_layers, cur_layers, self._prev_levels, cur_levels)
        avail = list(getattr(frame, "available_actions", []) or [1, 2, 3, 4, 5, 6, 7])
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

    def is_done(self, frame):
        return frame.state == GameState.WIN
'''

INLINED_QWEN = r'''
# --------------------------------------------------------------------
# Phase-1 Qwen agent
# --------------------------------------------------------------------

_DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
_DEFAULT_DTYPE = "bf16"
_DEFAULT_DEVICE_MAP = "auto"
_DEFAULT_MAX_NEW = 24
_DEFAULT_HIST_LEN = 6
_DEFAULT_CANDIDATES = 8
_FRAME_UPSCALE = 8


def _find_qwen_model_path():
    """Resolve mounted Kaggle dataset path for Qwen weights (gotcha #11).

    Tries (in order):
      1. QWEN_MODEL_PATH env var if it points to a dir with config.json
      2. /kaggle/input/qwen3-6-35b-a3b-bf16 (flat layout)
      3. /kaggle/input/datasets/cataluna84/qwen3-6-35b-a3b-bf16 (nested layout)
      4. Recursive search under /kaggle/input for any config.json mentioning 'qwen'
    Returns None on failure; caller falls back to _DEFAULT_MODEL (HF hub id).
    """
    from pathlib import Path
    env_p = os.environ.get('QWEN_MODEL_PATH')
    if env_p and Path(env_p).exists() and any(Path(env_p).glob('config.json')):
        return env_p
    for root in [
        Path('/kaggle/input/qwen3-6-35b-a3b-bf16'),
        Path('/kaggle/input/datasets/cataluna84/qwen3-6-35b-a3b-bf16'),
        Path('/kaggle/input'),
    ]:
        if not root.exists():
            continue
        if any(root.glob('config.json')):
            return str(root)
        for cfg in root.rglob('config.json'):
            text = cfg.read_text(errors='ignore')[:4000].lower()
            path_text = str(cfg.parent).lower()
            if 'qwen' in text or 'qwen' in path_text:
                return str(cfg.parent)
    return None

_PALETTE = [
    (0, 0, 0), (30, 147, 255), (249, 60, 49), (79, 204, 48),
    (255, 220, 0), (153, 153, 153), (229, 58, 163), (255, 133, 27),
    (135, 216, 241), (146, 18, 49), (96, 16, 96), (16, 96, 16),
    (16, 16, 96), (96, 96, 16), (96, 16, 96), (240, 240, 240),
]


def _frame_to_pil_image(grid):
    try:
        from PIL import Image
    except ImportError:
        return None
    tolist = getattr(grid, "tolist", None)
    iterable = tolist() if callable(tolist) else grid
    rows = list(iterable)
    if not rows:
        return None
    H = len(rows)
    W = len(rows[0]) if rows else 0
    img = Image.new("RGB", (W, H), (0, 0, 0))
    px = img.load()
    for y in range(H):
        row = rows[y]
        for x in range(W):
            v = int(row[x]) % len(_PALETTE)
            px[x, y] = _PALETTE[v]
    return img.resize((W * _FRAME_UPSCALE, H * _FRAME_UPSCALE), resample=Image.Resampling.NEAREST)


_ACTION_RE = re.compile(
    r"\bACTION\s*[_:]?\s*(\d)\b|\bRESET\b|\bACTION_(\d)\b",
    flags=re.IGNORECASE,
)
_COORD_RE = re.compile(
    r"\(?\s*x\s*[:=]?\s*(\d+)\s*[, ]\s*y\s*[:=]?\s*(\d+)\s*\)?", flags=re.IGNORECASE
)
_COORD_RE_BARE = re.compile(r"\((\d{1,2})\s*,\s*(\d{1,2})\)")
_JSON_OBJ_RE = re.compile(
    r'\{[^{}]*"action"\s*:\s*"[^"]+"[^{}]*\}',
    flags=re.IGNORECASE,
)
_CAND_LABEL_RE = re.compile(r"C(\d+)", flags=re.IGNORECASE)


def _parse_json_first(text):
    if not text:
        return None
    match = _JSON_OBJ_RE.search(text)
    if match is None:
        return None
    try:
        result = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    return result if isinstance(result, dict) else None


def build_prompt(frame, history, *, visit_count=0, untried=None, action6_candidates=None, tried_clicks=None):
    avail = [int(a) for a in (getattr(frame, "available_actions", None) or [])]
    avail_str = ", ".join(f"ACTION{a}" for a in avail) if avail else "ACTION1..ACTION7"
    state_str = getattr(getattr(frame, "state", None), "name", "?")
    levels = getattr(frame, "levels_completed", 0)
    win_levels = getattr(frame, "win_levels", 1)
    layers = getattr(frame, "frame", None) or []
    grid = layers[0] if layers else None
    image = _frame_to_pil_image(grid)

    hist_lines = []
    for entry in history or []:
        if isinstance(entry, (tuple, list)) and len(entry) >= 3:
            name = str(entry[0])
            change_score = float(entry[1] or 0.0)
            level_delta = int(entry[2] or 0)
            marker = "changed" if change_score > 0 else "no-change"
            hist_lines.append(f"  - {name}: {marker} delta={change_score:.0f} levelΔ={level_delta:+d}")
        elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
            name = str(entry[0])
            changed = bool(entry[1])
            marker = "changed" if changed else "no-change"
            hist_lines.append(f"  - {name}: {marker}")
    hist_str = "\n".join(hist_lines) if hist_lines else "  (no actions yet)"

    system = (
        "You are an ARC-AGI-3 game agent. Do not reason step-by-step. Output "
        'ONE action only as STRICT JSON on one line: '
        '{"action":"ACTIONn","x":int|null,"y":int|null,"why":"<=12 chars"}. '
        "Pick ACTIONn from Available. For ACTION6 you MUST set x,y in [0,63] -- "
        "PREFER one of the listed Click candidates. If 'Already tried at this "
        "state' is non-empty, PICK A DIFFERENT click candidate (different (x,y)) "
        "because the listed coords have already produced no change here. Avoid "
        "actions in the recent history that caused no-change at this state."
    )
    user_content = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    parts = [
        f"state={state_str} level={levels}/{win_levels}",
        f"visit_count={visit_count}",
        f"Available: {avail_str}",
    ]
    if untried:
        parts.append(f"Untried at this state: " + ", ".join(f"ACTION{a}" for a in untried))
    if action6_candidates:
        cand_lines = ["Click candidates:"]
        for c in action6_candidates:
            cand_lines.append(
                f"  - {c.get('label', '?')} ACTION6 "
                f"({int(c.get('x', 0))},{int(c.get('y', 0))}) "
                f"tier{int(c.get('tier', 4))}"
            )
        parts.append("\n".join(cand_lines))
    if tried_clicks:
        tried_str = ", ".join(f"({int(x)},{int(y)})" for x, y in tried_clicks)
        parts.append(
            f"Already tried at this state (no-change): {tried_str}. Pick a DIFFERENT click candidate."
        )
    parts.append(f"History (oldest->newest):\n{hist_str}")
    parts.append("Output ONE JSON object only.")
    user_content.append({"type": "text", "text": "\n".join(parts)})
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ], image


class QwenAgent:
    """Phase-1 Qwen agent with state-graph + segmenter + guards + TriggerBFS fallback."""

    name = "qwen3_6_35b_a3b_phase1"

    def __init__(self, seed=0, **kwargs):
        # Prefer mounted Kaggle dataset; fall back to HF hub id (gotcha #11).
        # _find_qwen_model_path checks QWEN_MODEL_PATH env first, then searches.
        self.model_path = _find_qwen_model_path() or _DEFAULT_MODEL
        self.dtype_str = (os.environ.get("QWEN_DTYPE") or _DEFAULT_DTYPE).lower()
        self.device_map = os.environ.get("QWEN_DEVICE_MAP") or _DEFAULT_DEVICE_MAP
        self.max_new_tokens = int(os.environ.get("QWEN_MAX_NEW_TOKENS") or _DEFAULT_MAX_NEW)
        self.history_len = int(os.environ.get("QWEN_HISTORY_LEN") or _DEFAULT_HIST_LEN)
        self._debug = os.environ.get("QWEN_DEBUG_PROMPTS") == "1"

        self._outcome_history = deque(maxlen=self.history_len)
        self._state_graph = StateGraph()
        self._prev_grid_hash = 0
        self._prev_action_name = None
        self._prev_layers = None
        self._prev_hash = None
        self._prev_action_id = None
        self._prev_levels = 0
        # D17 Path B: per-state click-coord history for ACTION6-only games.
        self._tried_clicks_per_state = {}
        self._prev_click = None
        self._fallback = TriggerBFSAgent(seed=int(seed) if seed is not None else 0)
        self._fallback_count = 0
        self._parse_failure_count = 0

        self._processor = None
        self._model = None
        self._torch = None
        self._loaded = False

    def reset_counters(self):
        self._fallback_count = 0
        self._parse_failure_count = 0

    def _ensure_model_loaded(self):
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        dtype_map = {
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16, "float16": torch.float16,
            "fp32": torch.float32, "float32": torch.float32,
        }
        dtype = dtype_map.get(self.dtype_str, torch.bfloat16)
        t0 = time.time()
        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, dtype=dtype, device_map=self.device_map,
            trust_remote_code=True, low_cpu_mem_usage=True,
        ).eval()
        self._torch = torch
        self._loaded = True
        if self._debug:
            logger.info("QwenAgent loaded %s in %.1fs", self.model_path, time.time() - t0)

    def _log(self, msg):
        try:
            with open("/tmp/qwen_trace.log", "a") as fh:
                fh.write(msg + "\n")
        except OSError:
            pass

    def _grid_hash(self, grid):
        if grid is None:
            return 0
        tobytes = getattr(grid, "tobytes", None)
        if callable(tobytes):
            return hash(tobytes())
        return hash(tuple(tuple(int(v) for v in row) for row in grid))

    def _segment_action6_candidates(self, frame_layers):
        try:
            if not frame_layers:
                return []
            layer = frame_layers[-1]
            if isinstance(layer, np.ndarray):
                grid = layer
            else:
                grid = np.asarray(layer, dtype=np.uint8)
            if grid.ndim != 2:
                return []
            label_map, segments = segment_frame(grid)
            _sb_mask, sb_groups = identify_status_bars(label_map, segments)
            sb_ids = set()
            for g in sb_groups:
                sb_ids.update(g)
            tiers = frame_segments_to_priority_tiers(segments, status_bar_segment_ids=sb_ids)
            frame_pixels = grid.shape[0] * grid.shape[1]
            half = frame_pixels // 2
            rng = random.Random(0)
            results = []
            for tier_idx in range(4):
                tier_sids = [sid for sid in tiers[tier_idx] if segments[sid].area <= half]
                tier_sids.sort(key=lambda sid: -segments[sid].area)
                for sid in tier_sids:
                    coord = mask_to_click_coords(label_map, sid, rng=rng)
                    if coord is None:
                        continue
                    results.append({"label": f"C{len(results)}", "x": int(coord[0]), "y": int(coord[1]), "tier": tier_idx})
                    if len(results) >= _DEFAULT_CANDIDATES:
                        return results
            return results
        except Exception:
            return []

    def _apply_guards(self, parsed, frame, candidates, node=None):
        avail = [int(a) for a in (getattr(frame, "available_actions", None) or [])]
        avail_set = {int(a) for a in avail}
        raw = parsed.get("action") if isinstance(parsed, dict) else ""
        raw_text = str(raw) if raw is not None else ""
        chosen = None
        m = _ACTION_RE.search(raw_text)
        if m:
            if m.group(0).upper().startswith("RESET"):
                chosen = 0
            else:
                d = m.group(1) or m.group(2)
                if d is not None:
                    try:
                        chosen = int(d)
                    except ValueError:
                        chosen = None
        if chosen is None or chosen not in avail_set:
            non_reset = sorted(a for a in avail_set if a != 0)
            chosen = non_reset[0] if non_reset else 1
        if (
            node is not None
            and chosen != 0
            and chosen in getattr(node, "edges", {})
            and chosen not in getattr(node, "untried_actions", set())
        ):
            succ_hash = node.edges.get(chosen)
            succ = self._state_graph.nodes.get(succ_hash) if succ_hash is not None else None
            if (
                succ is not None
                and getattr(succ, "visit_count", 0) >= 1
                and getattr(succ, "incoming_change_score", 0.0) == 0.0
            ):
                untried_in_avail = sorted(a for a in node.untried_actions if a in avail_set)
                if untried_in_avail:
                    chosen = untried_in_avail[0]
        action = GameAction.from_id(chosen)
        if action.is_complex():
            x_val = y_val = None
            px = parsed.get("x") if isinstance(parsed, dict) else None
            py = parsed.get("y") if isinstance(parsed, dict) else None
            if isinstance(px, int) and isinstance(py, int) and 0 <= px < 64 and 0 <= py < 64:
                x_val, y_val = px, py
            else:
                why = parsed.get("why") if isinstance(parsed, dict) else None
                if isinstance(why, str) and candidates:
                    cm = _CAND_LABEL_RE.search(why)
                    if cm:
                        try:
                            idx = int(cm.group(1))
                            if 0 <= idx < len(candidates):
                                x_val = int(candidates[idx].get("x", 32))
                                y_val = int(candidates[idx].get("y", 32))
                        except (ValueError, KeyError, TypeError):
                            pass
                if (x_val is None or y_val is None) and candidates:
                    x_val = int(candidates[0].get("x", 32))
                    y_val = int(candidates[0].get("y", 32))
                if x_val is None or y_val is None:
                    x_val, y_val = 32, 32
            action.set_data({"x": int(x_val) % 64, "y": int(y_val) % 64})
        return action

    def choose_action(self, frame):
        try:
            return self._choose_action_inner(frame)
        except Exception as exc:
            self._fallback_count += 1
            self._log(f"[QwenAgent] fallback after {type(exc).__name__}: {exc}")
            try:
                return self._fallback.choose_action(frame)
            except Exception as exc2:
                self._log(f"[QwenAgent] fallback also raised {exc2!r}")
                return GameAction.RESET

    def _choose_action_inner(self, frame):
        state = getattr(frame, "state", None)
        if state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET
        cur_layers = list(getattr(frame, "frame", []) or [])
        # Masked hash collapses status-bar drift to a stable digest (D17 fix).
        cur_hash = _hash_frame_masked(cur_layers) if cur_layers else b"\x00" * 8
        cur_levels = int(getattr(frame, "levels_completed", 0))
        if cur_levels >= 0 and cur_levels != self._prev_levels:
            self._state_graph.maybe_reset_for_level(cur_levels)
            self._outcome_history.clear()
            self._tried_clicks_per_state.clear()  # D17 Path B
        avail = list(getattr(frame, "available_actions", []) or [1, 2, 3, 4, 5, 6, 7])
        node = self._state_graph.add_or_get(cur_hash, available_actions=avail, levels=cur_levels)
        if self._prev_hash is not None and self._prev_action_id is not None:
            change_score = _trigger_score(self._prev_layers, cur_layers, self._prev_levels, cur_levels)
            self._state_graph.observe(self._prev_hash, self._prev_action_id, cur_hash, change_score)
            level_delta = cur_levels - self._prev_levels
            self._outcome_history.append((
                self._prev_action_name or f"ACTION{self._prev_action_id}",
                float(change_score), int(level_delta),
            ))
            # D17 Path B: record no-change ACTION6 clicks for prompt diversification.
            if (
                self._prev_action_id == 6
                and change_score == 0.0
                and self._prev_hash == cur_hash
                and self._prev_click is not None
            ):
                bucket = self._tried_clicks_per_state.setdefault(cur_hash, [])
                if self._prev_click not in bucket:
                    bucket.append(self._prev_click)
        avail_set = {int(a) for a in avail}
        candidates = self._segment_action6_candidates(cur_layers) if 6 in avail_set else []
        untried = sorted(a for a in node.untried_actions if a in avail_set and a != 0)
        tried_clicks_here = list(self._tried_clicks_per_state.get(cur_hash, []))
        messages, image = build_prompt(
            frame, list(self._outcome_history),
            visit_count=node.visit_count, untried=untried, action6_candidates=candidates,
            tried_clicks=tried_clicks_here,
        )
        self._ensure_model_loaded()
        torch = self._torch
        try:
            text_in = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            text_in = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        proc_kwargs = {"text": text_in, "return_tensors": "pt"}
        if image is not None:
            proc_kwargs["images"] = [image]
        inputs = self._processor(**proc_kwargs).to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                temperature=0.0, use_cache=True,
            )
        gen_tokens = out[0, inputs["input_ids"].shape[1] :]
        reply = self._processor.batch_decode([gen_tokens], skip_special_tokens=True)[0]
        if self._debug:
            self._log(f"---\nREPLY:\n{reply}\n")
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
        # D17 Path B: capture click coords (arcengine stores on action_data).
        if int(action.value) == 6:
            ad = getattr(action, "action_data", None)
            if ad is not None and hasattr(ad, "x") and hasattr(ad, "y"):
                try:
                    self._prev_click = (int(ad.x), int(ad.y))
                except (AttributeError, TypeError, ValueError):
                    self._prev_click = None
            elif isinstance(data, dict) and "x" in data and "y" in data:
                try:
                    self._prev_click = (int(data["x"]), int(data["y"]))
                except (ValueError, TypeError):
                    self._prev_click = None
            else:
                self._prev_click = None
        else:
            self._prev_click = None
        self._state_graph.record_action(cur_hash, int(action.value), data)
        return action

    def is_done(self, frame):
        return getattr(frame, "state", None) == GameState.WIN
'''

INLINED_MYAGENT = r'''
# --------------------------------------------------------------------
# MyAgent shim — delegates the official harness to QwenAgent
# --------------------------------------------------------------------


class MyAgent(Agent):
    """Phase-1 Qwen Direct Policy++ agent, wrapped for the ARC-AGI-3-Agents harness."""

    MAX_ACTIONS = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inner = QwenAgent(seed=0)

    def is_done(self, frames, latest_frame):
        return latest_frame.state == GameState.WIN

    def choose_action(self, frames, latest_frame):
        return self._inner.choose_action(latest_frame)
'''


def make_cell(cell_type, source, **extra):
    if isinstance(source, str):
        lines = source.splitlines(keepends=True)
    else:
        lines = source
    cell = {"cell_type": cell_type, "metadata": {}, "source": lines}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    cell.update(extra)
    return cell


# ---------------------------------------------------------------------------
# Cell bodies (re-used by both comp_kernel_v2 and dev_kernel_v3)
# ---------------------------------------------------------------------------


CELL0_PIP = [
    "!pip install --no-index --find-links \\\n",
    "    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n",
    "    arc-agi python-dotenv",
]

CELL1_OVERLAY = """\
import os, sys, subprocess
from pathlib import Path


def find_dir_with(pattern, roots):
    for root in roots:
        if not root.exists():
            continue
        if any(root.glob(pattern)):
            return root
        for hit in root.rglob(pattern):
            return hit.parent
    return None


def run(cmd, timeout=300):
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    print(proc.stdout[-2000:])
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-2000:])
    return proc.returncode


# Pillow 12.2.0 from competition wheels (gotcha #13)
wheel_roots = [
    Path("/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"),
    Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"),
    Path("/kaggle/input"),
]
comp_wheels = find_dir_with("pillow-*.whl", wheel_roots)
if comp_wheels is not None:
    pillow_target = Path("/tmp/_pillow_pkg")
    pillow_target.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "--no-index",
         f"--find-links={comp_wheels}", "--upgrade", "--target", str(pillow_target),
         "pillow"], timeout=180)
    sys.path.insert(0, str(pillow_target))
    for mod in [m for m in list(sys.modules) if m.startswith("PIL")]:
        del sys.modules[mod]
    print("Pillow overlay installed to", pillow_target)

# transformers 5.7.0 overlay (gotcha #14)
tx_roots = [
    Path("/kaggle/input/arc-agi-3-transformers-wheels"),
    Path("/kaggle/input/datasets/cataluna84/arc-agi-3-transformers-wheels"),
    Path("/kaggle/input"),
]
tx_wheels = find_dir_with("transformers-*.whl", tx_roots)
if tx_wheels is not None:
    tx_target = Path("/tmp/_transformers_pkg")
    tx_target.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "--no-index",
         f"--find-links={tx_wheels}", "--upgrade", "--target", str(tx_target), "--no-deps",
         "transformers", "tokenizers", "accelerate", "huggingface_hub",
         "safetensors", "regex", "filelock", "fsspec", "pyyaml", "tqdm"], timeout=300)
    sys.path.insert(0, str(tx_target))
    purge_roots = {"transformers", "tokenizers", "accelerate", "huggingface_hub", "safetensors"}
    for mod in [m for m in list(sys.modules) if m.split(".")[0] in purge_roots]:
        del sys.modules[mod]
    print("transformers overlay installed to", tx_target)

print("Offline overlay install complete.")
"""


def cell2_writefile() -> list[str]:
    """Cell 2: %%writefile /kaggle/working/my_agent.py with the full inlined agent."""
    agent_src = (
        INLINED_HEADER
        + INLINED_STATE_GRAPH
        + INLINED_SEGMENTER
        + INLINED_TRIGGER_BFS
        + INLINED_QWEN
        + INLINED_MYAGENT
    )
    return ["%%writefile /kaggle/working/my_agent.py\n"] + agent_src.splitlines(keepends=True)


CELL3_MARKDOWN = ["this only runs if you submit to the competition, not when you do tests"]

CELL4_COMP_RERUN = [
    "import os\n",
    "if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n",
    "    os.environ['QWEN_MODEL_PATH'] = '/kaggle/input/qwen3-6-35b-a3b-bf16'\n",
    "    os.environ['QWEN_MAX_NEW_TOKENS'] = '24'\n",
    "    os.environ['QWEN_HISTORY_LEN'] = '6'\n",
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
]

CELL5_DUMMY = [
    "import os\n",
    "if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n",
    "    import pandas as pd\n",
    "    submission = pd.DataFrame(data=[['1_0','1',True,1]],columns=['row_id','game_id','end_of_game','score'])\n",
    "    submission.to_parquet('/kaggle/working/submission.parquet',index=False)",
]

CELL6_MARKDOWN = ["This is a dummy submission fallback, important to keep"]


def main() -> None:
    cells = [
        make_cell("code", CELL0_PIP),
        make_cell("code", CELL1_OVERLAY),
        make_cell("code", cell2_writefile()),
        make_cell("markdown", CELL3_MARKDOWN),
        make_cell("code", CELL4_COMP_RERUN),
        make_cell("code", CELL5_DUMMY),
        make_cell("markdown", CELL6_MARKDOWN),
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
    out_path = REPO / "experiments/exp004_qwen_agent/comp_kernel_v2/qwen_phase1_comp.ipynb"
    with out_path.open("w") as f:
        json.dump(notebook, f, indent=1)
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
