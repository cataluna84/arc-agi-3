"""qwen_agent.py - Vision-language agent backed by Qwen3.6-35B-A3B (BF16).

Phase-1 "Direct Policy++" (D17, 2026-05-15): the agent uses an internal
StateGraph + frame-segmenter to provide structured context (visit count,
untried actions, ACTION6 click candidates) to the model and falls back
to TriggerBFS on any unhandled exception (gotcha #17).

Model: `Qwen/Qwen3.6-35B-A3B` (image-text-to-text, MoE 35B total / 3B active).
Loads via `transformers 5.7.0 + accelerate` on Kaggle's RTX 6000 image (94 GB
VRAM); BF16 weights allocate ~70 GB and inference proceeds with ~24 GB
headroom for KV cache + activations.

Why both image and text in the prompt: ARC-AGI-3 frames are 64x64 with 16
colors. The vision encoder sees the full grid as a 512x512 PIL image
(8x nearest-neighbour upscale) for spatial pattern recognition; in parallel,
the chat message includes a compact list of ACTION6 click candidates from
frame segmentation so the model can ground click coordinates in actual
interactive regions instead of inventing them.

Local-only smoke test (no GPU, no model load - exercises Phase-1 logic):
    .venv/bin/python scripts/qwen_phase1_smoke_local.py

Backward-compat 22-check smoke:
    .venv/bin/python scripts/qwen_agent_smoke_local.py

Kaggle dev kernel use (model weights mounted at /kaggle/input/<slug>/):
    QWEN_MODEL_PATH=/kaggle/input/qwen3-6-35b-a3b-bf16 \\
    .venv/bin/python experiments/local_runner.py \\
        --agent agents.qwen_agent:QwenAgent \\
        --use-sdk --games ls20 --max-actions 200

Environment variables (all optional):
    QWEN_MODEL_PATH      - HF model id or local dir (default: Qwen/Qwen3.6-35B-A3B)
    QWEN_DTYPE           - torch dtype: bf16|fp16|fp32 (default: bf16)
    QWEN_DEVICE_MAP      - accelerate device_map (default: auto)
    QWEN_MAX_NEW_TOKENS  - cap per generation (default: 24, bumped for JSON output)
    QWEN_HISTORY_LEN     - outcome-memory entries in prompt (default: 6)
    QWEN_DEBUG_PROMPTS   - if "1", logs each prompt + response to /tmp/qwen_trace.log
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import re
import time
from collections import deque
from typing import Any

from . import GameAction, GameState
from .state_graph import StateGraph, hash_frame  # noqa: F401 - kept for back-compat
from .trigger_bfs_agent import TriggerBFSAgent, _trigger_score


def hash_frame_masked(frame_layers) -> bytes:
    """8-byte blake2b digest of the PRIMARY visual layer with status-bar pixels masked.

    Two compounding mitigations for the D17 ACTION6-spam collapse:

    1. **Primary-layer only**: many ARC-AGI-3 games expose more than one layer
       (`frame.frame[1+]` often carries score/animation overlays that drift
       per step). The legacy `state_graph.hash_frame` iterates ALL layers, so
       a stable game grid + a drifting overlay still produces a fresh digest
       every step. We hash only `frame_layers[0]` (the primary visual layer)
       to drop overlay drift. The D17 smoke confirmed ft09's
       `no_change_rate` over `frame.frame[0]` was 100%, so this alone fixes
       ft09.

    2. **Status-bar masking (best effort)**: even within the primary layer, a
       step counter or score bar can drift. When `frame_segmenter` detects a
       status-bar segment, its pixels are rewritten to `STATUS_BAR_COLOR`
       before hashing so equivalent game states collide.

    Falls back to the unmasked layer-bytes digest on any failure (gotcha #17
    style: never raise from a hash function).
    """
    h = hashlib.blake2b(digest_size=8)
    if not frame_layers:
        return h.digest()
    layer = frame_layers[0]
    try:
        import numpy as np

        from . import frame_segmenter as fs

        grid = layer if isinstance(layer, np.ndarray) else np.asarray(layer, dtype=np.uint8)
        if grid.ndim != 2:
            h.update(grid.tobytes())
            return h.digest()
        label_map, segments = fs.segment_frame(grid)
        sb_mask, _ = fs.identify_status_bars(label_map, segments)
        if sb_mask is not None and sb_mask.any():
            masked = grid.copy()
            masked[sb_mask] = fs.STATUS_BAR_COLOR
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


# Heavy imports (torch, transformers, PIL) are deferred to _ensure_model_loaded
# so this module can be imported on a CPU-only laptop for prompt-build + parser
# checks.

_DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
_DEFAULT_DTYPE = "bf16"
_DEFAULT_DEVICE_MAP = "auto"
# JSON output target is ~16-20 tokens; cap at 24 for safety margin.
_DEFAULT_MAX_NEW = 24
# Outcome-memory entries fed to the model per call. Richer per-entry content
# (delta, levelΔ) costs tokens, so we trim from 8 to 6 vs the D2 baseline.
_DEFAULT_HIST_LEN = 6
# Top-K ACTION6 click candidates surfaced in the prompt.
_DEFAULT_CANDIDATES = 8
_FRAME_UPSCALE = 8  # 64*8 = 512 px; matches Qwen vision encoder's preferred res

# 16-colour palette (matches arc-agi-3 conventions: 0=black, 1-15 ARC colours)
_PALETTE: list[tuple[int, int, int]] = [
    (0, 0, 0),  # 0  black
    (30, 147, 255),  # 1  blue
    (249, 60, 49),  # 2  red
    (79, 204, 48),  # 3  green
    (255, 220, 0),  # 4  yellow
    (153, 153, 153),  # 5  grey
    (229, 58, 163),  # 6  pink (fuchsia)
    (255, 133, 27),  # 7  orange
    (135, 216, 241),  # 8  light blue
    (146, 18, 49),  # 9  dark red
    (96, 16, 96),  # 10 deep purple (extras for 16-colour)
    (16, 96, 16),  # 11
    (16, 16, 96),  # 12
    (96, 96, 16),  # 13
    (96, 16, 96),  # 14
    (240, 240, 240),  # 15 near-white
]


def _frame_to_text_grid(grid: Any) -> str:
    """Render a 64x64 grid (numpy or list-of-list) as 64 lines of hex digits."""
    if grid is None:
        return ""
    rows = []
    tolist = getattr(grid, "tolist", None)
    iterable = tolist() if callable(tolist) else grid
    for row in iterable:
        rows.append("".join(f"{int(v) % 16:x}" for v in row))
    return "\n".join(rows)


def _frame_to_pil_image(grid: Any) -> Any:
    """Render a 64x64 grid into a PIL.Image upscaled by _FRAME_UPSCALE."""
    try:
        from PIL import Image  # type: ignore
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
    return img.resize(
        (W * _FRAME_UPSCALE, H * _FRAME_UPSCALE),
        resample=Image.Resampling.NEAREST,
    )


# Regex used by the action parser. Tolerates: "ACTION3", "action 3", "Action: 6 (12, 34)"
_ACTION_RE = re.compile(
    r"\bACTION\s*[_:]?\s*(\d)\b|\bRESET\b|\bACTION_(\d)\b",
    flags=re.IGNORECASE,
)
_COORD_RE = re.compile(
    r"\(?\s*x\s*[:=]?\s*(\d+)\s*[, ]\s*y\s*[:=]?\s*(\d+)\s*\)?", flags=re.IGNORECASE
)
_COORD_RE_BARE = re.compile(r"\((\d{1,2})\s*,\s*(\d{1,2})\)")
# Phase-1 JSON output: tolerant of stray prose around the first {...} object.
_JSON_OBJ_RE = re.compile(
    r'\{[^{}]*"action"\s*:\s*"[^"]+"[^{}]*\}',
    flags=re.IGNORECASE,
)
_CAND_LABEL_RE = re.compile(r"C(\d+)", flags=re.IGNORECASE)


def _parse_json_first(text: str) -> dict | None:
    """Return the first JSON object whose payload mentions "action".

    Never raises. Returns None on any parse failure or if the matched
    substring doesn't decode as a dict.
    """
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


def parse_action(
    text: str,
    available_ids: list[int],
    rng_seed: int = 0,
) -> tuple[GameAction, tuple[int, int] | None]:
    """Extract a (GameAction, optional (x,y)) from the model's free-text reply.

    Strategy:
      1. Find every ACTIONn / RESET mention in `text`.
      2. Filter to those in `available_ids`.
      3. Pick the FIRST such mention (model's primary intent).
      4. If none match, fall back to the LAST resort: lowest available id.
      5. If chosen action is ACTION6, additionally try to parse (x,y) coords;
         if absent, return None and let the caller pick (e.g. centre / random).
    """
    candidates: list[int] = []
    for m in _ACTION_RE.finditer(text or ""):
        if m.group(0).upper().startswith("RESET"):
            candidates.append(0)
        else:
            d = m.group(1) or m.group(2)
            if d is not None:
                with contextlib.suppress(ValueError):
                    candidates.append(int(d))

    avail_set = {int(a) for a in available_ids}
    if avail_set:
        filtered = [c for c in candidates if c in avail_set]
    else:
        filtered = candidates

    if filtered:
        chosen = filtered[0]
    elif avail_set:
        chosen = sorted(avail_set)[0]  # deterministic fallback
    else:
        chosen = 1  # last-ditch: ACTION1

    coords: tuple[int, int] | None = None
    if chosen == 6:
        m = _COORD_RE.search(text or "")
        if not m:
            m = _COORD_RE_BARE.search(text or "")
        if m:
            try:
                x = int(m.group(1)) % 64
                y = int(m.group(2)) % 64
                coords = (x, y)
            except ValueError:
                pass

    return GameAction.from_id(chosen), coords


def build_prompt(
    frame: Any,
    history: list,
    *,
    text_grid: bool = True,
    visit_count: int = 0,
    untried: list[int] | None = None,
    action6_candidates: list[dict] | None = None,
    tried_clicks: list[tuple[int, int]] | None = None,
) -> tuple[list[dict], Any | None]:
    """Construct the chat-message list and PIL image (or None) to feed Qwen.

    Phase-1 signature extends the original with `visit_count`, `untried`,
    `action6_candidates`, and `tried_clicks`. Old 2-tuple history entries
    `(name, changed_bool)` are still accepted for backward compatibility
    with the existing `scripts/qwen_agent_smoke_local.py` 22-check smoke.

    `tried_clicks` is the list of (x, y) ACTION6 coords already exercised at
    this state-hash with no resulting frame change. The system prompt
    instructs the model to PICK A DIFFERENT click candidate when this list
    is non-empty (D17 Path B fix for ACTION6-only games where rotating to a
    different action id is impossible).
    """
    avail = [int(a) for a in (getattr(frame, "available_actions", None) or [])]
    avail_str = ", ".join(f"ACTION{a}" for a in avail) if avail else "ACTION1..ACTION7"
    state_str = getattr(getattr(frame, "state", None), "name", "?")
    levels = getattr(frame, "levels_completed", 0)
    win_levels = getattr(frame, "win_levels", 1)

    # Grid: pick the FIRST layer (most games are single-layer).
    layers = getattr(frame, "frame", None) or []
    grid = layers[0] if layers else None
    image = _frame_to_pil_image(grid)
    # text_grid kept for API compat / future hybrid prompts; current prompt is
    # image-only to keep input tokens (and decode time) low.
    _ = _frame_to_text_grid(grid) if text_grid else ""

    # History formatting: accepts both 2-tuple (legacy) and 3-tuple (Phase-1).
    hist_lines: list[str] = []
    for entry in history or []:
        if isinstance(entry, (tuple, list)) and len(entry) >= 3:
            name = str(entry[0])
            change_score = float(entry[1] or 0.0)
            level_delta = int(entry[2] or 0)
            marker = "changed" if change_score > 0 else "no-change"
            hist_lines.append(
                f"  - {name}: {marker} delta={change_score:.0f} levelΔ={level_delta:+d}"
            )
        elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
            name = str(entry[0])
            changed = bool(entry[1])
            marker = "changed" if changed else "no-change"
            hist_lines.append(f"  - {name}: {marker}")
    hist_str = "\n".join(hist_lines) if hist_lines else "  (no actions yet)"

    system = (
        "You are an ARC-AGI-3 game agent. Do not reason step-by-step. Output "
        "ONE action only as STRICT JSON on one line: "
        '{"action":"ACTIONn","x":int|null,"y":int|null,"why":"<=12 chars"}. '
        "Pick ACTIONn from Available. For ACTION6 you MUST set x,y in [0,63] -- "
        "PREFER one of the listed Click candidates. If 'Already tried at this "
        "state' is non-empty, PICK A DIFFERENT click candidate (different (x,y)) "
        "because the listed coords have already produced no change here. Avoid "
        "actions in the recent history that caused no-change at this state."
    )

    user_content_parts: list[dict] = []
    if image is not None:
        user_content_parts.append({"type": "image", "image": image})

    text_parts = [
        f"state={state_str} level={levels}/{win_levels}",
        f"visit_count={visit_count}",
        f"Available: {avail_str}",
    ]
    if untried:
        untried_names = ", ".join(f"ACTION{a}" for a in untried)
        text_parts.append(f"Untried at this state: {untried_names}")
    if action6_candidates:
        cand_lines = ["Click candidates:"]
        for c in action6_candidates:
            cand_lines.append(
                f"  - {c.get('label', '?')} ACTION6 "
                f"({int(c.get('x', 0))},{int(c.get('y', 0))}) "
                f"tier{int(c.get('tier', 4))}"
            )
        text_parts.append("\n".join(cand_lines))
    if tried_clicks:
        tried_str = ", ".join(f"({int(x)},{int(y)})" for x, y in tried_clicks)
        text_parts.append(
            f"Already tried at this state (no-change): {tried_str}. "
            "Pick a DIFFERENT click candidate."
        )
    text_parts.append(f"History (oldest->newest):\n{hist_str}")
    text_parts.append("Output ONE JSON object only.")
    user_text = "\n".join(text_parts)
    user_content_parts.append({"type": "text", "text": user_text})

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content_parts},
    ]
    return messages, image


# ---------------------------------------------------------------------------
# QwenAgent: the actual local_runner-compatible class
# ---------------------------------------------------------------------------


class QwenAgent:
    """ARC-AGI-3 agent driven by Qwen3.6-35B-A3B (BF16, image-text-to-text).

    Heavy deps (torch, transformers, PIL) are imported lazily in
    `_ensure_model_loaded()` so the module can be unit-tested without them.
    """

    name = "qwen3_6_35b_a3b"

    def __init__(
        self,
        seed: int = 0,
        arc_env: Any | None = None,
        game_id: str = "",
        model_path: str | None = None,
        dtype: str | None = None,
        device_map: str | None = None,
        max_new_tokens: int | None = None,
        history_len: int | None = None,
        **kwargs: Any,
    ) -> None:
        del arc_env, kwargs  # unused (model is deterministic by default)
        self.game_id = game_id

        self.model_path = model_path or os.environ.get("QWEN_MODEL_PATH") or _DEFAULT_MODEL
        self.dtype_str = (dtype or os.environ.get("QWEN_DTYPE") or _DEFAULT_DTYPE).lower()
        self.device_map = device_map or os.environ.get("QWEN_DEVICE_MAP") or _DEFAULT_DEVICE_MAP
        self.max_new_tokens = int(
            max_new_tokens or os.environ.get("QWEN_MAX_NEW_TOKENS") or _DEFAULT_MAX_NEW
        )
        self.history_len = int(
            history_len or os.environ.get("QWEN_HISTORY_LEN") or _DEFAULT_HIST_LEN
        )
        self._debug = os.environ.get("QWEN_DEBUG_PROMPTS") == "1"

        # Backward-compat 2-tuple history (used by existing FakeQwen smoke).
        self._prev_grid_hash: int = 0
        self._prev_action_name: str | None = None
        self._history: deque[tuple[str, bool]] = deque(maxlen=self.history_len)

        # Phase-1 outcome memory (3-tuple) and state-graph integration.
        self._outcome_history: deque[tuple[str, float, int]] = deque(maxlen=self.history_len)
        self._state_graph = StateGraph()
        self._prev_layers: list | None = None
        self._prev_hash: bytes | None = None
        self._prev_action_id: int | None = None
        self._prev_levels: int = 0

        # D17 Path B: per-state-hash list of (x, y) ACTION6 coords already tried
        # with no resulting frame change. Surfaced to the LLM so it can choose a
        # different click candidate on the next step. Cleared on level transition.
        self._tried_clicks_per_state: dict[bytes, list[tuple[int, int]]] = {}
        self._prev_click: tuple[int, int] | None = None

        # Single shared TriggerBFS fallback for catastrophic failure path.
        self._fallback = TriggerBFSAgent(seed=int(seed) if seed is not None else 0)

        # Smoke counters (per-game; resettable).
        self._fallback_count: int = 0
        self._parse_failure_count: int = 0

        # Lazy model attrs.
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._loaded = False

    # ---- Smoke instrumentation -------------------------------------------

    def reset_counters(self) -> None:
        """Zero per-game counters (used by the 4-game smoke harness)."""
        self._fallback_count = 0
        self._parse_failure_count = 0

    # ---- Heavy-init -------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import torch  # type: ignore
            from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore
        except ImportError as e:
            raise ImportError(
                "QwenAgent requires `torch` and `transformers`. On Kaggle's RTX "
                "6000 image both are pre-installed (with the offline overlay for "
                "transformers 5.7.0); locally install via "
                "`uv pip install --python .venv/bin/python torch transformers`. "
                f"Underlying error: {e}"
            )

        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.dtype_str, torch.bfloat16)

        t0 = time.time()
        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()
        self._torch = torch
        self._loaded = True
        if self._debug:
            self._log(f"[QwenAgent] loaded {self.model_path} in {time.time() - t0:.1f}s")

    # ---- Helpers ----------------------------------------------------------

    def _log(self, msg: str) -> None:
        try:
            with open("/tmp/qwen_trace.log", "a") as fh:
                fh.write(msg + "\n")
        except OSError:
            pass

    def _grid_hash(self, grid: Any) -> int:
        if grid is None:
            return 0
        tobytes = getattr(grid, "tobytes", None)
        if callable(tobytes):
            return hash(tobytes())
        return hash(tuple(tuple(int(v) for v in row) for row in grid))

    def _segment_action6_candidates(self, frame_layers: list | None) -> list[dict]:
        """Return up to `_DEFAULT_CANDIDATES` ACTION6 click candidates.

        Mirrors `trigger_bfs_agent._sample_click_xy` but returns top-K dicts
        with keys (label, x, y, tier) instead of picking one. Whole body is
        wrapped in try/except → [] for defensive safety (gotcha #17).
        """
        try:
            import numpy as np

            from . import frame_segmenter as fs

            if not frame_layers:
                return []
            layer = frame_layers[-1]
            if isinstance(layer, np.ndarray):
                grid = layer
            else:
                grid = np.asarray(layer, dtype=np.uint8)
            if grid.ndim != 2:
                return []

            label_map, segments = fs.segment_frame(grid)
            _sb_mask, sb_groups = fs.identify_status_bars(label_map, segments)
            sb_ids: set[int] = set()
            for g in sb_groups:
                sb_ids.update(g)
            tiers = fs.frame_segments_to_priority_tiers(segments, status_bar_segment_ids=sb_ids)

            frame_pixels = grid.shape[0] * grid.shape[1]
            half = frame_pixels // 2
            rng = random.Random(0)

            results: list[dict] = []
            for tier_idx in range(4):
                tier_sids = [sid for sid in tiers[tier_idx] if segments[sid].area <= half]
                # Largest area first within tier (deterministic).
                tier_sids.sort(key=lambda sid: -segments[sid].area)
                for sid in tier_sids:
                    coord = fs.mask_to_click_coords(label_map, sid, rng=rng)
                    if coord is None:
                        continue
                    results.append(
                        {
                            "label": f"C{len(results)}",
                            "x": int(coord[0]),
                            "y": int(coord[1]),
                            "tier": tier_idx,
                        }
                    )
                    if len(results) >= _DEFAULT_CANDIDATES:
                        return results
            return results
        except Exception:
            return []

    def _apply_guards(
        self,
        parsed: dict,
        frame: Any,
        candidates: list[dict],
        node: Any | None = None,
    ) -> GameAction:
        """Snap action to available; fill ACTION6 from candidates; skip no-change."""
        avail = [int(a) for a in (getattr(frame, "available_actions", None) or [])]
        avail_set = {int(a) for a in avail}

        # Extract action from parsed.action using the same regex as parse_action.
        raw = parsed.get("action") if isinstance(parsed, dict) else ""
        raw_text = str(raw) if raw is not None else ""

        chosen: int | None = None
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

        # Snap to available if missing or out of range.
        if chosen is None or chosen not in avail_set:
            non_reset = sorted(a for a in avail_set if a != 0)
            chosen = non_reset[0] if non_reset else 1

        # Skip known-no-change action: if this state already tried `chosen`,
        # and the recorded successor has been visited at least once with
        # incoming_change_score == 0, rotate to the smallest untried action.
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

        # ACTION6: fill (x, y) from parsed > candidates > center.
        if action.is_complex():
            x_val: int | None = None
            y_val: int | None = None
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

    # ---- Main entrypoint --------------------------------------------------

    def choose_action(self, frame: Any) -> GameAction:
        """Phase-1 outer wrapper: catches any exception and falls back to TriggerBFS."""
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

    def _choose_action_inner(self, frame: Any) -> GameAction:
        # 1. Reset states.
        state = getattr(frame, "state", None)
        if state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._record_history(frame, "RESET")
            return GameAction.RESET

        # 2. Backward-compat: update legacy 2-tuple history for FakeQwen smoke.
        self._update_history_from_observation(frame)

        # 3. State-graph integration.
        cur_layers = list(getattr(frame, "frame", []) or [])
        # Masked hash collapses status-bar drift to a stable digest (D17 fix).
        cur_hash = hash_frame_masked(cur_layers) if cur_layers else b"\x00" * 8
        cur_levels = int(getattr(frame, "levels_completed", 0))

        # Level guard (gotcha #19: gate on cur_levels >= 0).
        if cur_levels >= 0 and cur_levels != self._prev_levels:
            self._state_graph.maybe_reset_for_level(cur_levels)
            self._outcome_history.clear()
            self._tried_clicks_per_state.clear()  # D17 Path B

        avail = list(getattr(frame, "available_actions", []) or [1, 2, 3, 4, 5, 6, 7])
        node = self._state_graph.add_or_get(cur_hash, available_actions=avail, levels=cur_levels)

        # Observe the previous edge and append to outcome history.
        if self._prev_hash is not None and self._prev_action_id is not None:
            change_score = _trigger_score(
                self._prev_layers, cur_layers, self._prev_levels, cur_levels
            )
            self._state_graph.observe(self._prev_hash, self._prev_action_id, cur_hash, change_score)
            level_delta = cur_levels - self._prev_levels
            self._outcome_history.append(
                (
                    self._prev_action_name or f"ACTION{self._prev_action_id}",
                    float(change_score),
                    int(level_delta),
                )
            )
            # D17 Path B: if the previous step was ACTION6 with no frame change
            # AND we stayed at the same state, record the (x, y) so the next
            # prompt can ask the LLM to try a DIFFERENT click candidate.
            if (
                self._prev_action_id == 6
                and change_score == 0.0
                and self._prev_hash == cur_hash
                and self._prev_click is not None
            ):
                bucket = self._tried_clicks_per_state.setdefault(cur_hash, [])
                if self._prev_click not in bucket:
                    bucket.append(self._prev_click)

        # 4. Candidate generation.
        avail_set = {int(a) for a in avail}
        candidates = self._segment_action6_candidates(cur_layers) if 6 in avail_set else []
        untried = sorted(a for a in node.untried_actions if a in avail_set and a != 0)
        tried_clicks_here = list(self._tried_clicks_per_state.get(cur_hash, []))

        # 5. Build prompt with structured context.
        messages, image = build_prompt(
            frame,
            list(self._outcome_history),
            visit_count=node.visit_count,
            untried=untried,
            action6_candidates=candidates,
            tried_clicks=tried_clicks_here,
        )

        # 6. Lazily load the model on first real call.
        self._ensure_model_loaded()
        torch = self._torch

        # 7. Render chat -> tokens. enable_thinking=False is mandatory (gotcha #22).
        try:
            text_in = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text_in = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        proc_kwargs: dict = {"text": text_in, "return_tensors": "pt"}
        if image is not None:
            proc_kwargs["images"] = [image]
        inputs = self._processor(**proc_kwargs).to(self._model.device)

        # 8. Generate (gotcha #17 — exception re-raises to outer fallback).
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                use_cache=True,
            )
        gen_tokens = out[0, inputs["input_ids"].shape[1] :]
        reply = self._processor.batch_decode([gen_tokens], skip_special_tokens=True)[0]

        if self._debug:
            self._log(f"---\nPROMPT-tail:\n{text_in[-1500:]}\n\nREPLY:\n{reply}\n")

        # 9. Parse JSON first; fall back to {"action": reply} for the regex path.
        parsed = _parse_json_first(reply)
        if parsed is None:
            self._parse_failure_count += 1
            parsed = {"action": reply}

        # 10. Apply guards (snap to avail, fill ACTION6, skip known no-change).
        action = self._apply_guards(parsed, frame, candidates, node)

        # 11. Bookkeeping for next call.
        layers = getattr(frame, "frame", None) or []
        grid = layers[0] if layers else None
        self._prev_grid_hash = self._grid_hash(grid)
        self._prev_action_name = action.name
        self._prev_action_id = int(action.value)
        self._prev_layers = cur_layers
        self._prev_levels = cur_levels
        self._prev_hash = cur_hash
        data = getattr(action, "_data", {}) or {}
        # Capture click coords for D17 Path B tracking. Works for both
        # arcengine.GameAction (stores on action_data) and the mock IntEnum.
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

    # ---- History (legacy 2-tuple; kept for FakeQwen smoke compat) ---------

    def _record_history(self, frame: Any, action_name: str) -> None:
        layers = getattr(frame, "frame", None) or []
        grid = layers[0] if layers else None
        new_hash = self._grid_hash(grid)
        if self._prev_action_name is not None:
            changed = new_hash != self._prev_grid_hash
            self._history.append((self._prev_action_name, changed))
        self._prev_grid_hash = new_hash
        self._prev_action_name = action_name

    def _update_history_from_observation(self, frame: Any) -> None:
        if self._prev_action_name is None:
            return
        layers = getattr(frame, "frame", None) or []
        grid = layers[0] if layers else None
        new_hash = self._grid_hash(grid)
        changed = new_hash != self._prev_grid_hash
        self._history.append((self._prev_action_name, changed))

    def is_done(self, frame: Any) -> bool:
        return getattr(frame, "state", None) == GameState.WIN


__all__ = ["QwenAgent", "build_prompt", "parse_action"]
