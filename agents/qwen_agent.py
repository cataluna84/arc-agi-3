"""qwen_agent.py - Vision-language agent backed by Qwen3.6-35B-A3B (BF16).

Model: `Qwen/Qwen3.6-35B-A3B` (image-text-to-text, MoE 35B total / 3B active).
Loads via `transformers 5.0.0 + accelerate` (both pre-installed on Kaggle's
H100 image; see `runs/h100_probe/`). Uses BF16 to fit in 80 GB H100 HBM3
without quantization (Apache-2.0 license, ~70 GB shards).

Why both image and text in the prompt: ARC-AGI-3 frames are 64x64 with 16
colors. The vision encoder sees the full grid as a 512x512 PIL image
(8x nearest-neighbour upscale) for spatial pattern recognition; in parallel,
the chat message includes a compact hex-grid text dump so the model can
cross-check exact cell values without relying solely on the visual encoder's
decoding of solid colour blocks.

Local-only smoke test (no GPU, no model load - exercises prompt build + parser):
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
    QWEN_MAX_NEW_TOKENS  - cap per generation (default: 96)
    QWEN_HISTORY_LEN     - frames of (action, changed?) history in prompt (default: 8)
    QWEN_DEBUG_PROMPTS   - if "1", logs each prompt + response to /tmp/qwen_trace.log
"""

from __future__ import annotations

import contextlib
import os
import re
import time
from collections import deque
from typing import Any

from . import GameAction, GameState

# These three imports are deliberately lazy (deferred to _ensure_model_loaded)
# so this module can be imported without pulling in torch / transformers / PIL.
# That keeps `scripts/qwen_agent_smoke_local.py` runnable on a CPU-only laptop
# for prompt-build + parser checks.

_DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
_DEFAULT_DTYPE = "bf16"
_DEFAULT_DEVICE_MAP = "auto"
# Decode-time dominates per-action latency (~57 ms/token on H100 BF16 35B-A3B).
# An action label is 1-3 tokens; a coord pair another ~6. Cap at 16 to stay
# under 1 s/action steady-state, leaving budget for 25 games x ~300 actions
# in <6 h wall.
_DEFAULT_MAX_NEW = 16
_DEFAULT_HIST_LEN = 8
_FRAME_UPSCALE = 8  # 64*8 = 512 px; matches Qwen vision encoder's preferred resolution

# 16-colour palette (matches arc-agi-3 conventions: 0=black, 1-15 = standard ARC colours)
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
    """Render a 64x64 grid (numpy or list-of-list) as 64 lines of hex digits.

    Each cell is one character in 0-9a-f (16 colours). Saves ~3-4x tokens vs
    space-separated decimals, which matters for keeping the prompt short.
    """
    if grid is None:
        return ""
    rows = []
    tolist = getattr(grid, "tolist", None)
    iterable = tolist() if callable(tolist) else grid
    for row in iterable:
        rows.append("".join(f"{int(v) % 16:x}" for v in row))
    return "\n".join(rows)


def _frame_to_pil_image(grid: Any) -> Any:
    """Render a 64x64 grid into a PIL.Image upscaled by _FRAME_UPSCALE.

    PIL is imported lazily so this module loads cleanly on machines without
    Pillow. Returns None if PIL is missing (callers fall back to text-only).
    """
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
    history: list[tuple[str, bool]],
    *,
    text_grid: bool = True,
) -> tuple[list[dict], Any | None]:
    """Construct the chat-message list and the PIL image (or None) to feed Qwen.

    Returns (messages, image). `messages` follows the OpenAI/transformers chat
    convention with `image` + `text` content parts. `image` is also returned
    separately so callers using `processor(text=..., images=[img], ...)` can
    pass it through; vision-language pipelines vary.
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
    # text_grid kept for API compat / future hybrid prompts; current prompt
    # is image-only to keep input tokens (and decode time) low.
    _ = _frame_to_text_grid(grid) if text_grid else ""

    # History: most-recent first, capped externally
    hist_lines = []
    for action_name, changed in history:
        marker = "changed" if changed else "no-change"
        hist_lines.append(f"  - {action_name}: {marker}")
    hist_str = "\n".join(hist_lines) if hist_lines else "  (no actions yet)"

    system = (
        "You are an ARC-AGI-3 game agent. Reply with ONE action only, NO "
        "explanation. Format: 'ACTIONn' on one line (e.g. ACTION3). For "
        "ACTION6 add '(x, y)' both in [0,63]. Pick an action that visibly "
        "changes the grid; vary actions if the previous one made no change."
    )

    user_content_parts: list[dict] = []
    if image is not None:
        user_content_parts.append({"type": "image", "image": image})
    user_text = (
        f"state={state_str} level={levels}/{win_levels}\n"
        f"Available: {avail_str}\n"
        f"History (oldest->newest):\n{hist_str}\n"
        "Output ONE action only."
    )
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
        del seed, arc_env, kwargs  # unused (model is deterministic by default)
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

        # Frame-change detection across calls
        self._prev_grid_hash: int = 0
        self._prev_action_name: str | None = None
        # Capped history of (action_name, frame_changed)
        self._history: deque[tuple[str, bool]] = deque(maxlen=self.history_len)

        # Lazy model attrs
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._loaded = False

    # ---- Heavy-init -------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import torch  # type: ignore
            from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore
        except ImportError as e:
            raise ImportError(
                "QwenAgent requires `torch` and `transformers`. On Kaggle's H100 "
                "image both are pre-installed; locally install via "
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

    # ---- Main entrypoint --------------------------------------------------

    def choose_action(self, frame: Any) -> GameAction:
        # 1. Always handle reset states deterministically (no model call)
        state = getattr(frame, "state", None)
        if state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._record_history(frame, "RESET")
            return GameAction.RESET

        # 2. Update frame-change history before adding the next entry
        self._update_history_from_observation(frame)

        # 3. Build prompt (image + text grid + available_actions + history)
        messages, image = build_prompt(frame, list(self._history))

        # 4. Lazily load the model on first real call
        self._ensure_model_loaded()
        torch = self._torch

        # 5. Render chat -> tokens. transformers 5.x supports `apply_chat_template`
        # with `add_generation_prompt=True`; processors that wrap tokenizer
        # accept the messages list directly via the convenience API.
        text_in = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        proc_kwargs: dict = {"text": text_in, "return_tensors": "pt"}
        if image is not None:
            proc_kwargs["images"] = [image]
        inputs = self._processor(**proc_kwargs).to(self._model.device)

        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                use_cache=True,
            )
        # Strip the prompt tokens; decode only the generated suffix.
        gen_tokens = out[0, inputs["input_ids"].shape[1] :]
        reply = self._processor.batch_decode([gen_tokens], skip_special_tokens=True)[0]

        if self._debug:
            self._log(f"---\nPROMPT-tail:\n{text_in[-1500:]}\n\nREPLY:\n{reply}\n")

        # 6. Parse action from reply, snap to available_actions
        avail = [int(a) for a in (getattr(frame, "available_actions", None) or [])]
        action, coords = parse_action(reply, avail)

        # 6b. Anti-repeat: greedy decoding can lock the model into ACTION1
        #     forever if the first-action prediction is sticky. If the recent
        #     history shows this action has been picked AND yielded no-change
        #     >= STUCK_THRESHOLD times in a row, deterministically rotate to
        #     the next available action.
        STUCK_THRESHOLD = 3
        recent_no_change = [name for name, changed in self._history if not changed]
        n_recent_same = sum(1 for n in recent_no_change[-STUCK_THRESHOLD:] if n == action.name)
        if n_recent_same >= STUCK_THRESHOLD and avail:
            sorted_avail = sorted(avail)
            try:
                idx = sorted_avail.index(int(action.name.replace("ACTION", "")))
                next_id = sorted_avail[(idx + 1) % len(sorted_avail)]
            except (ValueError, IndexError):
                next_id = sorted_avail[0]
            action = GameAction.from_id(next_id)
            coords = None  # let downstream pick centre / random for ACTION6

        # 7. ACTION6 (click) coords: prefer parsed; else centre of grid
        if action.is_complex():
            x, y = coords if coords is not None else (32, 32)
            action.set_data({"x": int(x) % 64, "y": int(y) % 64})

        # 8. Cache for next call's frame-change detection
        layers = getattr(frame, "frame", None) or []
        grid = layers[0] if layers else None
        self._prev_grid_hash = self._grid_hash(grid)
        self._prev_action_name = action.name

        return action

    # ---- History ----------------------------------------------------------

    def _record_history(self, frame: Any, action_name: str) -> None:
        layers = getattr(frame, "frame", None) or []
        grid = layers[0] if layers else None
        new_hash = self._grid_hash(grid)
        # If we have a previous action, log whether the frame changed because of it
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
