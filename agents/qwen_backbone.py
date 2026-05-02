"""qwen_backbone.py - shared Qwen3.6-35B-A3B inference wrapper.

Used by all five exp004 reboot variants:
    - agents/qwen_policy_agent.py        (Option A: direct policy)
    - agents/qwen_clicker.py             (Option D: ACTION6 specialist)
    - agents/qwen_verifier.py            (Option B: BFS-frontier scorer)
    - agents/qwen_orchestrator.py        (Option C: explore/replay mode picker)
    - agents/qwen_world_model.py         (Option E: state-distance scorer)

Design:

* **Two runtimes, one API**. Tries vLLM 0.7+ with `enable_prefix_caching=True`
  first (the proper fix to POSTMORTEM cause 4.2). Falls back to transformers
  5.7+ greedy/sample. Local CPU smoke tests use neither -- they exercise
  prompt building and parsing only.

* **Lazy heavy-imports**. `torch`, `vllm`, `transformers`, `PIL` are imported
  inside `_ensure_loaded()` so this module can be imported on a CPU laptop
  without a 60-second torch import.

* **Frame rendering at upscale=4** (default; configurable). 64x64 grid
  upscaled 4x -> 256x256 PIL image -> 64 patch tokens. Quarter the prefill
  cost of upscale=8 (256 patch tokens). POSTMORTEM 4.2 secondary mitigation.

* **Change-feedback history**. Records `(action_name, frame_changed, dpx,
  dlevels)` per turn so the prompt can encode "this action did/didn't move
  the world" rather than just "this action was tried 30 times". Closes
  POSTMORTEM cause 4.4.

* **Universal action parser**. Tolerates `ACTION3`, `action 3`,
  `Action: 6 (12, 34)`, JSON-style `{"action": 6, "x": 12, "y": 34}`.
  Snaps to `available_actions`. ACTION6 coords parsed from text or
  sampled by caller. Closes POSTMORTEM cause 4.5.

Environment variables (read once on construction; all optional):

    QWEN_MODEL_PATH      HF id or local dir (default Qwen/Qwen3.6-35B-A3B)
    QWEN_RUNTIME         "vllm" | "transformers" | "auto" (default auto)
    QWEN_DTYPE           "bf16" | "fp16" | "fp32" (default bf16)
    QWEN_MAX_NEW_TOKENS  cap per generation (default 16)
    QWEN_TEMPERATURE     sampling temperature (default 0.7)
    QWEN_TOP_P           nucleus sampling (default 0.9)
    QWEN_FRAME_UPSCALE   image upscale factor (default 4 -- POSTMORTEM 4.2)
    QWEN_GPU_UTIL        vLLM gpu_memory_utilization (default 0.92)
    QWEN_MAX_MODEL_LEN   vLLM max_model_len (default 8192)
    QWEN_DEBUG_PROMPTS   if "1", logs each prompt + response to /tmp/qwen_trace.log

Local-only smoke (no GPU, no model load):
    .venv/bin/python scripts/qwen_backbone_smoke_local.py
"""

from __future__ import annotations

import contextlib
import os
import re
import time
from typing import Any

# 16-color ARC palette (mirrors agents/qwen_agent.py)
_PALETTE: list[tuple[int, int, int]] = [
    (0, 0, 0),
    (30, 147, 255),
    (249, 60, 49),
    (79, 204, 48),
    (255, 220, 0),
    (153, 153, 153),
    (229, 58, 163),
    (255, 133, 27),
    (135, 216, 241),
    (146, 18, 49),
    (96, 16, 96),
    (16, 96, 16),
    (16, 16, 96),
    (96, 96, 16),
    (96, 16, 96),
    (240, 240, 240),
]


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------


def render_frame_text(grid: Any) -> str:
    """64x64 grid -> 64 lines of hex digits (one char per cell)."""
    if grid is None:
        return ""
    tolist = getattr(grid, "tolist", None)
    iterable = tolist() if callable(tolist) else grid
    rows = ["".join(f"{int(v) % 16:x}" for v in row) for row in iterable]
    return "\n".join(rows)


def render_frame_image(grid: Any, *, upscale: int = 4) -> Any | None:
    """64x64 grid -> PIL.Image upscaled by `upscale` (default 4 -> 256x256).

    POSTMORTEM 4.2: upscale=4 gives ~64 patch tokens (vs 256 at upscale=8).
    Returns None if PIL is unavailable.
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
    h = len(rows)
    w = len(rows[0])
    img = Image.new("RGB", (w, h), (0, 0, 0))
    px = img.load()
    for y in range(h):
        row = rows[y]
        for x in range(w):
            v = int(row[x]) % len(_PALETTE)
            px[x, y] = _PALETTE[v]
    return img.resize(
        (w * upscale, h * upscale),
        resample=Image.Resampling.NEAREST,
    )


# ---------------------------------------------------------------------------
# Universal action parsers
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(
    r"\bACTION\s*[_:]?\s*(\d)\b|\bACTION_(\d)\b",
    flags=re.IGNORECASE,
)
_COORD_RE_LABELED = re.compile(
    r"\(?\s*x\s*[:=]?\s*(\d+)\s*[, ]\s*y\s*[:=]?\s*(\d+)\s*\)?",
    flags=re.IGNORECASE,
)
_COORD_RE_BARE = re.compile(r"\((\d{1,2})\s*,\s*(\d{1,2})\)")
_JSON_ACTION_RE = re.compile(r'"action"\s*:\s*(\d)', flags=re.IGNORECASE)
_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_action_id(text: str, available_ids: list[int]) -> int:
    """Extract action id 1..7 from free text. Falls back to lowest available."""
    candidates: list[int] = []
    for m in _ACTION_RE.finditer(text or ""):
        d = m.group(1) or m.group(2)
        if d is not None:
            with contextlib.suppress(ValueError):
                candidates.append(int(d))
    for m in _JSON_ACTION_RE.finditer(text or ""):
        with contextlib.suppress(ValueError):
            candidates.append(int(m.group(1)))

    avail_set = {int(a) for a in available_ids if int(a) != 0}
    filtered = [c for c in candidates if c in avail_set] if avail_set else candidates
    if filtered:
        return filtered[0]
    if avail_set:
        return min(avail_set)
    return 1


def parse_coords(text: str) -> tuple[int, int] | None:
    """Try labeled form (x=12, y=34), then bare (12, 34). Returns (x, y) in [0,63]."""
    m = _COORD_RE_LABELED.search(text or "")
    if not m:
        m = _COORD_RE_BARE.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(1)) % 64, int(m.group(2)) % 64
    except ValueError:
        return None


def parse_score(text: str) -> float:
    """Extract first float in text. Returns 0.0 if none. Used by verifier role."""
    m = _FLOAT_RE.search(text or "")
    if not m:
        return 0.0
    try:
        return float(m.group(0))
    except ValueError:
        return 0.0


def parse_choice(text: str, valid_choices: list[str]) -> str | None:
    """Pick the first occurrence of any token in valid_choices (case-insensitive)."""
    if not text or not valid_choices:
        return None
    text_low = text.lower()
    best_pos = -1
    best = None
    for c in valid_choices:
        idx = text_low.find(c.lower())
        if idx >= 0 and (best_pos < 0 or idx < best_pos):
            best_pos = idx
            best = c
    return best


# ---------------------------------------------------------------------------
# Change-feedback history (POSTMORTEM 4.4)
# ---------------------------------------------------------------------------


class ChangeLog:
    """Capped history of (action_name, changed, dpx, dlevels) for prompt encoding."""

    def __init__(self, capacity: int = 8) -> None:
        self.capacity = capacity
        self._items: list[tuple[str, bool, int, int]] = []

    def add(self, action_name: str, changed: bool, dpx: int = 0, dlevels: int = 0) -> None:
        self._items.append((action_name, changed, dpx, dlevels))
        if len(self._items) > self.capacity:
            self._items = self._items[-self.capacity :]

    def render(self) -> str:
        if not self._items:
            return "  (no actions yet)"
        lines = []
        for name, changed, dpx, dlevels in self._items:
            if dlevels > 0:
                tag = f"LEVEL UP +{dlevels} (+{dpx} px changed)"
            elif changed:
                tag = f"changed (+{dpx} px)"
            else:
                tag = "no-change"
            lines.append(f"  - {name}: {tag}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._items)


# ---------------------------------------------------------------------------
# QwenBackbone
# ---------------------------------------------------------------------------


_DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"


class QwenBackbone:
    """Lazy-loaded Qwen inference wrapper supporting vLLM (preferred) or transformers.

    Construction is cheap; the actual model load is deferred to first `generate()`.
    """

    def __init__(
        self,
        model_path: str | None = None,
        runtime: str | None = None,
        dtype: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        frame_upscale: int | None = None,
        gpu_util: float | None = None,
        max_model_len: int | None = None,
    ) -> None:
        self.model_path = model_path or os.environ.get("QWEN_MODEL_PATH") or _DEFAULT_MODEL
        self.runtime = (runtime or os.environ.get("QWEN_RUNTIME") or "auto").lower()
        self.dtype_str = (dtype or os.environ.get("QWEN_DTYPE") or "bf16").lower()
        self.max_new_tokens = (
            max_new_tokens if max_new_tokens is not None else _env_int("QWEN_MAX_NEW_TOKENS", 16)
        )
        self.temperature = (
            temperature if temperature is not None else _env_float("QWEN_TEMPERATURE", 0.7)
        )
        self.top_p = top_p if top_p is not None else _env_float("QWEN_TOP_P", 0.9)
        self.frame_upscale = (
            frame_upscale if frame_upscale is not None else _env_int("QWEN_FRAME_UPSCALE", 4)
        )
        self.gpu_util = gpu_util if gpu_util is not None else _env_float("QWEN_GPU_UTIL", 0.92)
        self.max_model_len = (
            max_model_len if max_model_len is not None else _env_int("QWEN_MAX_MODEL_LEN", 8192)
        )
        self._debug = os.environ.get("QWEN_DEBUG_PROMPTS") == "1"

        self._loaded = False
        self._engine: Any = None
        self._processor: Any = None
        self._actual_runtime: str = "none"

    def _log(self, msg: str) -> None:
        if not self._debug:
            return
        try:
            with open("/tmp/qwen_trace.log", "a") as fh:
                fh.write(msg + "\n")
        except OSError:
            pass

    def _try_vllm(self) -> bool:
        try:
            from vllm import LLM  # type: ignore
        except ImportError:
            return False
        t0 = time.time()
        self._engine = LLM(
            model=self.model_path,
            dtype="bfloat16" if self.dtype_str.startswith("bf") else self.dtype_str,
            enable_prefix_caching=True,
            gpu_memory_utilization=self.gpu_util,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 1},
        )
        self._actual_runtime = "vllm"
        self._log(f"[QwenBackbone] vLLM loaded in {time.time() - t0:.1f}s")
        return True

    def _try_transformers(self) -> bool:
        try:
            import torch  # type: ignore
            from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore
        except ImportError:
            return False
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        dtype = dtype_map.get(self.dtype_str, torch.bfloat16)
        t0 = time.time()
        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self._engine = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()
        self._torch = torch
        self._actual_runtime = "transformers"
        self._log(f"[QwenBackbone] transformers loaded in {time.time() - t0:.1f}s")
        return True

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.runtime in ("vllm", "auto"):
            if self._try_vllm():
                self._loaded = True
                return
            if self.runtime == "vllm":
                raise ImportError("QWEN_RUNTIME=vllm but vllm is not installed")
        if self._try_transformers():
            self._loaded = True
            return
        raise ImportError("QwenBackbone needs either `vllm` or `transformers`+`torch` installed.")

    @property
    def actual_runtime(self) -> str:
        return self._actual_runtime

    def generate(self, messages: list[dict], image: Any | None = None) -> str:
        """Single-shot generate. messages is a chat-template list. Returns reply text."""
        self._ensure_loaded()
        if self._actual_runtime == "vllm":
            return self._generate_vllm(messages, image)
        return self._generate_transformers(messages, image)

    def _generate_vllm(self, messages: list[dict], image: Any | None) -> str:
        from vllm import SamplingParams  # type: ignore

        params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )
        if image is not None:
            for msg in messages:
                if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                    has_image = any(part.get("type") == "image" for part in msg["content"])
                    if not has_image:
                        msg["content"] = [{"type": "image", "image": image}, *msg["content"]]
                    break
        outputs = self._engine.chat(messages, sampling_params=params, use_tqdm=False)
        return outputs[0].outputs[0].text

    def _generate_transformers(self, messages: list[dict], image: Any | None) -> str:
        torch = self._torch
        text_in = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        proc_kwargs: dict = {"text": text_in, "return_tensors": "pt"}
        if image is not None:
            proc_kwargs["images"] = [image]
        inputs = self._processor(**proc_kwargs).to(self._engine.device)
        with torch.inference_mode():
            out = self._engine.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                use_cache=True,
            )
        gen_tokens = out[0, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode([gen_tokens], skip_special_tokens=True)[0]


__all__ = [
    "ChangeLog",
    "QwenBackbone",
    "parse_action_id",
    "parse_choice",
    "parse_coords",
    "parse_score",
    "render_frame_image",
    "render_frame_text",
]
