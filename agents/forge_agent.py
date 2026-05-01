"""forge_agent.py - Local adapter for the verbatim-ported FORGE v19 agent.

The actual agent source lives in `agents/_forge_v19.py` and is a
**bit-for-bit copy of cell #1 of the upstream Kaggle notebook we forked
on 2026-04-29** (see NOTICE for upstream attribution). Do not edit the
`_forge_v19.py` file directly -- if the upstream notebook updates,
re-pull and overwrite. Local adaptations belong in this file instead.

This module:
  1. Lazily imports the verbatim source so a missing `torch` install gives
     a clear error message instead of a confusing ImportError at module load.
  2. Patches `find_game_source_and_class` to also search our local
     `data/kaggle/arc-prize-2026-arc-agi-3/environment_files/` tree
     (the upstream version only knows about `/kaggle/input/...`).
  3. Exposes `ForgeAgent` -- a thin subclass of the upstream `MyAgent` that
     plugs into our `experiments/local_runner.py` contract:
         agent = ForgeAgent(seed=0, arc_env=env)
         action = agent.choose_action(latest_frame)
         done   = agent.is_done(latest_frame)
     while delegating all real work to the upstream class.

To run the FORGE agent locally you need PyTorch installed in the venv
(it's NOT in the Kaggle-bundled wheels and isn't a pyproject dep):

    uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu

(CPU wheel ~250 MB; CUDA wheels are larger.)

Smoke test:
    .venv/bin/python experiments/local_runner.py \\
        --agent agents.forge_agent:ForgeAgent \\
        --use-sdk --games ls20 --max-actions 200 --seed 0
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Local data-dir lookup (used by patched find_game_source_and_class)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_ENV_DIR = _REPO_ROOT / "data" / "kaggle" / "arc-prize-2026-arc-agi-3" / "environment_files"


def _find_game_source_and_class_local(game_id: str, arc_env: Any | None = None):
    """Local-aware drop-in for upstream's find_game_source_and_class.

    Search order:
      1. arc_env.environment_info.local_dir (canonical for the running env).
      2. data/kaggle/.../environment_files/<gid>/<guid>/<gid>.py
      3. The upstream Kaggle-only paths (no-op locally; preserved for parity).
    """
    import glob
    import logging
    import re

    logger = logging.getLogger(__name__)

    # 0. The arc_agi env already knows the local dir for the resolved game.
    if arc_env is not None:
        info = getattr(arc_env, "environment_info", None) or getattr(arc_env, "info", None)
        local_dir = getattr(info, "local_dir", None) if info else None
        class_name = getattr(info, "class_name", None) if info else None
        if local_dir:
            local_dir_path = Path(str(local_dir))
            if not local_dir_path.is_absolute():
                local_dir_path = _REPO_ROOT / local_dir_path
            for cand in (
                local_dir_path / f"{(class_name or '').lower()}.py",
                local_dir_path / f"{game_id.split('-')[0]}.py",
            ):
                if cand.exists():
                    src = str(cand)
                    content = cand.read_text(encoding="utf-8", errors="replace")[:2000]
                    m = re.search(r"class\s+(\w+)\s*\(", content)
                    cls = m.group(1) if m else (class_name or game_id[:1].upper() + game_id[1:])
                    logger.info(f"BFS: found game source at {src}, class={cls}")
                    return src, cls

    parts = game_id.split("-", 1)
    gid = parts[0]
    guid_suffix = parts[1] if len(parts) > 1 else ""

    # 1. Local data-dir match (mirrors the Kaggle layout under data/kaggle/...)
    if guid_suffix:
        candidates = [
            _LOCAL_ENV_DIR / gid / guid_suffix / f"{gid}.py",
        ]
    else:
        # Game id wasn't qualified with a guid - try every version directory.
        candidates = (
            list((_LOCAL_ENV_DIR / gid).glob("*/" + f"{gid}.py"))
            if (_LOCAL_ENV_DIR / gid).exists()
            else []
        )

    for cand in candidates:
        if cand.exists():
            src = str(cand)
            content = cand.read_text(encoding="utf-8", errors="replace")[:2000]
            m = re.search(r"class\s+(\w+)\s*\(", content)
            cls = m.group(1) if m else gid[:1].upper() + gid[1:]
            logger.info(f"BFS: found game source at {src}, class={cls}")
            return src, cls

    # 2. Kaggle-only paths (kept verbatim for parity with the upstream notebook)
    kaggle_path = (
        f"/kaggle/input/competitions/arc-prize-2026-arc-agi-3"
        f"/environment_files/{gid}/{guid_suffix}/{gid}.py"
    )
    if os.path.exists(kaggle_path):
        content = open(kaggle_path).read()[:2000]
        m = re.search(r"class\s+(\w+)\s*\(", content)
        cls = m.group(1) if m else gid[:1].upper() + gid[1:]
        return kaggle_path, cls

    for pattern in (
        f"/kaggle/input/**/{gid}.py",
        f"/tmp/**/{gid}.py",
        f"/kaggle/working/**/{gid}.py",
    ):
        matches = glob.glob(pattern, recursive=True)
        if matches:
            src = matches[0]
            content = open(src).read()[:2000]
            m = re.search(r"class\s+(\w+)\s*\(", content)
            cls = m.group(1) if m else gid[:1].upper() + gid[1:]
            return src, cls

    logger.warning(f"BFS: game source not found for {game_id}")
    return None, gid[:1].upper() + gid[1:]


# ---------------------------------------------------------------------------
# Lazy upstream import (so missing torch fails gracefully)
# ---------------------------------------------------------------------------

_MyAgent = None  # populated on first ForgeAgent() construction
_import_error: BaseException | None = None


def _ensure_upstream_loaded() -> None:
    global _MyAgent, _import_error
    if _MyAgent is not None or _import_error is not None:
        return
    try:
        from . import _forge_v19 as _src  # imports torch, numpy, arcengine
    except ImportError as e:
        _import_error = e
        return

    # Monkey-patch the local-aware finder over the upstream one.
    _src.find_game_source_and_class = _find_game_source_and_class_local

    _MyAgent = _src.MyAgent


# ---------------------------------------------------------------------------
# ForgeAgent adapter
# ---------------------------------------------------------------------------


class ForgeAgent:
    """Adapter that bridges the upstream MyAgent (FORGE v19) to local_runner."""

    name = "forge"

    def __init__(
        self,
        seed: int = 0,
        arc_env: Any | None = None,
        game_id: str = "",
        **kwargs: Any,
    ) -> None:
        del seed  # MyAgent reseeds itself per game_id + wallclock
        del kwargs

        _ensure_upstream_loaded()
        if _MyAgent is None:
            raise ImportError(
                "ForgeAgent depends on PyTorch (torch / torch.nn / torch.optim) "
                "and the bundled arc-agi wheels, but the import failed:\n"
                f"    {type(_import_error).__name__}: {_import_error}\n"
                "Install torch into the venv:\n"
                "    uv pip install --python .venv/bin/python torch \\\n"
                "        --index-url https://download.pytorch.org/whl/cpu"
            )

        # Try to derive game_id from arc_env if not supplied (the SDK env's
        # observation_space carries it).
        if not game_id and arc_env is not None:
            obs = getattr(arc_env, "observation_space", None)
            if obs is not None:
                game_id = getattr(obs, "game_id", "") or game_id
        self._inner = _MyAgent(
            card_id="",
            game_id=game_id,
            agent_name="forge",
            ROOT_URL="",
            record=False,
            arc_env=arc_env,
        )

    def choose_action(self, frame: Any):
        # The upstream MyAgent expects (frames_history, latest_frame). For our
        # purposes the inner class maintains its own frame buffer via
        # append_frame(); we just give it the latest frame as both args.
        self._inner.append_frame(frame)
        action = self._inner.choose_action(self._inner.frames, frame)
        # Tick the inner counter so logging / MAX_ACTIONS work.
        self._inner.action_counter += 1
        return action

    def is_done(self, frame: Any) -> bool:
        return bool(self._inner.is_done(self._inner.frames, frame))


__all__ = ["ForgeAgent"]
