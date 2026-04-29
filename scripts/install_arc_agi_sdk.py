"""install_arc_agi_sdk.py — Offline-install the arc-agi SDK from downloaded wheels.

Run this AFTER `scripts/download_kaggle_data.py` so we can use:
    .venv/bin/python experiments/local_runner.py --use-sdk ...

Default: installs into the project's `.venv` via `uv pip install` (no internet).
Falls back to `python -m pip` if uv is unavailable.

Usage:
    .venv/bin/python scripts/install_arc_agi_sdk.py
    .venv/bin/python scripts/install_arc_agi_sdk.py --packages arc-agi arcengine
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WHEELS_DIR = REPO_ROOT / "data" / "kaggle" / "arc-prize-2026-arc-agi-3" / "arc_agi_3_wheels"
VENV_DIR = REPO_ROOT / ".venv"
DEFAULT_PACKAGES = ("arc-agi", "arcengine")


def _build_install_cmd(packages: list[str]) -> tuple[list[str], str]:
    """Choose uv pip if available; fall back to python -m pip."""
    uv_bin = shutil.which("uv")
    if uv_bin and VENV_DIR.exists():
        cmd = [
            uv_bin,
            "pip",
            "install",
            "--python",
            str(VENV_DIR / "bin" / "python"),
            "--no-index",
            "--find-links",
            str(WHEELS_DIR),
            *packages,
        ]
        return cmd, "uv"
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(WHEELS_DIR),
        *packages,
    ]
    return cmd, "pip"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--packages",
        nargs="+",
        default=list(DEFAULT_PACKAGES),
        help=f"Package names to install (default: {' '.join(DEFAULT_PACKAGES)}).",
    )
    args = parser.parse_args()

    if not WHEELS_DIR.exists():
        print(f"[error] Wheels dir not found: {WHEELS_DIR}", file=sys.stderr)
        print("        Run scripts/download_kaggle_data.py first.", file=sys.stderr)
        return 1

    wheels = sorted(WHEELS_DIR.glob("*.whl"))
    if not wheels:
        print(f"[error] No .whl files in {WHEELS_DIR}", file=sys.stderr)
        return 2

    print(f"[install] Found {len(wheels)} wheel(s) in {WHEELS_DIR.name}/")
    print(f"[install] Target packages: {' '.join(args.packages)}")

    cmd, mode = _build_install_cmd(args.packages)
    print(f"[install] Using: {mode}")
    print(f"[install] Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    print("\n[install] Verifying imports:")
    py = str(VENV_DIR / "bin" / "python") if VENV_DIR.exists() else sys.executable
    for pkg, mod in [("arc-agi", "arc_agi"), ("arcengine", "arcengine")]:
        if pkg not in args.packages:
            continue
        code = (
            f"import {mod}; print('{mod:10s} version:', getattr({mod}, '__version__', 'unknown'))"
        )
        try:
            subprocess.check_call([py, "-c", code])
        except subprocess.CalledProcessError as e:
            print(f"[warn] could not import {mod}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
