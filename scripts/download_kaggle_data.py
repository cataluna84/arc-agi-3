"""download_kaggle_data.py — Download ARC Prize 2026 competition data.

Idempotent: skips download if `data/kaggle/arc-prize-2026-arc-agi-3/` already
contains files (use --force to redownload).

Usage:
    python3 scripts/download_kaggle_data.py
    python3 scripts/download_kaggle_data.py --force
    python3 scripts/download_kaggle_data.py --tool cli       # use kaggle CLI
    python3 scripts/download_kaggle_data.py --tool hub       # use kagglehub
    python3 scripts/download_kaggle_data.py --tool auto      # try hub then cli (default)
    python3 scripts/download_kaggle_data.py --keep-zips      # keep downloaded zip files

Authentication: reads KAGGLE_USERNAME and KAGGLE_KEY from `.env` at project root.
Get credentials at https://www.kaggle.com/settings/account → Create New API Token.
You must also accept the competition rules first:
    https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "kaggle" / "arc-prize-2026-arc-agi-3"
COMP_SLUG = "arc-prize-2026-arc-agi-3"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def load_env(verbose: bool = True) -> None:
    """Load KAGGLE_USERNAME / KAGGLE_KEY from .env at project root."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        print(
            "[error] python-dotenv not installed. Run: python3 -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        print(
            f"[error] {env_path} not found.\n"
            f"        Copy .env.example to .env and fill in your Kaggle creds.",
            file=sys.stderr,
        )
        sys.exit(2)

    load_dotenv(env_path)

    user = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    if not user or user == "your_kaggle_username_here":
        print("[error] KAGGLE_USERNAME missing or unedited in .env", file=sys.stderr)
        sys.exit(3)
    if not key or key == "your_kaggle_api_key_here":
        print("[error] KAGGLE_KEY missing or unedited in .env", file=sys.stderr)
        sys.exit(3)

    # New Kaggle "KGAT_" tokens require Bearer auth, which kagglesdk only
    # enables when KAGGLE_API_TOKEN is set (kagglesdk/kaggle_env.py:91).
    # Without this, the kaggle CLI / kagglehub fall back to HTTP Basic Auth,
    # which the new API endpoints reject with 401 Unauthenticated.
    if key.startswith("KGAT_") and not os.environ.get("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_API_TOKEN"] = key
        if verbose:
            print("[ok] Detected KGAT_ token; setting KAGGLE_API_TOKEN for Bearer auth.")

    if verbose:
        print(f"[ok] Loaded Kaggle credentials for user: {user}")


# ---------------------------------------------------------------------------
# Downloaders
# ---------------------------------------------------------------------------


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _data_dir_has_payload() -> bool:
    """True if DATA_DIR has anything beyond .gitkeep / hidden files."""
    if not DATA_DIR.exists():
        return False
    for p in DATA_DIR.iterdir():
        if p.name.startswith("."):
            continue
        return True
    return False


def download_with_kagglehub() -> Path:
    """Use kagglehub to fetch the competition; mirror into DATA_DIR."""
    try:
        import kagglehub
    except ImportError:
        print(
            "[error] kagglehub not installed. Run: python3 -m pip install kagglehub",
            file=sys.stderr,
        )
        sys.exit(4)

    print(f"[kagglehub] Downloading competition: {COMP_SLUG}")
    cache_path_str = kagglehub.competition_download(COMP_SLUG)
    cache_path = Path(cache_path_str)
    print(f"[kagglehub] Cache: {cache_path}")

    _ensure_data_dir()
    if cache_path.is_dir():
        for item in cache_path.iterdir():
            dest = DATA_DIR / item.name
            if dest.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
    else:
        # Single-file fallback
        shutil.copy2(cache_path, DATA_DIR / cache_path.name)

    print(f"[kagglehub] Mirrored into: {DATA_DIR}")
    return DATA_DIR


def download_with_cli(keep_zips: bool = False) -> Path:
    """Use the kaggle CLI's Python API to fetch + extract the competition."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except ImportError:
        print(
            "[error] kaggle CLI not installed. Run: python3 -m pip install kaggle",
            file=sys.stderr,
        )
        sys.exit(5)

    _ensure_data_dir()
    api = KaggleApi()
    api.authenticate()  # uses KAGGLE_USERNAME / KAGGLE_KEY env vars

    print(f"[cli] Downloading competition: {COMP_SLUG}")
    api.competition_download_files(
        COMP_SLUG,
        path=str(DATA_DIR),
        force=False,
        quiet=False,
    )

    # Extract any zip archives Kaggle returned.
    zips = list(DATA_DIR.glob("*.zip"))
    for zf in zips:
        print(f"[cli] Extracting: {zf.name}")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(DATA_DIR)
        if not keep_zips:
            zf.unlink()
            print(f"[cli] Removed zip: {zf.name}")

    print(f"[cli] Downloaded + extracted to: {DATA_DIR}")
    return DATA_DIR


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_download() -> int:
    """Sanity-check the downloaded data; return number of missing expected paths."""
    expected_dirs = [
        "environment_files",  # public game files
        "arc_agi_3_wheels",  # offline pip wheels for arc-agi
    ]

    missing = []
    for name in expected_dirs:
        if not (DATA_DIR / name).exists():
            missing.append(name)

    if missing:
        print(f"\n[warn] Expected paths missing in {DATA_DIR}: {missing}")
        print("       (the competition data layout may have changed; check the Data tab on Kaggle)")
    else:
        print(f"\n[verify] All expected core paths present in {DATA_DIR}")

    print("\n[verify] Top-level contents:")
    if DATA_DIR.exists():
        for item in sorted(DATA_DIR.iterdir()):
            kind = "/" if item.is_dir() else ""
            try:
                size = (
                    f"  ({sum(p.stat().st_size for p in item.rglob('*') if p.is_file()) // 1024} KB)"
                    if item.is_dir()
                    else f"  ({item.stat().st_size // 1024} KB)"
                )
            except Exception:
                size = ""
            print(f"  {item.name}{kind}{size}")
    else:
        print(f"  (data dir does not exist: {DATA_DIR})")

    # Count games + wheels
    env_files = DATA_DIR / "environment_files"
    if env_files.is_dir():
        games = [p for p in env_files.iterdir() if p.is_dir()]
        print(f"\n[verify] {len(games)} game(s) in environment_files/")
    wheels = DATA_DIR / "arc_agi_3_wheels"
    if wheels.is_dir():
        whl = list(wheels.glob("*.whl"))
        print(f"[verify] {len(whl)} wheel(s) in arc_agi_3_wheels/")

    return len(missing)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--tool",
        choices=["hub", "cli", "auto"],
        default="auto",
        help="Which downloader to use (default: auto = hub then cli fallback).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if data/kaggle/.../ already has content.",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep .zip archives after extraction (CLI mode only).",
    )
    parser.add_argument(
        "--skip-env",
        action="store_true",
        help="Don't load .env (assume KAGGLE_USERNAME/KAGGLE_KEY already exported).",
    )
    args = parser.parse_args(argv)

    if not args.skip_env:
        load_env()

    if _data_dir_has_payload() and not args.force:
        print(f"[skip] {DATA_DIR} already has content. Use --force to redownload.")
        verify_download()
        return 0

    if args.tool == "hub":
        download_with_kagglehub()
    elif args.tool == "cli":
        download_with_cli(keep_zips=args.keep_zips)
    else:
        # auto: try kagglehub first, fall back to CLI on failure
        try:
            download_with_kagglehub()
        except SystemExit as e:
            if e.code in (0, None):
                pass  # success
            else:
                print("[fallback] kagglehub failed; trying kaggle CLI")
                download_with_cli(keep_zips=args.keep_zips)
        except Exception as e:
            print(f"[fallback] kagglehub raised: {e!r}; trying kaggle CLI")
            download_with_cli(keep_zips=args.keep_zips)

    n_missing = verify_download()
    return 0 if n_missing == 0 else 10


if __name__ == "__main__":
    raise SystemExit(main())
