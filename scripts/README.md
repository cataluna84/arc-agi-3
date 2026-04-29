# scripts/

Helper scripts for local Kaggle data setup. None of these run during a Kaggle notebook submission — they only exist to mirror the competition data + wheels onto this machine for offline experimentation via `experiments/local_runner.py --use-sdk`.

## One-time setup

### 1. Get a Kaggle API token

1. Go to https://www.kaggle.com/settings/account
2. Scroll to **API** → click **Create New API Token**
3. A `kaggle.json` file downloads. Open it and copy the `username` + `key`.

### 2. Accept the competition rules

You must visit https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules and click **I Understand and Accept** before any download will work.

### 3. Configure local credentials

```bash
cp .env.example .env
# Edit .env:
#   KAGGLE_USERNAME=<your_username>
#   KAGGLE_KEY=<your_key>
```

The `.env` file is gitignored. Both `kagglehub` and the `kaggle` CLI read these env vars (set by `python-dotenv` when the script starts).

### 4. Install dev dependencies (via uv)

```bash
uv venv --python 3.12          # one-time: create .venv/
uv sync                        # installs kaggle, kagglehub, python-dotenv (only)
```

That's intentionally the minimum needed to run the download script. Everything else (arc-agi SDK, numpy, pillow, etc.) ships as wheels inside the Kaggle competition data and gets installed by `scripts/install_arc_agi_sdk.py` after the download completes.

## Usage

### Download competition data

```bash
.venv/bin/python scripts/download_kaggle_data.py                   # auto: kagglehub then CLI fallback
.venv/bin/python scripts/download_kaggle_data.py --tool cli        # force kaggle CLI
.venv/bin/python scripts/download_kaggle_data.py --tool hub        # force kagglehub
.venv/bin/python scripts/download_kaggle_data.py --force           # re-download even if data exists
.venv/bin/python scripts/download_kaggle_data.py --keep-zips       # keep .zip archives after extraction
```

(Or activate the venv first with `source .venv/bin/activate` and drop the `.venv/bin/` prefix.)

The script is **idempotent** — re-running it on a populated `data/kaggle/arc-prize-2026-arc-agi-3/` is a no-op (just runs the verification step).

Expected output layout (after success):

```
data/kaggle/arc-prize-2026-arc-agi-3/
├── environment_files/        # ~25 public game files (one folder per game)
├── arc_agi_3_wheels/         # offline pip wheels for the arc-agi SDK
└── ARC-AGI-3-Agents/         # (optional) cloned harness
```

### Install the arc-agi SDK locally

After the download succeeds:

```bash
python3 scripts/install_arc_agi_sdk.py
```

This runs `pip install --no-index --find-links=data/kaggle/.../arc_agi_3_wheels arc-agi` (the same command the Kaggle notebook uses), then verifies the import.

After this you can do:

```bash
python3 experiments/local_runner.py \
    --agent agents.random_agent:RandomAgent \
    --use-sdk \
    --games <real-game-id-from-environment_files>
```

## Why both `kaggle` CLI and `kagglehub`?

- **`kagglehub`**: pythonic, version-pinned, integrates inside Python scripts. Used by `download_kaggle_data.py` as primary.
- **`kaggle` CLI**: battle-tested, full feature coverage. Used as fallback and for ad-hoc commands like:
  ```bash
  kaggle competitions list
  kaggle competitions submissions -c arc-prize-2026-arc-agi-3
  ```

Both share the same `.env` credentials.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `403 Forbidden` | You haven't accepted the competition rules. Visit the rules page and click accept. |
| `KAGGLE_USERNAME missing` | Ensure `.env` exists, edited, in repo root. |
| `No .whl files in arc_agi_3_wheels/` | The competition data layout changed. Inspect `data/kaggle/.../` manually. |
| `403 Quota exceeded` | Kaggle API limits ~30 calls/hour. Wait an hour. |
