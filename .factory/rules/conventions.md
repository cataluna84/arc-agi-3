# Conventions

## Environment

- **OS**: Linux 6.6 (WSL2). **Python**: 3.12.3 — use `python3` directly. `uv` not installed system-wide; install per-task with `pip install uv` or use plain `pip`.
- No `git` repo initialized in the project yet; no `rg`/`ffmpeg`/`jupyter` system-wide.
- Prefer `Read`/`Grep`/`LS`/`Glob` over shell `cat`/`grep`/`find`.

## Filesystem layout

- `agents/`             — agent classes (one per file)
- `experiments/`        — `expNNN_<slug>/` per Kaggle daily slot + `EXPERIMENTS.md` and `local_runner.py`
- `scripts/`            — local dev helpers (`download_kaggle_data.py`, `install_arc_agi_sdk.py`)
- `research/`           — paper notes, code dives, deep-research dossiers
- `documentation/kaggle/`
  - `*.mhtml`            — official Kaggle pages (overview/data/code/discussion/leaderboard)
  - `*_extracted.txt`    — extracted plain text (do not commit raw HTML)
- `runs/`               — local smoke-run JSON outputs (gitignored)
- `data/kaggle/arc-prize-2026-arc-agi-3/`
                        — competition data + wheels (gitignored; populate via `scripts/download_kaggle_data.py`)
- `.env` / `.env.example` — Kaggle API creds (real `.env` gitignored)
- `requirements.txt`    — local dev deps (kaggle, kagglehub, python-dotenv)
- `.factory/`           — Factory.ai canonical memory:
  - `plan.md`            — phased D0-D20 daily roadmap
  - `memories.md`        — append-only project decisions + history log
  - `verify.md`          — smoke-test commands (V1-V9)
  - `rules/`             — split-by-topic project conventions (this file)
  - `droids/commands/hooks/skills/` — Factory subagent / extension folders

## Code style

- **No emojis** in code or docs unless explicitly asked.
- Match the existing style in surrounding files; never introduce new dependencies without justification.
- Match Factory CLI norms: short replies, no unrequested changes, prefer `Read`/`Grep`/`LS` over shell.
- Comments only when absolutely necessary; never decorative.

## Memory + AGENTS.md update policy

- Treat `.factory/memories.md` as **append-only** — new dated section at the top, do not rewrite history.
- Treat `.factory/rules/*.md` as **slowly-evolving facts** — update only when something is genuinely wrong/new.
- Treat root `AGENTS.md` as the **single agent-readable briefing packet** — keep ≤150 lines, link out to the rules files for detail.

## Score-delta convention

- All "Δ over baseline" deltas are measured against **0.19** (vanilla Ash fork, rank 398). Not 0.25 nor 0.42.
