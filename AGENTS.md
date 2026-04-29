<coding_guidelines>
# AGENTS.md - briefing for human contributors and AI coding assistants

This file is the project's high-density briefing packet. It serves a
**dual audience**:

1. **Human contributors** -- read this top-to-bottom before opening a PR.
   For the friendly long-form version, see [`README.md`](README.md) and
   [`CONTRIBUTING.md`](CONTRIBUTING.md).
2. **AI coding assistants** (Factory.ai droids, Claude Code, Cursor,
   Continue, Aider, Copilot Workspace, etc.) -- this file is the
   canonical "context-loader" they should read first. Many such tools
   look for a top-level `AGENTS.md` automatically; some also pick up
   `.factory/` if present.

> **Hard rule for AI assistants**: read `.factory/plan.md`,
> `.factory/memories.md` (top section), and `.factory/rules/gotchas.md`
> before making any change. Treat `.factory/memories.md` as append-only.

---

## Project at a glance

Compete on Kaggle's [ARC-AGI-3][comp] ($850K prize pool, ends
**2026-11-02**, 1 submission/day).

**Baseline LB anchor**: **0.19** (vanilla fork of *Ash's ARC-AGI-3 Agent*,
rank 398, submitted 2026-04-29). All "delta over baseline" deltas are
measured against this number, not 0.25 nor 0.42.

[comp]: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3

## Project layout

| Path | Purpose |
| --- | --- |
| `agents/` | Agent classes (one `.py` per agent class) |
| `experiments/` | Per-day experiment folders (`expNNN_<slug>/`) + `EXPERIMENTS.md` plan |
| `experiments/local_runner.py` | Offline smoke harness (mock + arc-agi SDK fallback) |
| `scripts/` | Local dev helpers (Kaggle data download, SDK installer, smoke runners) |
| `research/` | Landscape review + Exa Deep Research dossier + captured Ash notebook |
| `documentation/kaggle/` | MHTML mirrors of comp pages + extracted plain text |
| `runs/` | Local smoke-run JSON outputs (gitignored) |
| `data/kaggle/arc-prize-2026-arc-agi-3/` | Kaggle competition data + wheels (gitignored; populate via `scripts/download_kaggle_data.py`) |
| `.factory/` | Factory.ai canonical memory (read this) |
| `.github/` | Issue templates, PR template, CI workflow, dependabot |
| `.env` | Kaggle API creds (gitignored; copy from `.env.example`) |
| `pyproject.toml` | Deps (`uv`-managed) + ruff + pytest config |
| `.pre-commit-config.yaml` | Pre-commit hooks (ruff, hygiene, actionlint, gitleaks) |
| `.venv/` | uv-managed virtualenv (gitignored) |

## Build, lint, smoke

Python 3.12 in a uv-managed venv (`.venv/`). One-time setup:

```bash
uv venv --python 3.12
uv sync --all-groups                # installs core + dev (ruff, pre-commit, pytest)
uv run pre-commit install
cp .env.example .env                # paste KAGGLE_USERNAME / KAGGLE_KEY
```

Routine commands (each is what CI also runs):

```bash
uv run ruff check .                              # lint
uv run ruff format --check .                     # format check (no edits)
uv run ruff format .                             # apply format edits
uv run pytest                                    # tests
uv run pre-commit run --all-files                # all hooks: ruff + secret-leak + JSON/YAML/TOML
uv run python scripts/qwen_agent_smoke_local.py  # 22-check QwenAgent smoke (no GPU)
uv run python experiments/local_runner.py \
    --agent agents.random_agent:RandomAgent \
    --games ls20-mock --max-actions 200          # offline mock-env smoke
```

Kaggle integration:

```bash
uv run python scripts/download_kaggle_data.py        # fetch comp data + wheels
uv run python scripts/install_arc_agi_sdk.py         # install arc-agi + arcengine offline
uv run python experiments/local_runner.py \
    --agent agents.ash_agent:AshAgent \
    --use-sdk --games <real-id>                      # real SDK loop
```

Full pre-Kaggle-submission checklist: [`.factory/verify.md`](.factory/verify.md) (V1..V9).

## Memory & rules (read before any change)

| File | What's in it |
| --- | --- |
| `.factory/plan.md` | Phased D0..D20+ daily Kaggle roadmap with per-day DoD |
| `.factory/memories.md` | Append-only log of decisions, findings, gotchas, next steps |
| `.factory/verify.md` | V1..V9 pre-submission checklist |
| `.factory/rules/conventions.md` | Repo + Python conventions |
| `.factory/rules/arc-agi-3-mechanics.md` | Frame, action vocab, RHAE math, comp snapshot |
| `.factory/rules/leaderboard-anchors.md` | LB top + public-notebook score landmarks |
| `.factory/rules/approach-taxonomy.md` | A..G classes (FORGE/Goose/Just-Explore/Redpill/MCTS/DSL/LLM) |
| `.factory/rules/repos-and-tools.md` | Critical repos table |
| `.factory/rules/kaggle-submission.md` | Offline notebook skeleton + pre-flight cell |
| `.factory/rules/gotchas.md` | 10 hard-won bugs and traps |

Custom Factory extensions (empty stubs): `.factory/{droids,commands,hooks,skills}/`.

## Code conventions (one-liners)

- **Apache 2.0 licensed.** All contributions are accepted under the same
  license (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)).
- **Style is enforced by Ruff.** Run `uv run ruff format` and the
  formatter handles indentation, quotes, line breaks, etc. The Ruff lint
  ruleset is intentionally broad (`E,W,F,I,B,C4,UP,RUF,SIM,TID,PTH,PERF,A,ARG,S,N,RET,TCH,ICN,ISC`)
  with deliberate per-file ignores for project-specific intentional patterns.
- **No emojis** in code or docs unless explicitly asked.
- **Match surrounding code style.** Never add deps without justification.
- **Lazy-import heavy deps** (torch, transformers, PIL, ...) inside agents
  so the module imports cleanly without GPU libs (the smoke runner walks
  the `agents/` package).
- **Do not edit `agents/_ash_my_agent_v19.py`** -- it is a verbatim
  vendored copy of upstream cell #1. Local adaptations belong in
  `agents/ash_agent.py`.
- **Treat `.factory/memories.md` as append-only.** New dated section
  at the **top** every day.
- **Treat `.factory/rules/*.md` as slowly-evolving facts.**
- **Keep this `AGENTS.md` <= 200 lines.** Link to rules files for detail.

Full convention details: [`.factory/rules/conventions.md`](.factory/rules/conventions.md).

## Daily ritual

1. Pick today's experiment from `.factory/plan.md`.
2. Smoke locally via `experiments/local_runner.py` -- never burn a Kaggle slot on a smoke test.
3. Submit on Kaggle.
4. After the LB result, append a dated section to the **top** of
   `.factory/memories.md` with score, delta vs 0.19, per-game notes,
   and next-step decisions. Mirror user-visible changes to
   [`CHANGELOG.md`](CHANGELOG.md).
5. If a new gotcha surfaces, also update `.factory/rules/gotchas.md`.

## Hard rules (DO NOT violate)

- **1 Kaggle submission/day.** Always smoke-test locally first.
- **No internet during Kaggle competition eval.** All weights / binaries
  must be packaged as Kaggle Datasets and mounted via
  `dataset_sources` in `kernel-metadata.json`.
- **6h wall clock cap** on Kaggle reruns -- leave a 1h buffer, target
  <=5h actual compute.
- **Pre-commit must pass before push.** The `gitleaks` hook is
  the last line of defence against accidentally committing tokens.
- **Apache 2.0 obligations**: all eligible (prize-claimable) submissions
  must **open-source** their solution by the milestone deadlines
  (2026-06-30 / 2026-09-30).

## Hint for AI assistants

If you are an AI coding assistant editing this repository:

1. Read `.factory/memories.md` (top 100 lines) for the latest project state.
2. Read `.factory/plan.md` to understand the current daily target.
3. Read `.factory/rules/gotchas.md` to avoid known traps.
4. Before any code change, check that `uv run ruff check .` and the smoke
   tests are clean. After your change, re-run them.
5. Surface non-trivial decisions back to the human via a structured
   question (Factory.ai: use `AskUser`); do not silently make
   architecturally significant choices.
</coding_guidelines>
