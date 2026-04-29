# Contributing to arc-agi-3

Thanks for your interest! This repository is a public lab notebook for the
[ARC Prize 2026 - ARC-AGI-3 Kaggle Code Competition][comp]. It documents one
person's daily attack on the leaderboard, day by day, alongside the agents,
experiments, and tooling produced along the way. We welcome contributions of
ideas, code, replays, and documentation.

[comp]: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3

> **License**: Apache 2.0. By submitting a contribution you agree to
> license it under the same terms (Apache 2.0). See [LICENSE](LICENSE) and
> [NOTICE](NOTICE).

---

## Table of contents

1. [Code of conduct](#code-of-conduct)
2. [Ways to contribute](#ways-to-contribute)
3. [Quickstart for contributors](#quickstart-for-contributors)
4. [Local dev workflow](#local-dev-workflow)
5. [Submitting changes (PR process)](#submitting-changes-pr-process)
6. [Coding conventions](#coding-conventions)
7. [Adding a new agent](#adding-a-new-agent)
8. [Adding a new experiment](#adding-a-new-experiment)
9. [Reporting bugs and security issues](#reporting-bugs-and-security-issues)
10. [Releasing and changelog](#releasing-and-changelog)

---

## Code of conduct

This project follows the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md).
Be kind, be specific, assume good faith.

## Ways to contribute

| Type of contribution | Where it lives |
| --- | --- |
| Bug reports / questions | GitHub Issues (use the templates) |
| Feature ideas | GitHub Issues (`Feature request` template) |
| New experiments / agent designs | GitHub Issues (`Experiment proposal` template) -> PR under `experiments/expNNN_<slug>/` |
| Fixes to existing agents / runner / scripts | Pull Request |
| Documentation polish | Pull Request (no issue needed) |
| Replay submissions / Kaggle results | PR appending to `.factory/memories.md` and `CHANGELOG.md` |
| Security concerns (leaked credentials, etc.) | Email the maintainer directly - DO NOT open a public issue |

You don't need to be a Kaggle competitor to contribute. Improvements to the
local runner, ruff config, CI, docs, or research notes are all welcome.

## Quickstart for contributors

```bash
# 1. Fork on GitHub then clone your fork
git clone git@github.com:<your-handle>/arc-agi-3.git
cd arc-agi-3

# 2. Install uv if you don't already have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create the venv and install all dependencies (including dev)
uv venv --python 3.12
uv sync --all-groups

# 4. Install pre-commit hooks (runs ruff + secret-leak guards on every commit)
uv run pre-commit install

# 5. Run the offline smoke test (no GPU, no Kaggle access required)
uv run python scripts/qwen_agent_smoke_local.py
# expect: ALL OK (22/22 checks)

# 6. (optional) Run the full offline runner against the mock environment
uv run python experiments/local_runner.py \
    --agent agents.random_agent:RandomAgent \
    --games ls20-mock --max-actions 50

# 7. Run all checks (what CI will run)
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run pre-commit run --all-files
```

If everything is green, you have a working dev environment and can start
making changes.

## Local dev workflow

Required tools:

| Tool | Version | Why |
| --- | --- | --- |
| Python | >= 3.12 | matches Kaggle's H100 image (`python 3.12.12`) |
| [uv][] | >= 0.11 | dependency / venv manager (`pyproject.toml` + `uv.lock`) |
| Ruff | >= 0.14 | lint + format + import sort |
| pre-commit | >= 4.0 | git hooks |
| pytest | >= 8.3 | test runner |
| `kaggle` CLI | >= 1.8.0 | downloads competition data, pushes kernels |
| `gh` CLI | optional | for PR / issue automation |
| `git` | >= 2.40 | obvious |

[uv]: https://docs.astral.sh/uv/

For Kaggle integration (downloading competition data, running submissions),
you also need:

```bash
cp .env.example .env
# then paste your KAGGLE_USERNAME / KAGGLE_KEY into .env
```

KGAT_-format Kaggle tokens require Bearer auth - the helper scripts in
`scripts/` handle this automatically (search the codebase for
`KGAT_` for the rationale).

## Submitting changes (PR process)

1. **Open an issue first** for non-trivial changes, so we can confirm scope
   and avoid duplicate work. Tiny doc fixes can skip this.
2. **Branch off `main`**: `git checkout -b feat/<short-slug>`. Use prefixes
   `feat/`, `fix/`, `docs/`, `chore/`, `exp/`, `agent/`, `runner/`.
3. **Make your changes**. Each commit message should be short and explanatory.
   We loosely follow [Conventional Commits][conv]; not strictly required.
4. **Run all checks locally**:
   ```bash
   uv run pre-commit run --all-files   # ruff + secret-leak + JSON/YAML/TOML check
   uv run pytest                        # tests
   uv run python scripts/qwen_agent_smoke_local.py  # agent smoke
   ```
5. **Open a PR** against `main` using the [PR template](.github/PULL_REQUEST_TEMPLATE.md).
6. **CI runs automatically** (`.github/workflows/ci.yml`). Make sure it's green.
7. We squash-merge by default. Your commit message will be the PR title.

[conv]: https://www.conventionalcommits.org/

## Coding conventions

- **Python style**: enforced by `ruff` (see `[tool.ruff]` in `pyproject.toml`).
  The formatter handles most stylistic choices; you should not need to think
  about line-length, quote style, etc.
- **Type hints**: prefer `X | None` over `Optional[X]`, `list[T]` over
  `List[T]`, etc. (Ruff's `UP` rules will nudge you.)
- **Imports**: sorted automatically by Ruff (`I` rules). First-party
  packages: `agents`, `scripts`, `experiments`.
- **Docstrings**: every public function/class should have a one-line summary;
  longer rationale belongs in the file's module docstring or a `# ---`
  separator block. We don't (yet) enforce a docstring style.
- **Logging vs print**: scripts may `print()` (CLI affordance). Library code
  in `agents/` should use `logging`.
- **Comments**: explain *why*, not *what*. The `_ash_my_agent_v19.py` file
  is the only intentionally un-touched code -- do **not** edit it directly.
  Local adaptations belong in `agents/ash_agent.py`.
- **No unrequested deps**: if you need a new third-party package, mention
  it in the PR description with a one-line justification. Most heavy ML
  deps (torch, transformers, accelerate, ...) are pre-installed on
  Kaggle's H100 image; we only declare them as project deps if local
  dev requires them.
- **Files we never commit**: `.env`, model weights, datasets, `data/`,
  `runs/`, `.venv/` (all gitignored). Pre-commit's `gitleaks` hook will
  catch most accidental token leaks.

## Adding a new agent

Agents are vanilla Python classes that conform to a tiny duck-typed
contract documented at the top of [`agents/__init__.py`](agents/__init__.py):

```python
class MyAgent:
    name: str = "<short-id-for-logs>"

    def __init__(self, seed: int = 0) -> None: ...

    def choose_action(self, frame) -> GameAction: ...

    def is_done(self, frame) -> bool: ...   # optional
```

To add a new agent:

1. Create `agents/my_agent.py` containing the class.
2. (If your agent has heavy deps like torch / transformers) lazy-import
   them inside `__init__` or `choose_action`, not at module level. The
   smoke runner imports every agent during discovery and shouldn't crash
   on a missing GPU library.
3. Add a smoke test under `scripts/` if there's pure-Python logic worth
   exercising without the full SDK / GPU.
4. Run it locally:
   ```bash
   uv run python experiments/local_runner.py \
       --agent agents.my_agent:MyAgent \
       --games ls20-mock --max-actions 200
   ```
5. PR with a description that includes the local smoke output and any
   open trade-offs. Reference an existing agent (e.g. `RandomAgent`,
   `GreedyExploreAgent`, `QwenAgent`) for shape.

## Adding a new experiment

Each Kaggle daily-slot use OR each non-trivial agent prototype gets its
own folder under `experiments/expNNN_<slug>/`:

```
experiments/exp005_my_idea/
├── README.md            # hypothesis + DoD + runbook
├── kernel-metadata.json # if it's a Kaggle kernel
├── notebook.ipynb       # the kernel itself
└── notes.md             # optional running notes
```

Append a new entry to `experiments/EXPERIMENTS.md` with one line per
experiment. After a Kaggle submission lands, append a dated section to
the **top** of `.factory/memories.md` with the LB score and per-game
notes (this is the project's running narrative).

## Reporting bugs and security issues

Bugs in agent code, the runner, or scripts: open a GitHub Issue using
the bug-report template.

Security issues (accidentally-committed credentials, supply-chain
problems, etc.): email the maintainer directly at the address shown in
`pyproject.toml`'s `authors` field. Do **not** open a public issue for
secret-leak reports until the secret has been rotated.

## Releasing and changelog

This is a research repository, so we don't cut formal releases on a
schedule. We update `CHANGELOG.md` for each meaningful merge using the
[Keep a Changelog][kc] format (`Added`, `Changed`, `Fixed`, `Security`,
`Removed`, `Deprecated` sections under each dated entry).

When a Kaggle submission lands a new score, the PR body should mention
the LB number and link to the corresponding `.factory/memories.md`
section.

[kc]: https://keepachangelog.com/

---

## Getting unstuck

- `.factory/memories.md` (top section) is the running project narrative.
- `.factory/plan.md` lays out the daily roadmap (D0..D20+).
- `.factory/rules/*.md` contain split-by-topic conventions and
  hard-won facts.
- `.factory/verify.md` is the V1..V9 pre-Kaggle-submission checklist.

If something in the docs disagrees with the code, the code is right and
the docs are stale -- please open an issue or PR.
