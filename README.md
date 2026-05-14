# arc-agi-3

[![CI](https://github.com/cataluna84/arc-agi-3/actions/workflows/ci.yml/badge.svg)](https://github.com/cataluna84/arc-agi-3/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

> A public lab notebook for the [ARC Prize 2026 - ARC-AGI-3 Kaggle Code
> Competition][comp] ($850K prize pool, ends 2026-11-02). One person's
> daily attack on the leaderboard, with the agents, experiments, tooling,
> and decision log produced along the way.

[comp]: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3

**Current state** (D16, 2026-05-14): 8 scored submissions; LB trace
**0.19 / 0.00 / 0.24 / 0.10 / 0.21 / 0.00 / 0.17 / 0.12**. Best to
date is D3's variance probe at 0.24. D9 confirmed the silent-crash
hypothesis on Goose CNN v1 (0.00 → v2 0.17 with `enable_gpu=false`
plus defensive `try/except`). D10+D11 ported the dolphin-in-a-coma
frame-segmentation algorithm (arXiv:2512.24156, MIT) into
`agents/frame_segmenter.py` and wired it as the ACTION6 click-coord
prior in `agents/trigger_bfs_agent.py`; D15's wired-up submission
landed at **LB 0.12**, so the segmenter prior alone is a marginal
+0.02 over trigger-bfs v0 but below Goose v2 and master_v7. Baseline
anchor remains **LB 0.19** (vanilla
fork of an upstream public Kaggle notebook implementing FORGE v19;
see [NOTICE](NOTICE) for upstream credit + paper attributions). All
"delta vs baseline" deltas are measured against this **0.19** number.
See [`.factory/memories.md`](.factory/memories.md) for the running
narrative and [CHANGELOG.md](CHANGELOG.md) for user-visible changes.

---

## Table of contents

1. [What this repo is (and isn't)](#what-this-repo-is-and-isnt)
2. [Quickstart](#quickstart)
3. [Repository layout](#repository-layout)
4. [Daily workflow](#daily-workflow)
5. [Agents in the box](#agents-in-the-box)
6. [Kaggle integration](#kaggle-integration)
7. [Phase plan and target scores](#phase-plan-and-target-scores)
8. [Tooling and code quality](#tooling-and-code-quality)
9. [Contributing](#contributing)
10. [Citing](#citing)
11. [License](#license)

---

## What this repo is (and isn't)

**This repo is**:

- An **agent zoo + harness** for ARC-AGI-3: a uniform `choose_action(frame) -> GameAction`
  contract that any agent (random, search-based, neural, LLM-driven) can
  conform to.
- An **offline smoke runner** (`experiments/local_runner.py`) that lets
  you exercise an agent end-to-end *without* burning a Kaggle daily
  submission slot, using either the real `arc-agi` SDK or a tiny
  built-in mock environment.
- A **Kaggle automation toolkit** (`scripts/`): downloads the
  competition data, installs the SDK from offline wheels, and helps
  push / track / submit Kaggle kernels.
- A **research log**: every Kaggle submission, every gotcha, every
  decision is captured in `.factory/memories.md` (append-only) and
  surfaced publicly in [`CHANGELOG.md`](CHANGELOG.md).

**This repo is NOT**:

- A finished competition entry (we're still climbing the leaderboard).
- A model zoo: trained model weights are NOT included; large weights
  are downloaded at runtime from HuggingFace and bundled as private
  Kaggle Datasets.
- A drop-in `pip install` library: it's a workspace, not a published
  package. `pyproject.toml` has `package = false` under `[tool.uv]`.

## Quickstart

```bash
# 1. Get the source
git clone https://github.com/cataluna84/arc-agi-3.git
cd arc-agi-3

# 2. Install uv (https://docs.astral.sh/uv/) if you don't already have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create the venv and install deps (including dev: ruff, pre-commit, pytest)
uv venv --python 3.12
uv sync --all-groups

# 4. Install the pre-commit hooks (ruff + secret-leak guards on every commit)
uv run pre-commit install

# 5. Run the offline smoke (no GPU, no Kaggle creds, no model weights)
uv run python scripts/qwen_agent_smoke_local.py
# expect: ALL OK (22/22 checks)

# 6. Run the offline runner against the mock environment
uv run python experiments/local_runner.py \
    --agent agents.random_agent:RandomAgent \
    --games ls20-mock --max-actions 50

# 7. (optional) Configure Kaggle access
cp .env.example .env  # paste KAGGLE_USERNAME and KAGGLE_KEY
uv run python scripts/download_kaggle_data.py
uv run python scripts/install_arc_agi_sdk.py
```

If steps 1-6 work, your dev environment is ready. Step 7 is only needed
if you want to push kernels or run agents against the real ARC games.

## Repository layout

```
arc-agi-3/
|-- agents/                         # agent classes (one per file)
|   |-- __init__.py                 # Agent Protocol + GameAction/GameState fallbacks + MockFrame
|   |-- agent.py                    # Local stub of upstream agents.agent.Agent
|   |-- random_agent.py             # baseline 1: uniform-random over available actions
|   |-- greedy_explore_agent.py     # baseline 2: empirical change-rate epsilon-greedy
|   |-- forge_agent.py              # adapter for the verbatim FORGE port
|   |-- _forge_v19.py                # VENDORED: bit-for-bit FORGE v19 cell #1 - do not edit
|   |-- qwen_agent.py               # Qwen3.6-35B-A3B vision-language agent (exp004)
|   |-- trigger_bfs_agent.py        # trigger-aware BFS over state-hash graph (exp005)
|   |-- goose_agent.py              # Stochastic-Goose-style CNN with conv coord head (exp007)
|   |-- state_graph.py              # state-hash graph + trigger scoring used by trigger_bfs
|   `-- frame_segmenter.py          # stateless port of dolphin-in-a-coma frame-segmentation (exp008)
|-- experiments/
|   |-- EXPERIMENTS.md              # tracker of all expNNN folders
|   |-- SPEC_4WEEKS.md              # 4-week per-day SPEC with kernel slugs + targets + outcomes
|   |-- local_runner.py             # offline smoke harness (mock + arc-agi SDK fallback)
|   |-- exp001_baseline_forge/      # LB 0.19 anchor (vanilla FORGE v19 fork)
|   |-- exp002_forge_variance_probe/ # variance probe for the FORGE baseline (LB 0.24)
|   |-- exp003_baseline_just_explore/   # orthogonal reference baseline
|   |-- exp004_qwen_agent/          # Qwen3.6-35B-A3B vision-language agent (LB 0.00; RTX probe load works)
|   |-- exp005_trigger_aware_bfs/   # trigger-aware BFS v0 (LB 0.10)
|   |-- exp007_goose_cnn/           # Goose CNN v1 (0.00) -> v2 (0.17)
|   |-- exp008_trigger_bfs_seg/     # trigger_bfs + frame-segmenter (LB 0.12)
|   |-- kernel_h100_probe/          # sanity probe of Kaggle's H100 image
|   `-- kernel_qwen_bridge_probe/   # probes HF -> Kaggle bridge feasibility
|-- scripts/
|   |-- download_kaggle_data.py     # pulls competition data via Kaggle API (KGAT_-aware)
|   |-- install_arc_agi_sdk.py      # offline install of arc-agi + arcengine wheels
|   |-- qwen_agent_smoke_local.py   # pure-Python QwenAgent smoke (no GPU)
|   |-- resubmit_forge.sh           # variance-probe helper (Track A in RUNBOOK_D2)
|   `-- README.md                   # script catalogue
|-- research/
|   |-- 01_landscape_review.md      # LB landscape + top public notebooks + attack ranking
|   |-- 02_exa_deep_research_2026-04-29.md   # Exa Deep Researcher Pro report
|   |-- 03_strategy_and_kaggle_compute_2026-04-29.md
|   `-- ash_notebook/               # captured upstream notebook + extracted text
|-- documentation/
|   `-- kaggle/                     # MHTML mirrors of comp pages + extracted text
|-- .factory/                       # canonical project memory (Factory.ai convention)
|   |-- plan.md                     # phased D0..D20+ daily Kaggle roadmap
|   |-- memories.md                 # append-only project decisions and gotchas log
|   |-- verify.md                   # V1..V9 pre-Kaggle-submission checklist
|   `-- rules/                      # split-by-topic project conventions (7 files)
|-- .github/
|   |-- ISSUE_TEMPLATE/             # bug, feature, experiment proposal templates
|   |-- workflows/ci.yml            # ruff + pytest + smoke runner CI
|   |-- dependabot.yml              # weekly Python + Actions updates
|   `-- PULL_REQUEST_TEMPLATE.md
|-- AGENTS.md                       # repo briefing (also doubles as Factory.ai context)
|-- CONTRIBUTING.md                 # how to contribute (read me before opening a PR)
|-- CODE_OF_CONDUCT.md              # Contributor Covenant 2.1 (link)
|-- CHANGELOG.md                    # Keep-a-Changelog dated entries
|-- CITATION.cff                    # Citation File Format (academic)
|-- LICENSE                         # Apache 2.0
|-- NOTICE                          # upstream attribution
|-- pyproject.toml                  # deps + ruff + pytest config
|-- uv.lock                         # reproducible lockfile
`-- .pre-commit-config.yaml         # ruff + hygiene + actionlint + gitleaks
```

Folders intentionally **not** in version control (gitignored):
`.venv/`, `data/`, `runs/`, `environment_files/`, `__pycache__/`,
`.env`, model weights of any kind.

## Daily workflow

The rhythm of the project is one Kaggle daily slot per day:

1. Open [`.factory/plan.md`](.factory/plan.md) and pick today's
   experiment (or pick from `experiments/EXPERIMENTS.md`).
2. Implement under `experiments/expNNN_<slug>/` and / or
   `agents/<my_agent>.py`.
3. Smoke-test locally:
   ```bash
   uv run python experiments/local_runner.py \
       --agent agents.<my_agent>:<MyAgent> \
       --games ls20-mock --max-actions 200 --seed 0
   ```
4. (Optional, free) push to a Kaggle dev kernel for runtime parity:
   ```bash
   uv run kaggle kernels push -p experiments/expNNN_<slug>/dev_kernel
   ```
5. Submit on Kaggle (this **burns the daily slot**; note the CLI
   subcommand is `submit`, not `submit-code` — see gotcha #15):
   ```bash
   uv run kaggle competitions submit arc-prize-2026-arc-agi-3 \
       -k cataluna84/<comp-kernel> -v <N> \
       -f submission.parquet -m "expNNN: <one-liner>"
   ```
6. After the LB result lands, append a dated section to the **top** of
   [`.factory/memories.md`](.factory/memories.md) with the score, delta
   vs 0.19, per-game notes, and the next-step decision.
7. Update [`CHANGELOG.md`](CHANGELOG.md) `[Unreleased]` with the
   user-visible change.

For tomorrow's specific runbook, see
[`experiments/exp004_qwen_agent/RUNBOOK_D2.md`](experiments/exp004_qwen_agent/RUNBOOK_D2.md).

## Agents in the box

| Agent | File | Approach | Status |
| --- | --- | --- | --- |
| `RandomAgent` | `agents/random_agent.py` | uniform over `available_actions`; ACTION6 click is uniform-random | working |
| `GreedyExploreAgent` | `agents/greedy_explore_agent.py` | epsilon-greedy on per-action empirical frame-change rate | working |
| `ForgeAgent` | `agents/forge_agent.py` | adapter around verbatim FORGE v19 (BFS + ForgeNet CNN) | LB 0.19 baseline; MASTER v7 remix reached 0.21 |
| `QwenAgent` | `agents/qwen_agent.py` | vision-language MoE: image + hex grid + history -> ACTION (`Qwen3.6-35B-A3B` BF16); RTX 6000 Phase-0 probe now loads/generates offline | LB 0.00 (D2); LLM-as-direct-policy is structurally worse than random unless constrained (see gotcha #18) |
| `TriggerBFSAgent` | `agents/trigger_bfs_agent.py` | trigger-aware BFS over state-hash graph; ACTION6 click coords come from `frame_segmenter` 5-tier saliency | LB 0.10 (D4 v0); segmenter prior exp008 = 0.12 |
| `GooseAgent` | `agents/goose_agent.py` | Stochastic-Goose-style 4-layer CNN with conv coord head, BCE on `frame_changed` over a 200K hash-dedup buffer | LB 0.00 v1 (D6) → 0.17 v2 (D9) after `enable_gpu=false` + defensive `try/except` |
| `frame_segmenter` (lib) | `agents/frame_segmenter.py` | stateless port of the dolphin-in-a-coma frame-segmentation algorithm (arXiv:2512.24156, MIT): per-color connected components, 5-tier saliency, status-bar detection | used by `TriggerBFSAgent` for ACTION6 click coords |

The agent contract is documented at the top of
[`agents/__init__.py`](agents/__init__.py). Adding a new agent is
described in [`CONTRIBUTING.md`](CONTRIBUTING.md#adding-a-new-agent).

## Kaggle integration

- `scripts/download_kaggle_data.py` reads `.env` for
  `KAGGLE_USERNAME` / `KAGGLE_KEY`, auto-detects KGAT_-format tokens
  and switches to Bearer auth, then downloads the competition data
  + bundled wheels into `data/kaggle/arc-prize-2026-arc-agi-3/`.
- `scripts/install_arc_agi_sdk.py` installs `arc-agi` + `arcengine`
  from those wheels into the venv (offline-friendly).
- `experiments/local_runner.py --use-sdk` uses the real ARC environment;
  without `--use-sdk`, it uses a tiny built-in mock.

The hardware constraints we've verified on Kaggle's H100 image
(`gcr.io/kaggle-gpu-images/python`) are documented in
[`.factory/memories.md`](.factory/memories.md):

- 1 x H100 80 GB HBM3, sm_90 Hopper, FP8 native.
- 31.4 GB system RAM (no large CPU offload viable).
- `/kaggle/working` only 19.5 GB; `/tmp` 1.2 TB free.
- `transformers 5.0.0`, `accelerate 1.12.0`, `torchao 0.10.0`,
  `triton 3.6.0` pre-installed; vLLM / SGLang / flash-attn NOT.

## Phase plan and target scores

| Phase | Days | Target | Δ vs 0.19 | Approach |
| --- | --- | --- | --- | --- |
| 0 - Foundation | D0..D4 | 0.19-0.30 | +0.00..+0.11 | Anchor on FORGE 0.19, variance probe, local runner + agent zoo |
| 1 - Core search + learning | D5..D7 | 0.30-0.35 | +0.11..+0.16 | Trigger-aware BFS, StochasticGoose CNN, hybrid search-and-learn |
| 2 - Object-centric + WM | D8..D12 | 0.40-0.50 | +0.21..+0.31 | Segmentation+click, MCTS+CNN prior, DreamerV3-lite |
| 3 - TTT, DSL, slot WM | D13..D16 | 0.50-0.55 | +0.31..+0.36 | Test-Time Training, DSL synthesis, slot-attention world model |
| 4 - Composition / ensemble | D17..D20+ | 0.60-0.70+ | +0.41+ | Per-game dispatcher, offline pretraining, LLM orchestrator |

Full breakdown in [`.factory/plan.md`](.factory/plan.md).

## Tooling and code quality

This project uses the **state-of-the-art** Python tooling stack (2026):

| Tool | Role | Config |
| --- | --- | --- |
| [uv][uv] | dep + venv manager | `pyproject.toml` + `uv.lock` |
| [Ruff][ruff] | lint + format + import sort | `[tool.ruff]` in `pyproject.toml` |
| [pre-commit][pc] | git hooks | `.pre-commit-config.yaml` |
| [pytest][pt] | test runner | `[tool.pytest.ini_options]` in `pyproject.toml` |
| [actionlint][al] | YAML CI lint | as a pre-commit hook |
| [gitleaks][gl] | secret leak detection | as a pre-commit hook |
| GitHub Actions | CI | `.github/workflows/ci.yml` |
| Dependabot | dep updates | `.github/dependabot.yml` |

[uv]: https://docs.astral.sh/uv/
[ruff]: https://docs.astral.sh/ruff/
[pc]: https://pre-commit.com/
[pt]: https://docs.pytest.org/
[al]: https://github.com/rhysd/actionlint
[gl]: https://github.com/gitleaks/gitleaks

The Ruff ruleset is intentionally broad - `E`, `W`, `F`, `I`, `B`,
`C4`, `UP`, `RUF`, `SIM`, `TID`, `PTH`, `PERF`, `A`, `ARG`, `S`, `N`,
`RET`, `TCH`, `ICN`, `ISC` - with pragmatic per-file ignores for
intentional patterns (e.g. `/tmp` paths on Kaggle, mirrors of upstream
APIs, etc.). The vendored FORGE port (`agents/_forge_v19.py`) is
excluded from linting since it is a verbatim copy.

Run all the checks locally exactly as CI runs them:

```bash
uv run pre-commit run --all-files   # ruff + secret-leak + JSON/YAML/TOML
uv run pytest                        # tests + smoke files
uv run python scripts/qwen_agent_smoke_local.py
uv run python experiments/local_runner.py \
    --agent agents.random_agent:RandomAgent --games ls20-mock --max-actions 30
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). TL;DR: fork, branch off `main`,
run all checks locally, open a PR using the template. By submitting a
contribution you agree to license it under Apache 2.0.

## Citing

If this work informs your research, please cite using the
[`CITATION.cff`](CITATION.cff) file or the BibTeX below:

```bibtex
@software{bhaskar_arc_agi_3_2026,
  author  = {Mayank Bhaskar},
  title   = {arc-agi-3: a Kaggle ARC Prize 2026 lab notebook},
  year    = {2026},
  url     = {https://github.com/cataluna84/arc-agi-3},
  license = {Apache-2.0}
}
```

## License

[Apache License 2.0](LICENSE) - see also [NOTICE](NOTICE) for upstream
attribution.

Copyright 2026 Mayank Bhaskar (cataluna84).
