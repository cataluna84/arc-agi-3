# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][kc] and this project loosely
adheres to [Semantic Versioning][sv]. Because this is an active research
repository tracking a daily Kaggle competition, expect frequent dated
entries under `[Unreleased]` rather than tagged version cuts.

[kc]: https://keepachangelog.com/en/1.1.0/
[sv]: https://semver.org/

---

## [Unreleased]

### Added - 2026-04-29 (D0/D1: bootstrap + exp004 scaffold)
- Project scaffold under Apache 2.0:
  `LICENSE`, `NOTICE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `CITATION.cff`, this `CHANGELOG.md`.
- Modern Python tooling: `pyproject.toml` with broad ruff ruleset
  (`E`, `W`, `F`, `I`, `B`, `C4`, `UP`, `RUF`, `SIM`, `TID`, `PTH`,
  `PERF`, `A`, `ARG`, `S`, `N`, `RET`, `TCH`, `ICN`, `ISC`),
  formatter, pytest config; `.pre-commit-config.yaml` with `ruff`,
  `ruff-format`, hygiene hooks (large-files, JSON/YAML/TOML check,
  detect-private-key, EOL fixers), `actionlint`, and `gitleaks`.
- Factory.ai canonical memory under `.factory/`: `plan.md` (D0..D20+
  daily roadmap), `memories.md` (append-only project log),
  `verify.md` (V1..V9 pre-submission checklist), and split-by-topic
  rules under `.factory/rules/`.
- Kaggle CLI automation: `scripts/download_kaggle_data.py` (handles
  KGAT_-format Bearer-auth tokens), `scripts/install_arc_agi_sdk.py`
  (offline SDK install from bundled wheels), `scripts/resubmit_ash.sh`
  (variance-probe helper for the Ash baseline).
- Research dossiers: ARC-AGI-3 landscape review, Exa Deep Research
  report, captured Ash notebook, H100 + Qwen environment probes.
- Local agent harness:
  - `experiments/local_runner.py` (offline mock + arc-agi SDK fallback,
    runner v2.0).
  - `experiments/EXPERIMENTS.md` (per-day experiment tracker).
  - `agents/__init__.py` (Protocol-based agent contract, MockFrame,
    SDK enum fallbacks).
  - `agents/agent.py` (local stub mirroring upstream
    `agents.agent.Agent`).
  - `agents/random_agent.py`, `agents/greedy_explore_agent.py`
    (baseline agents).
- Ash FORGE v19 port:
  - `agents/_ash_my_agent_v19.py` - bit-for-bit verbatim copy of
    the Kaggle Ash notebook cell #1; treated as vendored.
  - `agents/ash_agent.py` - local adapter with
    `find_game_source_and_class` monkey-patch for offline runs.
- Qwen3.6-35B-A3B vision-language agent (exp004 scaffold):
  - `agents/qwen_agent.py` - lazy torch/transformers/PIL imports,
    image+text-grid prompt builder, regex action parser,
    8-step frame-change history.
  - `scripts/qwen_agent_smoke_local.py` - GPU-free smoke
    (22/22 checks pass).
  - `experiments/exp004_qwen_agent/` - README, bundler kernel
    (HF -> private Kaggle Dataset), dev kernel (smoke on H100),
    `RUNBOOK_D2.md` (copy-paste-ready runbook for tomorrow).
- Variance-probe scaffold: `experiments/exp002_ash_variance_probe/`
  with README + helper script.
- Verified facts captured in `.factory/memories.md`:
  - Kaggle exposes 12 accelerators; H100 access works for our
    account.
  - Kaggle's H100 image (`gcr.io/kaggle-gpu-images/python`) ships
    `transformers 5.0.0`, `accelerate 1.12.0`, `torchao 0.10.0`,
    `triton 3.6.0`; lacks vLLM, SGLang, flash-attn, bitsandbytes
    out-of-the-box.
  - Kaggle H100 instance: 80 GB GPU, 31 GB system RAM, 19.5 GB
    `/kaggle/working`, 1.2 TB `/tmp`.
  - HF -> Kaggle bridge URL `kaggle.com/refs/hf-model/...` returns
    404; the path forward is a bundler dev kernel that
    `huggingface_hub.snapshot_download` -> `kagglehub.dataset_upload`.
  - Memory math: 1 x H100 80 GB fits Qwen3.6-35B-A3B at bf16 (~70 GB)
    with mild headroom; cannot fit Qwen3.5-235B or Qwen3.5-397B
    even at INT4. Multi-GPU is not exposed to individual users.

### Notes
- Today's daily Kaggle slot has been used (per user). `s2` from the
  Ash variance probe lands ~24 h after submission and will be
  appended to `.factory/memories.md` and this changelog tomorrow.
