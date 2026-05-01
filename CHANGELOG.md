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

### Added - 2026-04-30 (D2: exp004 Qwen comp submission via CLI)
- New private Kaggle Datasets:
  - `cataluna84/arc-agi-3-agents-pkg` (164 KB, 7 .py files): mirrored
    `agents/` tree for offline mounting in dev/comp kernels.
  - `cataluna84/qwen3-6-35b-a3b-bf16` (71.93 GB, 26 safetensors shards):
    HF mirror of `Qwen/Qwen3.6-35B-A3B` BF16, uploaded via the
    `bundle_qwen_kernel` one-shot internet-enabled bundler.
  - `cataluna84/arc-agi-3-transformers-wheels` (16 MB, 10 wheels):
    transformers 5.7.0 + tokenizers 0.22.2 + accelerate + huggingface_hub
    + safetensors + 5 small deps. Required because the Kaggle H100 image
    (April 2026) ships transformers 5.0.0 which does NOT recognize the
    `qwen3_5_moe` architecture.
- New experiment kernels:
  - `experiments/exp004_qwen_agent/dev_kernel/` (private, no submission):
    smoke-tests QwenAgent on H100 with 50 actions on `ls20`. Resolves four
    Kaggle-image bugs (nested mount layout, SDK anonkey call, Pillow
    11.3.0 `_Ink`/C-extension mismatch, transformers 5.0.0).
  - `experiments/exp004_qwen_agent/comp_kernel/` (private, BURNS daily
    slot): production submission kernel that wraps QwenAgent in
    `agents.agent.Agent` shim and runs the official ARC-AGI-3-Agents
    harness in `KAGGLE_IS_COMPETITION_RERUN` mode.
  - `experiments/exp004_qwen_agent/transformers_bundle_kernel/`: one-shot
    bundler (kept for reference; final wheels were uploaded from local box
    via `kaggle datasets create` because `kagglehub.dataset_upload` 403's
    on `CreateDatasetVersion` from inside a kernel).
- New documentation: `experiments/exp004_qwen_agent/D2_EXECUTION_LOG.md`
  -- the literal terminal log of the entire D2 CLI flow (P1-P5 pre-checks,
  Steps 1-2, Step 3 v1-v11 iteration log, decision points, speed math).
- New gotchas (`.factory/rules/gotchas.md` #11-14):
  - Kaggle datasets now mount nested at `/kaggle/input/datasets/<owner>/<slug>/`
    (legacy `/kaggle/input/<slug>/` no longer works).
  - arc-agi SDK calls `three.arcprize.org/api/games/anonkey` by default;
    must set `OperationMode.OFFLINE` + `environments_dir`.
  - Pillow 11.3.0 + transformers vision models: `--target` install +
    `sys.path.insert(0, ...)` is the only working fix.
  - transformers 5.0.0 does not recognize qwen3_5_moe; bundle 5.7.0 + deps
    as a private Dataset and install with `--target --no-deps`.
- Daily submission BURNED: `submission.parquet` from
  `cataluna84/qwen-comp-arc-agi-3` v1, message "exp004 D2 Qwen3.6-35B-A3B
  BF16 baseline (Track B, expect ~0)". Status: PENDING at submission time.

### Changed - 2026-04-30 (D2: agent speed + diversity tuning)
- `agents/qwen_agent.py`:
  - `_DEFAULT_MAX_NEW`: 96 -> 16. Decode contributes ~1s of the 5.5s
    per-action; cutting it shaves ~5s of total wall.
  - System prompt: "Reply with ONE action only, NO explanation." (was
    multi-line "think briefly" CoT instruction). Drops the text-grid
    block from the user prompt; the image alone encodes the grid.
  - Added anti-repeat post-processor: if an action has yielded no-change
    >= 3 times in recent history, deterministically rotate to the next
    available action. Prevents the greedy-decode lock-in that picked
    ACTION1 50 times in a row in v9 dev kernel smoke.
- `scripts/qwen_agent_smoke_local.py`: prompt assertions updated to
  match the terser system prompt; 21/21 checks pass.

### Changed - 2026-04-30 (D2: rename "ash" -> "forge" across the codebase)
- Rename agent file `agents/ash_agent.py` -> `agents/forge_agent.py` and
  the verbatim vendored cell `agents/_ash_my_agent_v19.py` ->
  `agents/_forge_v19.py`. Class `AshAgent` -> `ForgeAgent`.
- Rename experiment directories `exp001_baseline_ash` ->
  `exp001_baseline_forge` and `exp002_ash_variance_probe` ->
  `exp002_forge_variance_probe`.
- Rename helper script `scripts/resubmit_ash.sh` -> `scripts/resubmit_forge.sh`.
- Update all internal references (code, docs, plans, rules) to use the
  algorithm's technical name "FORGE" / "FORGE v19" instead of personal
  attribution. Upstream credit is preserved in `NOTICE` and
  `research/ash_notebook/` (the captured upstream artefact directory keeps
  its original name to make its provenance unambiguous).
- All file moves use `git mv` so history is preserved.

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
  (variance-probe helper for the FORGE baseline).
- Research dossiers: ARC-AGI-3 landscape review, Exa Deep Research
  report, captured upstream notebook, H100 + Qwen environment probes.
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
- FORGE v19 port (upstream attribution in `NOTICE`):
  - `agents/_forge_v19.py` - bit-for-bit verbatim copy of the
    upstream Kaggle notebook cell #1; treated as vendored.
  - `agents/forge_agent.py` - local adapter with
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
  FORGE variance probe lands ~24 h after submission and will be
  appended to `.factory/memories.md` and this changelog tomorrow.
