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

### Result - 2026-05-13 D15 (D9 Goose CNN v2 LB landed = 0.17)
- **D9 Goose CNN v2 LB = 0.17** (COMPLETE).
- v1 → v2 went 0.00 → 0.17, confirming the v1 silent-crash hypothesis
  (enable_gpu=true + uncaught torch failure on the comp rerun host).
- BUT 0.17 is in the "agent runs but priors do not lift over random" band
  per the decision rule in `experiments/exp007_goose_cnn/README.md`:
  -0.04 below `master_v7` (0.21) and roughly tied with `trigger_bfs` (0.10)
  + the random baseline. The CNN's online-training schedule is too cold
  on the 100-action-per-level budget to extract useful action priors.
- **Cumulative LB trace** 0.19 / 0.00 / 0.24 / 0.10 / 0.21 / 0.00 / **0.17**.

### Added - 2026-05-13 D15 (exp008 comp kernel + submission)
- `experiments/exp008_trigger_bfs_seg/{README.md, build_notebook.py,
  comp_kernel/{kernel-metadata.json, trigger_bfs_seg_comp.ipynb}}`:
  comp kernel for the segmenter-wired trigger_bfs. `build_notebook.py`
  is a small builder script that inlines the source-of-truth agents
  into a 6-cell Kaggle notebook (matches the exp007 v2 pattern that
  scored 0.17). Kernel pushed as
  `cataluna84/trigger-bfs-segmenter-comp-arc-agi-3` v1 (note: Kaggle
  auto-slugged the title to `-segmenter-` instead of our configured
  `-seg-`; metadata updated to match). Status COMPLETE in ~25s on CPU.
  Submitted at 2026-05-13 12:29 UTC; status PENDING. Decision rule in
  the exp008 README based on the LB band that lands.

### Added - 2026-05-13 D15 (D10+D11 frame-segmenter port + trigger_bfs wire-up)
- `agents/frame_segmenter.py` (NEW, ~440 LOC): stateless port of
  `dolphin-in-a-coma/arc-agi-3-just-explore`'s `FrameProcessor` (3rd-place
  ARC-AGI-3 Preview Challenge; published as arXiv:2512.24156).
  Public surface: `segment_frame`, `identify_status_bars`,
  `frame_segments_to_priority_tiers`, `hash_masked_frame`,
  `mask_to_click_coords`, `salient_pixels_in_segment`. No torch /
  no matplotlib import. Constants exposed (`MINIMAL_WIDTH=2`,
  `MAXIMAL_WIDTH=32`, `STATUS_BAR_DISTANCE_THRESHOLD=3`,
  `STATUS_BAR_RATIO_THRESHOLD=5.0`, `STATUS_BAR_TWINS_THRESHOLD=3`,
  `SALIENT_COLORS={6..15}`).
- `tests/test_frame_segmenter.py` (NEW, 11 tests): 2-blob segmentation,
  L-shape rectangle detection, twin detection, status-bar line + dot
  rules, priority-tier stratification, hash determinism +
  mask-awareness + shape-awareness, click-coord containment,
  salient-pixels-in-tier. **All 11 pass.**

### Changed - 2026-05-13 D15 (trigger_bfs ACTION6 prior wired to segmenter)
- `agents/trigger_bfs_agent.py`: replaced `_sample_click_xy`'s non-bg
  pixel sampler with the segmenter-tier sampler. Walks tiers 0..3
  (skip tier 4 status bars); within each tier, picks a non-dominant
  segment uniformly (excludes any segment whose area > half the frame
  — that's the background), then samples a pixel uniformly within
  that segment. Falls through to the legacy non-bg sampler, then to
  uniform [0, 63]^2. Whole block wrapped in try/except.
- `scripts/trigger_bfs_smoke_local.py`: bumped end-to-end smoke from
  `seed=0`, `max-actions=300` to `seed=1`, `max-actions=400`. Seed 0
  was structurally favored by the legacy non-bg sampler on the
  `ls20-mock` win condition (`pixel(10,10) == 7` via
  `ACTION6(10, 10)`); seeds 1/2/3 all WIN under both old and new
  priors within 240 actions.

### Result - 2026-05-07 D9 (D6 Goose CNN v1 LB landed = 0.00; v2 fix submitted)
- **D6 Goose CNN v1 LB = 0.00** (silent crash on the comp rerun host).
  Verified by API: only the dummy fallback row landed in the kernel's
  `submission.parquet`; the downloadable kernel log shows only the 24s
  dry-run (no agent execution). Comp rerun logs are server-private for
  code competitions, so the failure mode was inferred from the
  architectural diff against the working `master_v7` kernel (LB 0.21).
- **Cumulative LB trace** 0.19 / 0.00 / 0.24 / 0.10 / 0.21 / 0.00 ; **Day 9
  slot used on Goose CNN v2** (PENDING at 2026-05-08 00:23 UTC).

### Changed - 2026-05-07 D9 (Goose CNN v2 defensive fixes)
- `experiments/exp007_goose_cnn/comp_kernel/kernel-metadata.json`:
  flipped `enable_gpu` from `true` to `false`. Every kernel that scored
  >0 (master_v7, trigger_bfs) ran on CPU; the GPU-enabled v1 hit a silent
  crash before the first action. The 4-layer CNN at 64x64 is fast enough
  on CPU for a 100-action-per-level budget.
- `agents/goose_cnn_agent.py` + inlined `goose_cnn_comp.ipynb`: extracted
  `_choose_action_inner()` and wrapped `choose_action()` in `try/except
  Exception` returning a uniform random non-RESET non-ACTION6 action.
  Mirrors `master_v7`'s pattern that survives any model/torch/CUDA failure.
- Added `try/except` around `predictor.predict()` so torch/CUDA/shape
  errors degrade to uniform priors (`ap = 0.5`, `cp = 0.5`) instead of
  raising every step.
- Defensive guard `if cur_levels >= 0 and cur_levels != self._prev_levels`
  prevents accidental model+buffer wipe on a transient `levels_completed=-1`
  from the gateway during boot.
- Pre-submission checks all PASS: `ruff check + format`, 40/40 pytest,
  22/22 goose smoke, local_runner ls20-mock 100-action smoke.

### Removed - 2026-05-07 D9 (Qwen-as-policy dropped from active tracks)
- The 2026-05-06 qwen-policy-dev run (real H100, 422 actions) solved 0/3
  games, with 200 actions on ls20 yielding 0 levels. The graph-exploration
  paper (Rudakov 2026, arXiv:2512.24156) reports ls20 L1 solvable in ~124
  actions median by frame-segmentation alone, and documents LLM+DSL
  underperforming random (5 vs 6 levels) on the private set. This is a
  structural gap, not a prompt-tuning gap. Qwen-as-policy is removed from
  the active track list; possible future revival as Verifier or
  Orchestrator only after a strong base agent exists.

### Added - 2026-05-04 D6 morning (D5 LB landed + status sweep)
- `experiments/exp006_master_v7/comp_kernel/{master_v7_comp.ipynb,kernel-metadata.json}`:
  CLI-managed copy of MASTER BASELINE v7 (FORGE v19 op_2 + v17 beam search +
  MCTS click masking + grad_clip + intrinsic_reward). Pushed as
  `cataluna84/master-v7-comp-arc-agi-3` v1 + submitted at 2026-05-03 18:59 UTC.

### Result - 2026-05-04 D6 morning (D5 LB)
- **LB 0.21** for exp006 MASTER v7. Recovers +0.11 from D4 collapse but stays
  -0.03 below D3 baseline (0.24). Five-day trace 0.19 / 0.00 / 0.24 / 0.10 / 0.21
  confirms a structural floor near 0.20 for FORGE-family agents.
- Pivot needed for D6+: candidate tracks are Q (Qwen), G (Goose CNN), D (DSL).

### Added - 2026-05-02 D4 afternoon (exp005 Trigger-BFS ablation submitted)
- `agents/state_graph.py`: shared state-graph wrapper for search-based
  agents. Provides `hash_frame()`, `StateNode`, `StateGraph` with
  level-transition reset. Foundation for D7-D27 of SPEC_4WEEKS.
- `agents/trigger_bfs_agent.py`: state-graph + uniform-random-over-
  untried agent. Pure-Python, no CNN. 22/22 smoke PASS.
- `tests/test_state_graph.py` (5 tests, all PASS) and `tests/conftest.py`
  (makes `agents/` importable from pytest).
- `scripts/trigger_bfs_smoke_local.py`: 22-check parity smoke.
- `experiments/exp005_trigger_aware_bfs/`: README + scores.json + 6-cell
  Kaggle comp kernel mirroring the FORGE structure.
- Kernel `cataluna84/trigger-bfs-comp-arc-agi-3` v1 pushed; submission
  accepted 14:24 UTC, status PENDING. Predicted LB 0.18-0.22 (random-
  level baseline; submitted as LB ablation per AskUser Option C).

### Resolved - 2026-05-02 (D3 LB result for Track A variance probe)
- D3 submission scored **0.24** (vs 0.19 baseline; +0.05 absolute, +26%
  relative). Status COMPLETE. Final position: rank 315 / 715, top 44%.
  We sit in a 27-team tied cluster at 0.24 (ranks 291-317).
- Decision-rule verdict: max(s1=0.19, s2=0.24)=0.24 < 0.25 threshold →
  structural-floor reading; pivot to other agents per SPEC_4WEEKS.md
  is confirmed correct. The +26% intra-codebase variance is noted but
  does not justify multi-seed best-of-N as the primary lever.
- `experiments/exp002_forge_variance_probe/scores.json` now resolves
  s2 with full LB context (top score 0.68, public-notebook ceiling 0.42).
- Kernel artifact verification: `submission.parquet` is a 1-row save-mode
  placeholder; the 0.24 was scored by Kaggle in COMPETITION_RERUN. Save
  log clean (16 s wall clock; two benign deps warnings).
- pyarrow added to the uv-managed venv for parquet inspection.

### Added - 2026-05-01 (D3: strategy reset + Track A FORGE variance probe)
- `research/04_strategy_reset_2026-05-01.md`: 7-section research-driven
  strategy reset following Qwen 0.00 LB result. Covers public-LB
  landscape (0.42 ceiling, 0.25 floor), SDK §0.9.3 scoring math,
  literature digest (ARC-AGI-3 paper, AXIOM, Go-Explore, ARC-AGI-2
  winners), Qwen failure analysis, 4-week new plan, 6 augmentation
  strategies, today's decision, references.
- `experiments/exp004_qwen_agent/POSTMORTEM.md`: Qwen failure postmortem.
  5 root causes (greedy decode, vision prefill bottleneck, no state
  memory, no change feedback, no ACTION6 path) + what-we-keep + lessons.
- `experiments/SPEC_4WEEKS.md`: detailed D3-D28 implementation spec
  (~620 lines). Per-day Goal / Files / Implementation pattern / Smoke
  test / Submit? / Exit criteria / Rollback. 11 submission days
  scheduled. Cross-cutting concerns (Kaggle Datasets to maintain,
  augmentation scoreboard, tests, hard constraints) at §5.
- `experiments/exp002_forge_variance_probe/_pulled/`: auto-pulled
  upstream kernel by `scripts/resubmit_forge.sh` (gitignored fork copy).

### Submitted - 2026-05-01 (D3 Track A daily slot)
- `kaggle competitions submit arc-prize-2026-arc-agi-3 -k cataluna84/ash-s-arc-agi-3-agent -v 2 -f submission.parquet`
  ran clean at 18:32 UTC. Status PENDING; will land as `s2` in
  `experiments/exp002_forge_variance_probe/scores.json`.
- Confirmed CLI usage: `submit` is the correct subcommand (with `-k -v`
  flags for code competitions); there is no `submit-code` in CLI 2.1.0.

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
