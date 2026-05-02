# PROGRESS.md — Running log

> Format: append a new dated section at the **top** every day. Keep older entries.
> Don't rewrite history; if a fact changes, add a corrective entry above with a back-pointer.

---

## 2026-05-02 — D4 afternoon: exp005 Trigger-BFS ablation submitted (PENDING)

### What we did

Per user direction at the D4 morning AskUser checkpoint: Option B (skip
Goose, jump to D5 Trigger-Aware BFS), strict smoke gate, and Option C
when local results showed regression risk (submit anyway as ablation
data point).

#### 1. New shared module: agents/state_graph.py (D6 work, bundled)
- `hash_frame(layers) -> bytes`: 8-byte blake2b digest, deterministic.
- `StateNode` dataclass: state_hash, visit_count, untried_actions, edges,
  incoming_change_score, last_levels.
- `StateGraph`: nodes dict + frontier deque + action_history. Methods:
  `reset()`, `maybe_reset_for_level(levels) -> bool`, `add_or_get(...)`,
  `observe(prev, action, next, change_score)`, `record_action(...)`.
- 5 pytest unit tests in `tests/test_state_graph.py` (hash determinism +
  distinctness, untried seeding minus RESET, observe drains untried
  actions and frontier, level transition wipes graph). All PASS.
- `tests/conftest.py` added so `agents/` is importable from pytest.

#### 2. New agent: agents/trigger_bfs_agent.py
- Pure-Python state-graph agent. No CNN, no torch.
- `_trigger_score(p, n, prev_levels, next_levels) = delta_pixels +
  5*delta_levels + 2*new_colors`.
- `_sample_click_xy(layers, rng)`: bg-color detection via majority bin
  count, sample (x, y) from non-bg pixels; fallback uniform.
- Strategy after several iterations:
  - First attempt (priority-ordered untried): worse than random in
    sweep, monotonic ACTION1 because every step's slight frame change
    re-seeded "untried" with all simple actions.
  - Second attempt (weighted random + stuck detector): completely
    broken (0/25 games passed level 1).
  - Final (random-uniform over untried, fall back to highest-change-
    score edge, then uniform over all): matches RandomAgent (1/25).
- Smoke 22/22 PASS via `scripts/trigger_bfs_smoke_local.py`.

#### 3. Strict smoke gate result: NOT MET on ls20
- Required: levels_completed >= 1 on `ls20` real SDK.
- Achieved on `ls20`: 0 levels in 1000 actions (terminates GAME_OVER).
- RandomAgent also fails ls20 (0 levels in 1000 actions). The original
  SPEC's "ls20" gate was over-optimistic; FORGE BFS is needed for ls20.
- Achieved on broader sweep: 1/25 games (ft09 in 116 actions) -- net
  parity with RandomAgent's 1/25 (r11l). Likely LB outcome 0.18-0.22.

#### 4. Submission decision (per AskUser Option C)
- Acknowledged regression risk: predicted LB 0.18-0.22 < current best 0.24.
- Submitted anyway as an ablation: confirms whether state-graph dedup
  alone moves the needle vs random.

#### 5. Comp kernel push + submit
- Built `experiments/exp005_trigger_aware_bfs/comp_kernel/`
  (kernel-metadata.json + 6-cell trigger_bfs_comp.ipynb mirroring the
  FORGE kernel structure). Cell 1 inlines a self-contained 8.3 KB
  `my_agent.py` with hash + StateGraph + MyAgent (subclass of
  `agents.agent.Agent` with dual-arg signatures).
- `kaggle kernels push` -> version 1 -> RUNNING -> COMPLETE in ~30 s.
- Save-mode log clean (21 s, same benign deps warnings as FORGE save).
- `kaggle competitions submit arc-prize-2026-arc-agi-3 -k cataluna84/trigger-bfs-comp-arc-agi-3 -v 1 -f submission.parquet`
  accepted at 14:24:01 UTC. Status: PENDING.

### Files created/modified (all dated 2026-05-02 D4 afternoon)
- Created: `agents/state_graph.py`, `agents/trigger_bfs_agent.py`,
  `tests/test_state_graph.py`, `tests/conftest.py`,
  `scripts/trigger_bfs_smoke_local.py`,
  `experiments/exp005_trigger_aware_bfs/{README.md,scores.json}`,
  `experiments/exp005_trigger_aware_bfs/comp_kernel/{kernel-metadata.json,trigger_bfs_comp.ipynb}`.
- Submitted: kernel `cataluna84/trigger-bfs-comp-arc-agi-3` v1.

### Verification
- `uv run ruff check .`: clean.
- `uv run ruff format --check .`: clean.
- `uv run pytest tests/test_state_graph.py`: 5/5 PASS.
- `uv run python scripts/trigger_bfs_smoke_local.py`: 22/22 PASS.

### Open
- s1 (exp005) lands ~24 h. Append to scores.json at that point.
- D5: per SPEC_4WEEKS §1.3 was Trigger-BFS submit (now D4 instead).
  Tomorrow's D5 should be EITHER: extend with BFS replay (originally D7)
  if exp005 LB is encouraging, OR pivot to a different track if exp005
  regresses to the random floor.

---

## 2026-05-02 — D4 morning: D3 LB result landed = 0.24 (+0.05 vs 0.19)

### What we learned
- Kernel `cataluna84/ash-s-arc-agi-3-agent` v2 final LB score: **0.24** (status COMPLETE, submitted 2026-05-01 18:32:55 UTC).
- Variance probe data points so far: s1=0.19 (kernel v1, 2026-04-29), s2=0.24 (kernel v2, 2026-05-01). Range 0.05, mean 0.215.
- **Verdict (per `experiments/exp002_forge_variance_probe/scores.json` decision rule)**:
  - max(s1, s2) = 0.24 < 0.30 threshold → **NOT variance-dominated**, multi-seed best-of-N is not the right lift.
  - max(s1, s2) = 0.24 < 0.25 threshold → **structural floor near 0.19** is the leading explanation; pivot to other agents per SPEC_4WEEKS is the correct path.
  - Caveat: the +26% relative jump (0.19→0.24) is non-trivial, hinting at *some* run-to-run variance from the FORGE codebase. A third probe (s3) would give a stddev estimate; it is optional and can be deferred. Sticking with the rule: proceed to D4 Stochastic Goose port.

### Leaderboard context (as of 2026-05-02 13:52:04 UTC)
- **Public LB pulled** via `kaggle competitions leaderboard arc-prize-2026-arc-agi-3 --download`. Saved at `/tmp/lb/arc-prize-2026-arc-agi-3-publicleaderboard-2026-05-02T13:52:04.csv` (49 KB, 715 teams).
- Our row: rank **315 / 715** (top 44%), TeamId=15773862 "Mayank Bhaskar", username `cataluna84`, submission count 3.
- We are inside a **27-team tied cluster all at 0.24** (ranks 291-317 inclusive). One step up to 0.25 → 28 teams; up to 0.26 → another 28; up to 0.27 → 29.
- Score distribution heavy in the 0.17-0.28 noise band (≈250 teams of 715 ≈ 35%). Cliff above 0.30: 20 teams at 0.30, 20 at 0.29, 18 at 0.35, 17 at 0.33. Each +0.01 above 0.30 climbs ~10-20 ranks.
- Top: 0.68 (Redfield Rentals, 20 submissions). Public notebook ceiling 0.42 = StochasticGoose v7 + variants. Gap from us to that ceiling: 0.18.

### Kernel artifact verification (downloaded via `kaggle kernels output`)
- `submission.parquet` (2.6 KB, 1 row): `row_id='1_0', game_id='1', end_of_game=True, score=1`. Confirmed this is the **save-mode placeholder** — actual eval happens server-side in COMPETITION_RERUN; the 0.24 score is computed there, not from this file.
- `my_agent.py` (98 KB): inlined notebook source. Sanity-checks: it is a clean copy of upstream FORGE v19; no diff vs `experiments/exp002_forge_variance_probe/_pulled/ash-s-arc-agi-3-agent.ipynb`.
- `ash-s-arc-agi-3-agent.log` (9.6 KB JSON streamed): pip install of `arc-agi-0.9.8`, `arcengine-0.9.3`, `pillow-12.2.0` from competition wheels — succeeded; nbconvert wrote 130 KB notebook + 785 KB HTML; **save-mode wall clock = 16 s**. Two existing benign warnings: (a) gradio 5.50.0 wants pillow<12.0 (non-fatal — gradio not used in eval), (b) dopamine-rl wants gym<=0.25.2 (non-fatal — dopamine not used). No errors, no tracebacks.

### Files touched (all dated 2026-05-02)
- Updated: `experiments/exp002_forge_variance_probe/scores.json` (s2 resolved to 0.24, summary block populated, LB context).
- Updated: `.factory/memories.md` (this dated section, top).
- Updated: `CHANGELOG.md` (D4 morning section).
- Installed (uv venv): `pyarrow` for parquet inspection.

### Open
- D4 (today): begin Stochastic Goose port per SPEC_4WEEKS.md §1.2. Submit if smoke clean.
- Optional: a third FORGE probe (s3) on some later build day to confirm the +0.05 variance is real. Not blocking.

---

## 2026-05-01 — D3 strategy reset + Qwen postmortem + Track A FORGE resubmit

### What we did

#### 1. Strategy reset doc
- Yesterday's Qwen submission landed on the LB at **0.00**, regressing from 0.19 (back-pointer to 2026-04-30 entry: Qwen got the LB number we feared, not the ~0.0-0.1 hope). Two LB data points so far: (0.19, 0.00). Mean 0.095, neither is informative about FORGE variance.
- User asked for fresh deep research + new plan from scratch. Six exa searches across the ARC-AGI-3 literature (paper arXiv 2603.24621, Heins 2026 AXIOM, ARC-AGI-2 winners, public Kaggle code) yielded:
  - Public LB landscape: 0.42 (Ash) → 0.39 (hybrid) → 0.35 (Trigger-BFS) → 0.32 (Goose++) → 0.25 (sample-Goose) → 0.19 (OURS) → 0.18 (random).
  - **Painful insight**: 0.19 is barely above random; the public floor is 0.25.
  - Scoring rules confirmed from SDK §0.9.3: per-level squared, per-game weighted by `level_index` (1-indexed), averaged across 25 envs. COMPETITION mode forces level resets only.
  - 6 augmentation strategies catalogued: D₄ symmetries (8x), color permutations (50x), HER, CA perturbations, grid traversals, cross-game synthetic data.
- Created `research/04_strategy_reset_2026-05-01.md` (7 sections: hard facts, literature digest, Qwen failure analysis, 4-week plan, 6 augmentations, today's decision, references).

#### 2. Qwen postmortem
- Created `experiments/exp004_qwen_agent/POSTMORTEM.md` with 5 root causes:
  1. Greedy-decode determinism (no temperature, no sampling diversity → action collapse).
  2. Vision prefill bottleneck (4.6s of 5.13s/action on encoding, not decoding — model never gets enough budget).
  3. No state memory (every turn re-prompted from scratch → no progress tracking).
  4. No frame-change feedback (anti-repeat patch fired in some states but had no real change-detection).
  5. No ACTION6 click-coord path (model emits "ACTION6" with no `(x,y)` data; harness silently maps to (0,0)).
- What we keep: agent class scaffold, kernel-metadata patterns, transformers-bundle dataset (frozen).
- New gotchas #11-15 ratified in `gotchas.md`.

#### 3. SPEC_4WEEKS (D3-D28) drafted
- Created `experiments/SPEC_4WEEKS.md` (~620 lines): per-day Goal / Files / Implementation pattern / Smoke test / Submit? / Exit / Rollback. Submit days (11): D3, D4, D5, D7, D9, D12, D17, D20, D24, D27. Build-only days (15) deliberately leave slots open for emergency Track A fallbacks.
- Submission slot accounting table at §5.6 mirrors the daily cadence so an agent picking up mid-stream knows exactly which kernel slug to push.
- Cross-cutting concerns documented: Kaggle Datasets to maintain (6 entries), augmentation scoreboard table (`AUGMENTATION_SCOREBOARD.md`), tests to maintain (5 pytest modules), pre-commit hygiene, hard constraints, open questions reserved for D6/D11.

#### 4. Track A FORGE variance probe — slot consumed
- `bash scripts/resubmit_forge.sh` → kernel `cataluna84/ash-s-arc-agi-3-agent` v2 pushed → status COMPLETE in ~6 min wall.
- Initial submission attempt failed: tried `kaggle competitions submit-code ...`. CLI 2.1.0 does not have a `submit-code` subcommand; the actual code-competition submit API is `kaggle competitions submit <competition> -k <kernel> -v <version> -f <output_file>` (positional competition arg, no `-c`, no `submit-code`). Documented as gotcha #15.
- Correct command landed: `kaggle competitions submit arc-prize-2026-arc-agi-3 -k cataluna84/ash-s-arc-agi-3-agent -v 2 -f submission.parquet -m "exp002 D3 (2026-05-01) variance probe - FORGE baseline unchanged"`.
- `kaggle competitions submissions arc-prize-2026-arc-agi-3` confirms PENDING entry at 2026-05-01 18:32:55 UTC. Will resolve in ~24 h; will record as `s2` in `experiments/exp002_forge_variance_probe/scores.json`.

### Files touched (all dated 2026-05-01)
- Created: `research/04_strategy_reset_2026-05-01.md`, `experiments/exp004_qwen_agent/POSTMORTEM.md`, `experiments/SPEC_4WEEKS.md`, `experiments/exp002_forge_variance_probe/_pulled/` (auto-pulled).
- Modified: Kaggle kernel `cataluna84/ash-s-arc-agi-3-agent` pushed as version 2 (notebook unchanged).

### Verification
- `kaggle competitions submissions` shows 3 submissions (D0=0.19, D1/Qwen=0.00, D3 PENDING).
- SPEC_4WEEKS.md lints clean.

### Open
- s2 will land ~24 h. Append to `experiments/exp002_forge_variance_probe/scores.json` once visible.
- D4: begin Stochastic Goose port per SPEC_4WEEKS §1.2.

---

## 2026-04-30 — D2 exp004 Qwen comp submission via CLI (Track B)

### What we did
- **CLI-driven full submission walkthrough**, documented top-to-bottom in `experiments/exp004_qwen_agent/D2_EXECUTION_LOG.md`. This is now the canonical contributor onboarding doc for "how do I push a Kaggle kernel and submit it from the CLI for ARC-AGI-3?".
- **Three new private Kaggle Datasets**:
  - `cataluna84/arc-agi-3-agents-pkg` (164 KB): mirrored `agents/` package so dev/comp kernels can import `qwen_agent_lib.QwenAgent` without internet.
  - `cataluna84/qwen3-6-35b-a3b-bf16` (71.93 GB, 26 safetensors shards): HF mirror of `Qwen/Qwen3.6-35B-A3B`, uploaded via internet-enabled `bundle_qwen_kernel` (~19 min upload).
  - `cataluna84/arc-agi-3-transformers-wheels` (16 MB, 10 wheels): transformers 5.7.0 + tokenizers 0.22.2 + 8 deps. Uploaded from local box via `kaggle datasets create -p ... --dir-mode tar` (the in-kernel `kagglehub.dataset_upload` 403's on `CreateDatasetVersion`).
- **Step 3 dev kernel**: 11 iterations resolving Kaggle image bugs, all logged. Final v11 ran 50 actions on `ls20` in 780s (5.13s/action steady-state, 528s model load).
- **Step 4 comp kernel**: new directory `experiments/exp004_qwen_agent/comp_kernel/` with a notebook that mirrors dev_kernel setup but writes `/kaggle/working/my_agent.py` shim, copies the official ARC-AGI-3-Agents harness from competition data, registers `MyAgent`, and runs `python main.py --agent myagent` only inside `KAGGLE_IS_COMPETITION_RERUN`. Save-test passed clean at 09:20 UTC.
- **Step 5 submission BURNED** at 09:21 UTC: `kaggle competitions submit arc-prize-2026-arc-agi-3 -k cataluna84/qwen-comp-arc-agi-3 -v 1 -f submission.parquet -m "exp004 D2 Qwen3.6-35B-A3B BF16 baseline (Track B, expect ~0)"`. Status: PENDING.

### Speed iteration (v9 -> v10 -> v11)
- v9 (06:54-07:11 UTC): 5.50 s/action steady-state, 853s total, 0 levels in 50 actions on ls20.
- v10 (07:27 UTC): max_new_tokens 96 -> 16, dropped text-grid from prompt. Result: 5.96 s/action - tiny improvement. Conclusion: decode wasn't the bottleneck.
- v11 (07:47-08:03 UTC): added anti-repeat post-processor that rotates action when frame stays unchanged. Result: action distribution shifted from {ACTION1: 50} -> {ACTION1: 31, ACTION2: 19}. Per-action: 5.13s. Still 0 levels.
- Vision prefill is the bottleneck. The 35B-A3B model spends most of each 5s on encoding the upscaled 512x512 image + chat-template tokens. To cut further we need vllm/sglang OR smaller image (256x256) OR smaller model OR kv-cache persistence between turns.

### Expected score
- ~0.0-0.1. We submitted anyway because the user wanted a Track B LB datapoint locked in. Below the 0.19 ForgeAgent baseline (yesterday's submission).

### Files touched (all dated 2026-04-30)
- Created: `experiments/exp004_qwen_agent/D2_EXECUTION_LOG.md` (~850 lines), `experiments/exp004_qwen_agent/transformers_bundle_kernel/`, `experiments/exp004_qwen_agent/comp_kernel/`.
- Modified: `agents/qwen_agent.py` (max_new_tokens, prompt, anti-repeat), `scripts/qwen_agent_smoke_local.py` (prompt assertions), `experiments/exp004_qwen_agent/dev_kernel/qwen_agent_dev.ipynb` (path adapter, OFFLINE mode, PIL --target install, transformers --target install), `experiments/exp004_qwen_agent/dev_kernel/kernel-metadata.json` (dataset_sources +transformers-wheels), `.factory/rules/gotchas.md` (gotchas #11-14), `CHANGELOG.md`.

### Verification
- `uv run ruff check .` clean.
- `uv run ruff format --check .` clean.
- `uv run python scripts/qwen_agent_smoke_local.py`: 21/21 pass (was 22; lost the dropped "Text grid" check).
- `uv run pre-commit run --all-files`: all hooks green.

### Open
- Submission status: PENDING as of 09:21 UTC. Real eval can take many hours; check `kaggle competitions submissions arc-prize-2026-arc-agi-3` periodically.
- Next steps for D3: vision prefill speedup (vllm or smaller image), action diversity beyond 2-action rotation, possibly switch to Qwen2.5-VL-7B-Instruct (smaller, supported by transformers 5.0.0, no Track B dependencies).

---

## 2026-04-30 — Repo rename: "ash" → "forge" across all internal references

### What we did
- User feedback (rightly): the repo had "Ash" (the upstream Kaggle notebook author's handle) name-dropped throughout file names, class names, plan files, READMEs, and rules — making it look like the project was personally collaborating with that author when in fact we just forked their public notebook.
- Renamed the algorithm-level identifiers to **FORGE / FORGE v19 / ForgeAgent**. FORGE is the technical name of the algorithm (BFS + ForgeNet CNN) used in the upstream code and is descriptive without naming a person.
- File renames (all via `git mv` so history is preserved):
  - `agents/ash_agent.py` → `agents/forge_agent.py`
  - `agents/_ash_my_agent_v19.py` → `agents/_forge_v19.py`
  - `experiments/exp001_baseline_ash/` → `experiments/exp001_baseline_forge/`
  - `experiments/exp002_ash_variance_probe/` → `experiments/exp002_forge_variance_probe/`
  - `scripts/resubmit_ash.sh` → `scripts/resubmit_forge.sh`
- Symbol renames: `AshAgent` → `ForgeAgent`, `_ash_my_agent_v19` (module) → `_forge_v19`.
- Doc/comment renames: "Ash's ARC-AGI-3 Agent" → "the upstream FORGE-v19 notebook" / "FORGE v19 baseline".
- Kept upstream attribution (Apache 2.0 obligation) **only** in `NOTICE` and the `research/ash_notebook/` directory (the captured upstream artefact directory keeps its original name to make its provenance unambiguous).
- Score is unchanged: **0.19, rank 398, 2026-04-29**. The user clarified "I only achieved a score of 0.19" — every doc that says 0.19 stays 0.19; the upstream's advertised 0.42 is referenced only as the upstream's number, never as our achievement.

### Files touched
- Renamed via `git mv`: 5 files (above).
- Edited references: `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `NOTICE`, `CITATION.cff`, `pyproject.toml`, `.pre-commit-config.yaml`, `experiments/EXPERIMENTS.md`, `experiments/local_runner.py`, `experiments/exp004_qwen_agent/RUNBOOK_D2.md`, `.factory/plan.md`, `.factory/rules/conventions.md`, `.factory/rules/leaderboard-anchors.md`, `.factory/rules/gotchas.md`, `.factory/rules/kaggle-submission.md`.
- Pre-commit and ruff exclusion rules updated to point at the new vendored filename `agents/_forge_v19.py`.

### Verification
- `uv run ruff check .` clean.
- `uv run ruff format --check .` clean.
- `uv run python scripts/qwen_agent_smoke_local.py`: 22/22 pass.
- `uv run python experiments/local_runner.py --agent agents.random_agent:RandomAgent --games ls20-mock --max-actions 30` exits 0.
- `uv run pre-commit run --all-files` all hooks green.

### Open
- Today's daily Kaggle slot: still open. Decision tree from `RUNBOOK_D2.md` § 1 still applies. The score from yesterday's exp001 (now `exp001_baseline_forge`) is **0.19** — confirmed.

---

## 2026-04-29 (later x7) — exp004 scaffold for Qwen3.6-35B-A3B; H100 dev kernel hard limits

### What we did
- Pushed v2 of `cataluna84/h100-probe-arc-agi-3` to enumerate **inference frameworks pre-installed on Kaggle's H100 image**. Result: `torch 2.10.0+cu128, transformers 5.0.0, accelerate 1.12.0, torchao 0.10.0, peft 0.18.1, safetensors 0.7.0, tokenizers 0.22.2, huggingface_hub 1.4.1, triton 3.6.0` — all there. **NOT pre-installed: vllm, sglang, flash_attn, flashinfer, xformers, bitsandbytes, auto_gptq, autoawq, optimum, lmdeploy, tensorrt_llm.** Hopper FP8 native works (`torch.float8_e4m3fn` and `torch.float8_e5m2` both available).
- Pushed `cataluna84/qwen-bridge-probe-arc-agi-3` to verify (a) HF→Kaggle bridge URL, (b) HF reachability from a dev kernel, (c) actual Kaggle H100 instance specs. Results:
  - **`https://www.kaggle.com/refs/hf-model/<owner>/<model>` returns 404** — not a working endpoint.
  - **HF still works inside dev kernels** if `enable_internet=true`. Verified by downloading `Qwen/Qwen3.6-35B-A3B/config.json` via `huggingface_hub.hf_hub_download`.
  - **Kaggle H100 system specs (verified):** GPU = 1× H100 80 GB HBM3 (sm_90, FP8 native), **CPU RAM = 31.4 GB** (NOT 128 GB+ as I had assumed), **`/kaggle/working` = 19.5 GB** (cannot stage 70 GB weights here), **`/tmp` = 1220 GB free** of 8 TB total (used as HF cache). `kagglehub 1.0.0` is pre-installed and exposes `model_download` + `dataset_upload`.
- Verified Qwen3.6-35B-A3B on HF Hub: 35.95B params, **all BF16, 26 safetensors shards, ~70 GB total**, Apache-2.0, `image-text-to-text` pipeline, `Qwen3_5MoeForConditionalGeneration` arch, 1.5M downloads.

### Hard-won facts (DO NOT redo this analysis)
- **Memory math for 1× H100 80 GB**:
  - bf16: 35B fits (~70 GB), 72B doesn't (~144 GB), 397B-A17B emphatically doesn't (~794 GB).
  - INT4: up to ~140B fits. So Qwen3-Next-80B-A3B (~40 GB INT4) is the biggest model that fits via quantization.
  - **`Qwen3.5-397B-A17B` cannot run on Kaggle's single H100 at any precision** — it is 5× too big at INT4 (~200 GB) and 9.5× too big at bf16 (~794 GB). Multi-GPU inference is NOT exposed to individual users.
- **CPU offloading is dead**: Kaggle's 31 GB system RAM cannot help if the model exceeds 80 GB GPU. Models must fit fully on the GPU. Forget about expert-streaming / disk-offload setups for this competition.
- **Bundling path for HF model weights**: forget the `/refs/hf-model/` URL. The right path is a **bundler dev kernel** that does `huggingface_hub.snapshot_download` then `kagglehub.dataset_upload` → all on Kaggle's gigabit network, no upload from home. Mount the resulting Dataset via `dataset_sources` in the eval kernel.
- Confirmed: dev kernels do NOT consume the daily competition submission slot. We can iterate freely on dev kernels and only burn a slot when calling `kaggle competitions submit-code`.

### Files created (all local; no Kaggle pushes today beyond the two probes)
- `agents/qwen_agent.py` (~340 lines): VLM agent with lazy torch/transformers/PIL imports, image+text-grid prompt builder, action regex parser (handles `ACTION3`, `Action: 3`, `action_4`, `RESET`, `(x, y)`, `x= y=`), `choose_action(frame) -> GameAction`, frame-change history capped to 8 steps. Reads env vars `QWEN_MODEL_PATH/QWEN_DTYPE/QWEN_DEVICE_MAP/QWEN_MAX_NEW_TOKENS/QWEN_HISTORY_LEN/QWEN_DEBUG_PROMPTS`.
- `experiments/exp004_qwen_agent/README.md`: runbook for the bundle → dev → submit sequence.
- `experiments/exp004_qwen_agent/bundle_qwen_kernel/{bundle_qwen.ipynb, kernel-metadata.json}`: dev kernel that downloads Qwen3.6-35B-A3B from HF (allow-list `*.json`, `model-*.safetensors`, `tokenizer*`, `model.safetensors.index.json`; ignore `*.bin/*.gguf/*.onnx`) into `/tmp/qwen36_bf16/` then uploads to private Kaggle Dataset `cataluna84/qwen3-6-35b-a3b-bf16`. `enable_gpu=false` (CPU is enough for download + upload), `enable_internet=true`.
- `experiments/exp004_qwen_agent/dev_kernel/{qwen_agent_dev.ipynb, kernel-metadata.json}`: smoke kernel that mounts the bundled Dataset + the competition data, installs the SDK from the bundled wheels, runs `QwenAgent` against `ls20` for up to 50 actions. `enable_internet=false`, `accelerator=NvidiaH100`. **Depends on a separate `cataluna84/arc-agi-3-agents-pkg` Dataset that bundles our `agents/` directory** — TODO: create this Dataset before pushing the dev kernel.
- `scripts/qwen_agent_smoke_local.py`: GPU-free verification of `build_prompt`, `parse_action`, and an end-to-end loop on the offline `MockGame` with synthetic replies. **All 22 checks pass.** No torch / transformers / Qwen weights required.

### Open trade-offs explicitly deferred
- **INT4 / FP8 alternatives**: exp005 will swap weights to `Qwen3-Next-80B-A3B-INT4` and `Qwen3.5-VL-30B-A3B-Thinking` using the same agent code. Same Kaggle Dataset bundling pattern.
- **vLLM / SGLang**: skip in v1 (transformers + accelerate is fine for single-batch step inference). Reconsider only if per-action latency exceeds ~5-10 s on the dev kernel.
- **ACTION6 coordinate handling**: v1 parses `(x, y)` from the model's reply; if missing, falls back to grid centre `(32, 32)`. Sufficient for first smoke test; refine after we see actual replies.

### Next steps (D2-D3)
- **D2 (tomorrow)**:
  1. Create `cataluna84/arc-agi-3-agents-pkg` Kaggle Dataset (just our `agents/` directory) — needed by the dev kernel.
  2. Push `bundle_qwen_kernel` → wait ~45-60 min → verify `cataluna84/qwen3-6-35b-a3b-bf16` exists.
  3. Push `dev_kernel` → wait ~5-15 min for first run (model load is ~3-5 min for 70 GB shards) → review `qwen_smoke.json` output. Pass criteria: per-action latency < 10 s, valid actions ≥ 90%, levels_completed ≥ 1 on `ls20`.
- **D3 (depending on D2 result)**: Either iterate on prompt template (still uses 0 daily slots) or promote dev kernel to a competition kernel and burn the daily slot.

### Today's daily Kaggle slot usage
- User reports they have already submitted today's slot (likely exp001 or exp002; outcome will land later today / tomorrow morning UTC).

---

## 2026-04-29 (later x6) — H100 access CONFIRMED + strategy doc + revised Docker analysis

### What we did
- Ran two parallel research probes (Exa Deep Reasoning + Ref MCP for Kaggle docs) on (a) ARC-AGI-3 LB strategy and (b) Kaggle programmatic compute access. Saved synthesis as `research/03_strategy_and_kaggle_compute_2026-04-29.md`.
- Discovered the official `kaggle-cli/docs/kernels.md` lists 12 accelerator types (Feb 2026): P100, T4, T4Highmem, **A100**, **L4**, L4X1, **H100**, **RtxPro6000**, plus 4 TPU types. Older docs (e.g. product-feedback #173129 saying "P100 only") are stale.
- Inspected the upstream FORGE notebook's `kernel-metadata.json`: it declares `"machine_shape": "NvidiaTeslaT4"`. Older PUBLIC interpretation was "T4 is the eval GPU".
- **Pushed a 1-cell `nvidia-smi + torch.cuda` probe kernel to Kaggle with `--accelerator NvidiaH100` to verify allocation.** Result: kernel ran in ~14s, allocated **NVIDIA H100 80 GB HBM3 (sm_90, Hopper), driver 580.105.08, CUDA 13.0, container `gcr.io/kaggle-gpu-images/python@sha256:00377c...` (same SHA as the `private-byod` image we couldn't pull earlier — they're mirrored), Python 3.12.12, torch 2.10.0+cu128**. Saved to `runs/h100_probe/{h100-probe-arc-agi-3.log, h100_probe_result.json}`.

### Hard-won facts
- **`gcr.io/kaggle-private-byod/python` and `gcr.io/kaggle-gpu-images/python` mirror the SAME SHA** (`00377cd1b3d470a605bc5b0ceca79969e369644e9b36802242a1c70e627372f9`). The 401 we hit earlier on `private-byod` was a permission misdirect — the same image is publicly accessible under `kaggle-gpu-images`. **If we ever want Docker parity, pull `gcr.io/kaggle-gpu-images/python:latest` (~40 GB).**
- **The user has H100 access today via `kaggle kernels push --accelerator NvidiaH100`.** No special opt-in needed beyond attaching the kernel to `competition_sources: ["arc-prize-2026-arc-agi-3"]`. Quota is undocumented but the probe ran fine.
- **The upstream FORGE notebook's `machine_shape: NvidiaTeslaT4` is metadata-only / probably stale.** The actual rerun container is the H100 image with the same SHA we saw on the H100 probe. Top public agents likely run on H100 during the competition rerun even if their dev metadata says T4.
- **Container Python 3.12.12 + torch 2.10.0+cu128** matches our local venv (Python 3.12, torch 2.11.0+cu128). Bit-perfect parity is essentially achieved without pulling the Docker image.
- The KGAT_ token works for `kaggle kernels push` / `status` / `output` but **not** for `kaggle competitions list` (returns 401). Likely a Bearer-auth code path missing on that endpoint. Workaround: use the kernels-API path for competition kernels (which is what we want anyway).

### Strategy implications (full doc: research/03_strategy_and_kaggle_compute_2026-04-29.md)
- **Tier 4 moonshots (bundled 7B-30B LLM, DreamerV3) are now feasible-on-Kaggle.** 80 GB H100 fits a 4-bit-quantized 30B-class LLM with room to spare for context. We were treating these as 4+ week longshots; they're now realistic 1-2 week experiments if we want them.
- **The H100 doesn't help BFS-bound agents (FORGE / Trigger-Aware BFS).** BFS is pure-Python deepcopy + `perform_action` on the game class. No CUDA path. Our local 132s/50-actions on RTX 2070 is essentially what it'll be on H100 modulo CPU speed.
- **Local development + Kaggle dev-kernel parity:** since the H100 image SHA is identical to the rerun image SHA (or at least the same family), pushing experimental kernels via `kaggle kernels push` (without submitting) gives us bit-perfect runtime tests at the cost of a few minutes of GPU time. Far cheaper than burning the daily 1-submission slot on a probe.
- **Public LB landscape (April 2026)**: top private = 0.68; top public ≈ 0.42 (the upstream FORGE-v19 notebook's advertised number); our reproduction of that fork = 0.19 (rank 398). Mid public 0.30-0.39 (Trigger-Aware BFS, FORGE, Hybrid Search-and-Learn). Low public 0.18-0.28 (Random, Just Explore, Stochastic Goose 0.25 sample baseline). Many speculative architectures published April 2026 with NO score = author wrote, never submitted. Frontier LLMs zero-shot = 0.001-0.005, so pure-language reasoning is worse than Random.

### Files touched
- Created: `research/03_strategy_and_kaggle_compute_2026-04-29.md`, `experiments/kernel_h100_probe/{kernel-metadata.json, h100_probe.ipynb}`, `runs/h100_probe/{h100_probe_result.json, h100-probe-arc-agi-3.log}`.
- Modified: `.factory/rules/kaggle-submission.md` (corrected the "H100 only for ARC-AGI-3 notebooks" claim — actually all 12 accelerators are exposed; T4 is what `machine_shape` typically declares but H100 IS allocated when explicitly requested via `--accelerator NvidiaH100`).

### Next steps (revised, given H100 confirmed)
- D1 (today/tomorrow): exp002 — FORGE variance probe, resubmit unchanged (1 daily slot).
- D2: exp002 — FORGE resubmit #2 (variance probe).
- **OR** in parallel via dev-kernels (no submission cost): start prototyping a Tier 4 path (DreamerV3-torch or 7B local-Qwen agent) on H100 dev kernels — this is now realistic.
- Reset .factory/rules/kaggle-submission.md to reflect H100 reality.

---

## 2026-04-29 (later x5) — FORGE-v19 port runs locally + CUDA torch stack + Docker / H100 parity decision

### What we did
- **Ported the upstream FORGE-v19 cell (`ashvinsingh/ash-s-arc-agi-3-agent`) into `agents/forge_agent.py`** (file later renamed from `ash_agent.py`; see 2026-04-30 entry above).
  - Pulled the notebook via `kaggle kernels pull` (saved under `research/ash_notebook/` — captured-artefact directory keeps its original name).
  - Saved cell #1 verbatim (2055 lines) as `agents/_forge_v19.py` (file later renamed from `_ash_my_agent_v19.py`). **Do NOT edit; treat as a vendored copy.** If the upstream notebook updates, re-pull and overwrite.
  - Added `agents/agent.py` — local stub for `agents.agent.Agent` so the verbatim file's `from agents.agent import Agent` resolves. Mirrors the upstream harness surface (game_id, arc_env, frames, action_counter, etc.). **Does not set `self.recorder = None`** because the vendored upstream `append_frame` does `if hasattr(self, "recorder"): self.recorder.record(...)` — a `None` recorder would crash the call.
  - `agents/forge_agent.py` is the adapter: lazy-imports `_forge_v19` (so missing torch gives a clear error), monkey-patches `find_game_source_and_class` to also search our local `data/kaggle/.../environment_files/<gid>/<guid>/<gid>.py`, and exposes `ForgeAgent` matching our `local_runner` contract (`choose_action(frame)` instead of upstream's `choose_action(frames, lf)`).
  - `experiments/local_runner.py` now passes `arc_env` and `game_id` to agents whose `__init__` accepts them (introspected via `inspect.signature`).
- **Smoke tested FORGE + SDK on `ls20`** (50-action budget):
  - Result: `levels_completed=1`, `win_levels=7`, action histogram across [1,2,3,4]. **Random / Greedy got 0 levels in the same budget — the FORGE BFS solver is a real win.**
  - Wall-clock: 130.5s on CPU torch, 132.6s on CUDA torch — **GPU offers no speedup** because the dominant cost is BFSSolver (`copy.deepcopy(game)` + `perform_action()` thousands of times in pure Python+numpy). The CNN (ForgeNet, 4 conv layers, batch=1) is tiny relative to BFS.
- **Docker / H100 parity decision** (asked the user, picked Option 3 = no Docker).
  - The image referenced in `kernel-metadata.json` is `gcr.io/kaggle-private-byod/python@sha256:00377c…` and **returns HTTP 401** (private registry).
  - The public alternative `gcr.io/kaggle-images/python` (HTTP 200, ~40 GB) is ~99% equivalent but NOT bit-perfect.
  - Local hardware is RTX 2070 + GTX 1650 SUPER (both Turing, sm_75) — bit-perfect H100 reproduction is impossible at the GPU layer regardless of Docker.
  - Decision: **skip Docker. Stay on uv venv. Use CUDA torch on the local RTX 2070** for any future GPU-heavy work.
- **Installed CUDA torch into the venv**: replaced `torch==2.11.0+cpu` with `torch==2.11.0+cu128` from `https://download.pytorch.org/whl/cu128`. Verified: `torch.cuda.is_available()=True`, 2 GPUs detected, matmul on cuda:0 works.

### Hard-won facts
- `gcr.io/kaggle-private-byod/python` is private (HTTP 401). The kernel-metadata.json `docker_image` field cannot be honored as-is from outside Kaggle.
- For ARC-AGI-3 BFS-heavy agents (FORGE / Trigger-Aware BFS), **GPU is not the bottleneck**. Wall-clock is dominated by the Python-level state-space search (deepcopy + perform_action). GPU helps the CNN inference but that's <5% of total time.
- WSL2 GPU passthrough with a Windows host driver works **without** Docker — `libcuda.so.1` lives at `/usr/lib/wsl/lib/libcuda.so.1` and torch's cu128 wheels detect it correctly.
- The upstream `append_frame` does `if hasattr(self, "recorder"): self.recorder.record(...)`. If a base class sets `self.recorder = None`, hasattr returns True and the code crashes on `None.record(...)`. **Leave the attribute unset** in the base class.

### Files touched
- Created: `agents/_forge_v19.py` (verbatim upstream FORGE source, 2055 lines; renamed 2026-04-30 from `_ash_my_agent_v19.py`), `agents/forge_agent.py` (adapter; renamed 2026-04-30 from `ash_agent.py`), `agents/agent.py` (Agent base stub), `research/ash_notebook/{ash-s-arc-agi-3-agent.ipynb, kernel-metadata.json, ash_my_agent_v19_verbatim.py, ash_notebook_extracted.txt}` (captured-artefact directory keeps its original slug for provenance).
- Modified: `experiments/local_runner.py` (passes arc_env / game_id to agents that accept them).
- venv state: `torch 2.11.0+cu128` + CUDA 12.8 deps (~3 GB).

### Next steps
- **exp002** — variance probe: run FORGE + SDK on ls20 across 5 seeds with `--max-actions 200` to estimate per-game variance and see how high `levels_completed` climbs with a real budget.
- Decide whether to try a non-FORGE baseline next (Just-Explore / MCTS / GooseBumps) or focus on improving the FORGE BFS (per-game heuristics, better hidden-field detection).
- If a future agent actually needs an H100-class GPU for inference, revisit the Kaggle-image plan.

---

## 2026-04-29 (later x3) — Agents + local_runner migrated to real SDK contract

### What we did
- Migrated `agents/*` and `experiments/local_runner.py` from the placeholder string-based action contract to the real `arc_agi 0.9.x` / `arcengine 0.9.3` interface.
- New agent contract (matches official harness conventions):
  ```python
  class MyAgent:
      name: str = "..."
      def __init__(self, seed: int = 0): ...
      def choose_action(self, frame) -> GameAction: ...
      def is_done(self, frame) -> bool: ...      # optional
  ```
  where `frame` quacks like `arcengine.FrameDataRaw` (has `.state`, `.levels_completed`, `.win_levels`, `.available_actions`, `.frame`, `.guid`, `.full_reset`).
- `agents/__init__.py` now provides:
  - Real `GameAction` / `GameState` from `arcengine` if installed, else lightweight stand-ins.
  - `MockFrame` dataclass for the offline mock backend.
  - `Agent` Protocol for static type-checking against the runner's loop.
- `agents/random_agent.py` and `agents/greedy_explore_agent.py` now return `GameAction` enums and filter by `frame.available_actions`. Greedy hashes frames numpy-aware so it works on both backends.
- `agents/forge_agent.py` placeholder updated to the new contract; still raises `NotImplementedError`.
- `experiments/local_runner.py` rewritten to:
  - Use `env.observation_space` + `env.step(action, data=..., reasoning=None)` for the SDK backend.
  - Produce `MockFrame` objects for the offline mock backend so agents are backend-agnostic.
  - Report `levels_completed` / `win_levels` / `final_state` from `FrameDataRaw`.
  - Add `--quiet-sdk-logs` flag for less noisy CI output.

### Hard-won facts
- **`arcengine.GameAction` is a plain `Enum`, not `IntEnum`** despite values being ints. `GameAction(int_value)` raises `ValueError`. Always use **`GameAction.from_id(int)`** (or the enum literal `GameAction.ACTION3` etc).
- `Arcade.make(env_id)` constructs a `LocalEnvironmentWrapper` and immediately calls `reset()`. The initial observation is `env.observation_space` (which is a property returning the latest `FrameDataRaw`).
- `env.action_space` is the **filtered** list of `GameAction`s allowed for the current level. `frame.available_actions` is the same info as a `list[int]`. Agents must filter on this — for example, `ls20` has `available_actions = [1, 2, 3, 4]` (ACTION5/6/7 are NOT available for that game).
- `step()` signature: `step(action: GameAction, data: dict | None, reasoning: dict | None) -> Optional[FrameDataRaw]`. Returning `None` means env terminated (treat as game-over).
- `GameAction.is_complex()` is True only for `ACTION6` (which takes `(x, y)` click coords via `action.set_data({"x":..,"y":..})`). All others are simple.
- `arcengine.GameState` is a `str` `Enum` with values `NOT_PLAYED`, `NOT_FINISHED`, `WIN`, `GAME_OVER`. To serialize to JSON, use `state.value`.
- `FrameDataRaw.frame` is `list[np.ndarray]` (one 64x64 array per layer), NOT a list of lists. Hashing / comparing frames must handle both numpy and list inputs.

### Verified outputs (smoke tests, seed 0)
- **Mock backend / RandomAgent**: 100 actions → GAME_OVER (random uniform doesn't reliably hit win condition; expected).
- **Mock backend / GreedyExploreAgent**: 141 actions → WIN all 3 levels (greedy correctly learns ACTION1/ACTION2 are the change-driving actions for the mock).
- **SDK / RandomAgent on ls20**: 50 actions, action histogram `{ACTION1:13, ACTION2:10, ACTION3:14, ACTION4:13}` — proves `available_actions = [1,2,3,4]` filter works (no ACTION5/6/7 attempted).
- **SDK / GreedyExploreAgent on ls20**: 50 actions, no levels completed (50 actions is too short to learn ls20; baseline behaviour).

### Next steps (D2 onwards)
1. **exp002 — FORGE variance probe**: Now that we have a working local harness, port `agents/forge_agent.py` from the upstream notebook (Just-Explore-style heuristics) and verify locally on `ls20` + `ft09` before submitting.
2. Add a `--multi-game` flag to local_runner that loops over all 25 games to measure rough RHAE.
3. Consider adding a `Recorder`-style replay capture so we can reproduce specific failure modes offline.

---

## 2026-04-29 (latest) — Kaggle data + SDK installed locally

### What we did
- Hit a stubborn **401 Unauthenticated** on every comp-data call from `kaggle 2.1.0` and `kagglehub 1.0.1`, despite valid creds.
- Diagnosed by hitting Kaggle's REST endpoints directly with `curl`:
  - `Authorization: Basic` (the legacy way both clients use by default with `KAGGLE_USERNAME`+`KAGGLE_KEY`) → **401** on `/api/v1/competitions/list` and `/v1/competitions.CompetitionApiService/...`.
  - `Authorization: Bearer <key>` → **200** on the same endpoints; comp file list returned successfully.
- Read `kagglesdk/kaggle_env.py:91` and `kagglesdk/kaggle_http_client.py:248-260`: kagglesdk only enables Bearer auth when **`KAGGLE_API_TOKEN`** env var is set. With only `KAGGLE_USERNAME`+`KAGGLE_KEY` set, it falls back to Basic.
- Patched `scripts/download_kaggle_data.py` to detect `KAGGLE_KEY` values starting with `KGAT_` and auto-export `KAGGLE_API_TOKEN` for the kaggle/kagglehub clients to pick up.
- Re-ran download: **42.3 MB pulled**, mirrored to `data/kaggle/arc-prize-2026-arc-agi-3/` with 3 expected dirs:
  - `environment_files/`  — 25 games (`ar25, bp35, cd82, ..., ls20, ..., wa30`)
  - `arc_agi_3_wheels/`   — 31 offline wheels
  - `ARC-AGI-3-Agents/`   — official agent harness (with its own pyproject.toml)
- Patched `scripts/install_arc_agi_sdk.py` to:
  - Use `uv pip install` (uv-managed `.venv` doesn't ship pip).
  - Install BOTH `arc-agi` AND `arcengine` (default).
- SDK install succeeded: `arc-agi==0.9.8`, `arcengine==0.9.3` + 19 transitive deps (numpy, matplotlib, pillow, pydantic, flask, ...).
- Patched `experiments/local_runner.py`: real SDK is `Arcade.make(env_id)`, not the placeholder `.create(...)`.
- SDK round-trip: env created, game `ls20` loaded from `environment_files/ls20/9607627b/`, but step() failed because `RandomAgent` emits string action names; the real `ActionInput.id` is an int enum (0-7). **Deferred to next session.**

### Key facts (for future sessions)
- **KGAT_ tokens require Bearer auth.** Setting `KAGGLE_KEY` alone is NOT enough for newer Kaggle endpoints. Either:
  - Set `KAGGLE_API_TOKEN=<KGAT_...>` directly, OR
  - Let our downloader's auto-detection do it (it now does).
- **Real SDK API surface** (arc_agi 0.9.8):
  - Top level: `Arcade`, `EnvironmentInfo`, `EnvironmentScorecard`, `ScorecardManager`, `LocalEnvironmentWrapper`, `RemoteEnvironmentWrapper`, `RestAPI`, ...
  - `Arcade` methods: `make`, `get_environments`, `create_scorecard`, `open_scorecard`, `close_scorecard`, `get_scorecard`, `listen_and_serve`.
  - `ActionInput.id` is an enum `0..7` (NOT a string like `"ACTION4"`).
  - `arc_agi.local_wrapper.LocalEnvironmentWrapper.step()` is the entry point.
- **SDK auto-fetches game files at runtime** from `https://three.arcprize.org` into `environment_files/<id>/<version>/`. This means `Arcade.make(id)` works even before `download_kaggle_data.py` if you have internet — but during Kaggle eval (no internet) you must pre-bundle the files.

### Next steps (D1 onwards)
1. Migrate `agents/random_agent.py` and `agents/greedy_explore_agent.py` to emit integer action IDs (0-7) and consume the real SDK observation/state shape.
2. Wire `experiments/local_runner.py` to call `env.step(int_action)` and parse `step()` return tuple correctly.
3. Once SDK round-trip is green on `ls20`, port `agents/forge_agent.py` from the upstream notebook (Just-Explore + heuristics).
4. Submit exp002 (FORGE variance probe) to lock down LB stddev.

---

## 2026-04-29 (later) — Baseline anchored at 0.19

### What we did
- User confirmed an existing Kaggle submission: a vanilla fork of an upstream public **FORGE-v19** notebook scored **LB 0.19, rank 398** (see `NOTICE` for upstream attribution).
- Locked in **0.19 as the project baseline anchor** (replacing the assumed 0.25 Stochastic Goose target).
- Created `experiments/exp001_baseline_forge/` (renamed 2026-04-30 from `exp001_baseline_ash/`) with `README.md`, `score.txt` (0.19), `rank.txt` (398).
- Re-numbered Phase-0 in `EXPERIMENTS.md` and `PLAN.md`:
  - exp001 = FORGE baseline (DONE)
  - exp002 = FORGE variance probe (resubmit 2× to estimate stddev) (D1, D2)
  - exp003 = Just-Explore baseline (D3)
  - exp004 = local runner + agents/ skeleton (D4, no Kaggle slot)
  - exp005+ = Phase 1 reproductions (BFS, StochasticGoose CNN, Hybrid)

### Findings worth remembering
1. **−0.23 reproduction gap**: the upstream FORGE-v19 notebook is advertised at ~0.42 public, we got 0.19. Possible causes (in order of likelihood):
   - Stochastic agent + unlucky seed → fixed by exp002 best-of-N.
   - Cherry-picked / best-of-N number originally → expected scoring closer to 0.19–0.30.
   - Eval set has been hardened (more games added) since publication.
2. **The variance probe (exp002) is the highest-value next experiment** — it determines whether re-using the FORGE notebook as a platform is worthwhile or whether we must build our own from scratch.

### Gotchas surfaced
- The leaderboard MHTML snapshot only captured ranks 1–373; rank 398 wasn't in it. (The user confirmed score directly.)
- "Public-LB advertised score" of a notebook is NOT a reliable predictor of what a fork will achieve — variance is high.

### Decisions
- All downstream experiment targets recomputed as Δ over 0.19 (not 0.25).
- Phase 0 expanded from 3 days → 4 days to absorb the variance probe before pivoting.
- exp002 will resubmit the forked FORGE notebook **identically twice** before any agent code changes.

### Next steps (today / tomorrow)
- [x] Wrote `experiments/local_runner.py` (offline smoke harness with mock + SDK fallback).
- [ ] Build `agents/` package skeleton: `random_agent.py`, `greedy_explore_agent.py`, `forge_agent.py` (port of forked notebook).
- [ ] D1: resubmit FORGE unchanged → record `s2` in `exp002_forge_variance_probe/scores.json`.

### Open questions (carried forward)
- Same as 2026-04-29 (bootstrap) entry below — none resolved yet.
- New: what was the **timestamp + dataset version** of the FORGE notebook fork? Pin it so exp002 uses identical inputs.

---

## 2026-04-29 — Bootstrap

### What we did
- Read `documentation/memory-system.png` — established 4-file external memory pattern (AGENTS / PROGRESS / PLAN / VERIFY).
- Extracted plain text from all 5 Kaggle MHTML files at `documentation/kaggle/*_extracted.txt` (overview, data, code, discussion, leaderboard).
- Cataloged the public-LB top entries, top-upvoted public notebooks, and the canonical baseline notebooks (Random=0.18, Stochastic Goose=0.25, Just Explore=0.19).
- Pulled the technical paper (arXiv 2603.24621) and the Graph-Based Exploration paper (arXiv 2512.24156).
- Surveyed the canonical agent repos: `arcprize/ARC-AGI-3-Agents`, `arcprize/ARC-AGI`, `arcprize/arc-agi-3-benchmarking`, `DriesSmit/ARC3-solution` (StochasticGoose), `dolphin-in-a-coma/arc-agi-3-just-explore`, `alexisfox7/RGB-Agent`.
- Started an Exa Deep Researcher Pro task (`r_01kqcepv7wx50df3kg3ad6s3er`) for ARC-AGI-3 SOTA architectures; result still pending at log time.
- Created memory layout: `memory/AGENTS.md`, `memory/PLAN.md`, `memory/PROGRESS.md` (this file), `memory/VERIFY.md` to follow.
- Created scaffolding directories: `research/`, `experiments/`.

### Findings worth remembering
1. **Action efficiency, not levels, drives the score** (RHAE squares the per-level ratio). Solving fewer levels with elite efficiency can outscore solving more levels sloppily.
2. **The 0.66–0.68 cluster at the LB top is achievable with non-LLM methods** — Just-Explore alone got 12/25 levels in the dev preview.
3. **The FORGE-v19 notebook (advertised 0.42)** is the highest publicly-shared starting point. Forking + extending it is a low-risk Day-3 plan.
4. **The Kaggle-eval is fully offline** — RGB-Agent / OpenCode style approaches must be rebuilt with a *local quantized LLM* (Qwen2.5-7B GGUF) bundled as a Kaggle Dataset.
5. The 2026-03 release of the **`ARC-AGI-3-Agents v0.9.3` API** renames `score → levels_completed` and `win_score → win_levels`. Watch for older notebooks still using the old fields.

### Gotchas surfaced
- `dolphin-in-a-coma/arc-agi-3-just-explore` had a graph-reset bug between levels (cost ~5 levels). Always reset state graph + hash table on level transitions.
- The Kaggle `arc-agi` wheel directory must be added as a Kaggle Dataset for offline pip install (`--no-index --find-links=...`).
- Do NOT trust raw level-1 success metrics — tutorials are designed to allow random wins.

### Decisions
- **Cadence**: 1 deliberate submission/day. Always smoke-test locally first.
- **Memory system convention**: 4 markdown files + per-experiment subfolders.
- **Top-priority phase order**: Phase 0 (baselines) → Phase 1 (reproduce 0.35) → Phase 2 (compose 0.45+) before any experimental novel research.

### Next steps (today)
- [x] Finish writing `VERIFY.md` with smoke-test commands and submission verification checklist.
- [ ] Wait for Exa Deep Research result (id `r_01kqcepv7wx50df3kg3ad6s3er`); record in `research/exa_deep_research_2026-04-29.md`.
- [ ] Stub `experiments/exp000_baseline_replay/README.md` so we can fork the Stochastic Goose notebook on Kaggle tomorrow.

### Open questions
- Can we **mirror the public game files locally** to truly reproduce the Kaggle eval offline? (`environment_files/` ships 25 of the 30+ games.)
- Is `ACTION7 == UNDO` always available? (StochasticGoose blog says UNDO was *unavailable* during dev preview; current `v0.9.3` lists ACTION7 as a `GameAction`.)
- What is the **per-game step budget** the eval enforces beyond the 6h wall clock? — flag for the Discussion thread "Intended compute budget".
