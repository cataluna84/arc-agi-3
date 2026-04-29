# PLAN.md — Daily Experiments + Definition of Done

> **North Star**: surpass the public-LB top score (currently 0.68) by 2026-09-30, and strive toward the 100% Grand Prize threshold.
>
> **Constraint**: 1 Kaggle submission/day. Each daily slot must move the score, build a foundation, or kill a hypothesis.
>
> **Locked-in baseline**: the user's existing submission (vanilla fork of **Ash's ARC-AGI-3 Agent**) scored **0.19** at rank 398. Every "Δ over baseline" downstream is measured against **0.19**, not 0.25 nor 0.42.

---

## Phase 0 — Foundation: anchor & understand variance (Days 0–3)

### [x] D0: Vanilla fork of Ash's ARC-AGI-3 Agent (anchor)

**Goal**: anchor the LB pipeline and lock in our baseline number.

- [x] Forked "Ash's ARC-AGI-3 Agent" on Kaggle (advertised public ~0.42)
- [x] Submitted unchanged
- [x] **Result: LB 0.19, rank 398** — a −0.23 reproduction gap from the published number

**DoD**: green submission with a real LB number. (DONE)
See: `experiments/exp001_baseline_ash/`.

### [x] D1: Ash variance probe — resubmit #1 (exp002) — slot used 2026-04-29

**Goal**: figure out whether 0.19 is "the real expected score" or just a low-variance draw.

- [x] Re-submitted (or planned to re-submit) the SAME forked Ash's notebook unchanged
- [ ] Record `s2`; compare to 0.19  (result lands ~24 h after submission, append to memories.md)

**DoD**: a second LB data point recorded; absolute diff `|s2 - 0.19|` logged in `.factory/memories.md`.

### [ ] D2 (2026-04-30): Ash variance probe #2 (exp002) **OR** Qwen agent submission (exp004)

The two tracks compete for tomorrow's daily slot. Decision is made the morning of D2 once `s2` from D1 is visible:

- **Track A — Ash variance probe #2** (consume slot): only sensible if `s2` is unexpectedly high (≥ 0.30) or unexpectedly low (≤ 0.10). Otherwise we already have enough information from `(0.19, s2)` to decide.
- **Track B — exp004 Qwen3.6-35B-A3B agent** (consume slot): submit ONLY if exp004 dev-kernel smoke shows healthy results overnight (per-action latency < 10 s, ≥ 1 level completed on `ls20`). Otherwise dev-kernel iteration continues with no slot used.
- **Track C — defer slot, accelerate exp004**: if neither A nor B is ready, hold the slot and spend the day finishing exp004 dev-kernel iteration. We never have to use the slot.

**DoD**: by end of D2, either (a) `s3` is in `memories.md` for variance, or (b) a Qwen submission has its own LB number, or (c) the day's notes record why we held the slot and what we built instead.

### exp004 milestones (parallel to the slot-using D1-D8 track; runs on dev kernels)

- **D2 morning**: create `cataluna84/arc-agi-3-agents-pkg` Kaggle Dataset bundling our local `agents/` directory (one-time, < 1 min upload).
- **D2 (afternoon)**: push `experiments/exp004_qwen_agent/bundle_qwen_kernel/` → wait ~45-60 min → verify private Dataset `cataluna84/qwen3-6-35b-a3b-bf16` exists with all 26 safetensors shards (~70 GB).
- **D2 (evening) → D3**: push `experiments/exp004_qwen_agent/dev_kernel/` → review `qwen_smoke.json`. **Pass criteria**: model loads in BF16 on H100 without OOM, per-action latency < 10 s, ≥ 90% emitted actions are in `available_actions`, ≥ 1 level completed on `ls20` within 200 actions.
- **D3 / D4**: iterate on prompt template if smoke results are mediocre. Each iteration reuses the dev kernel — no daily slot consumed.
- **D5 (latest)**: promote the dev kernel to a competition kernel; submit; record LB number. **Decision criterion** for promoting: dev-kernel smoke shows ≥ 1 level on at least 3 of 5 sampled training games. If exp004 LB ≥ 0.25 we keep iterating on this track; if < 0.20 we pivot to exp005 (Qwen3-Next-80B-A3B-INT4 swap; same code, different weights).

### [ ] D3: Just-Explore baseline (exp003) — orthogonal reference

- [ ] Fork "ARC3 Sample Submission - Just Explore"; submit unchanged.
- [ ] Read agent loop top-to-bottom; lift the state-graph code into `agents/just_explore.py` for reuse in exp005+.

**DoD**: score recorded; agent loop annotated.

### [x] D4: Local runner + agents/ skeleton (no Kaggle slot)

- [x] `experiments/local_runner.py` drafted with mock + SDK fallback
- [x] `agents/random_agent.py`, `agents/greedy_explore_agent.py`, `agents/ash_agent.py` (port of forked notebook), `agents/qwen_agent.py` (Qwen3.6-35B-A3B VLM)
- [x] `python experiments/local_runner.py --agent agents.random_agent:RandomAgent --games ls20-mock` exits 0
- [x] `python experiments/local_runner.py --agent agents.ash_agent:AshAgent --use-sdk --games ls20` reaches `levels_completed=1` in 50 actions

**DoD**: agents importable; runner exits 0; QwenAgent local smoke (`scripts/qwen_agent_smoke_local.py`) passes 22/22 checks without GPU.

> Note: D4 was originally scoped as "build the runner + 3 baseline agents". It absorbed the Qwen agent build during the late-D0/early-D1 sprint. Real exp004 (Qwen submission) is tracked under the **exp004 milestones** above and competes with D2 for the daily slot.

---

## Phase 1 — Public-notebook reproductions (Days 5–8)

### [ ] D5: Reproduce FORGE Trigger-Aware BFS (target 0.32–0.35)

Read 0.39 + 0.35 notebooks. Re-implement minimally:

- Frame hashing (xxhash or blake2 over (state_grid, available_actions))
- BFS frontier ordered by:
  - **untried action count desc**
  - **trigger score** (Δpixels, Δscore, new colors)
  - **distance to nearest unexplored state** (Just-Explore style)
- State-graph reset on level transitions (avoid the `dolphin` bug)

**DoD**: ≥ 0.30 on Kaggle (Δ ≥ +0.11 over 0.19 baseline).

### [ ] D6: Reproduce StochasticGoose CNN (target 0.25–0.32)

- 16-channel one-hot 64×64 input → 4-layer CNN (32→64→128→256)
- Action head (5-way) + 64×64 conv coordinate head
- Binary loss: did frame change? Hash-deduped buffer up to 200k samples
- Reset model + buffer per level

**DoD**: ≥ 0.32 on Kaggle (Δ ≥ +0.13).

### [ ] D7: Hybrid `BFS + CNN-predicted action change` (target 0.35+)

Combine D5 and D6: use CNN to score each action's change probability, then expand BFS frontier biased by that probability. Falls back to uniform priority when CNN is cold.

**DoD**: ≥ 0.35 on Kaggle (Δ ≥ +0.16).

---

## Phase 2 — Compose toward 0.45+ (Days 9–14)

### [ ] D8: Push past the Ash 0.42 ceiling

The user's Ash fork landed at 0.19 (heavy reproduction gap). After exp002 variance probe, we know if Ash is a useful platform; if so, layer on:
- per-level model reset
- ACTION6 saliency from segmentation
- frame-delta priority

**DoD**: ≥ 0.40 on Kaggle (Δ ≥ +0.21).

### [ ] D9: Add **per-game adaptive budget** (efficiency optimization)

- Detect whether agent has progressed in last K actions.
- If stuck > K (e.g. 200 actions on a level), apply **restart-with-noise** or **switch policy**.
- Save action budget for late levels (RHAE weights levels by index).

**DoD**: same level-completion as D8 but with -10% actions on average → score lift via efficiency.

### [ ] D10: **Object segmentation** + **clickable-object library**

- Extract connected components per color.
- Maintain *symbolic* state (objects + relations + bounding boxes).
- Replace ACTION6(x,y) random with ACTION6(centroid_of_object).

**DoD**: ≥ 0.43, Ash's ceiling broken.

### [ ] D11: **MCTS with neural prior** (UCT + StochasticGoose CNN as policy prior)

- Tree search with PUCT(s,a) = Q(s,a) + c·prior_CNN(s,a)·sqrt(N_parent)/(1+N_child)
- Bound rollouts by remaining action budget
- Reuse subtree on action commit

**DoD**: ≥ 0.45.

### [ ] D12: **DreamerV3 lite** — RSSM world-model rolled in latent (5–20M params, fits H100)

- Train online on every (frame, action, next_frame) transition
- Use latent imagination for 8-step planning
- Use *intrinsic curiosity* (prediction error) to drive exploration

**DoD**: ≥ 0.45 (parity with MCTS) but **better on long-horizon games**.

---

## Phase 3 — Push toward 0.55+ (Days 13–22)

### [ ] D13: **Test-Time Training (TTT)** on transitions of current game

- Per-game LoRA adapter on top of frozen 1B–3B Qwen/DeepSeek backbone
- Fine-tune in-flight on first 100 transitions, then plan
- Inspired by NVIDIA's NVARC win on ARC-AGI-2

**DoD**: ≥ 0.50.

### [ ] D14: **Program/DSL synthesis** (Stitch-style library learner)

- Define a grid DSL (move_object, fill, mirror, swap, copy_region…)
- BFS with MDL prior over short programs explaining observed transitions
- Discover rule from level 1 → reuse in higher levels

**DoD**: solves at least one *deep* level (5+) on a public game.

### [ ] D15: **Slot-attention object-centric world model**

- Slot-Attention encoder → object slots → graph net for relations → next-state predictor
- Combines well with D10 object library

**DoD**: ≥ 0.52.

### [ ] D16: **Causal counterfactual reasoning**

- Maintain belief over which (object, action) edges modify game state.
- Plan to test the most informative edge first (active-learning).

**DoD**: ≥ 0.55.

---

## Phase 4 — Beat the LB (Days 23+)

### [ ] D17: **Ensemble of orthogonal agents**

Per-game, run all of: BFS+CNN, MCTS, DreamerV3-lite, DSL synthesizer in parallel/sequence. Pick the first-finishing winning policy. Use action efficiency as tie-breaker.

### [ ] D18: **Offline-RL warm-start** from replays of all 25 public games + community replays from `three.arcprize.org/replay/...`

Pre-train DreamerV3 / Decision Transformer on action–state traces of public games, then test-time fine-tune.

### [ ] D19: **Local LLM-as-orchestrator** (Qwen 2.5 7B GGUF in offline notebook)

- Use small local LLM only for "what should the next exploration target be?"
- All low-level actions still come from the symbolic+neural agents
- Mirrors RGB-Agent architecture but offline

### [ ] D20+: iterate on top-2 ideas based on which got the highest Δ-score.

---

## Always-on Hygiene (every day)

- **Before** burning the daily Kaggle slot:
  - [ ] Smoke test on at least 3 public games locally; agent does not crash, completes ≥ 1 level on tutorial
  - [ ] Confirm runtime extrapolation: (avg_actions_per_game × 110 games × per_action_seconds) < 5h
  - [ ] Diff vs last successful submission (`git diff` or notebook compare)
- **After** every submission:
  - [ ] Capture LB score, game-by-game scorecard, and any tracebacks → `.factory/memories.md`
  - [ ] If score regressed > 0.02, immediately roll back and bisect

## Definition of Done for each phase

- **Phase 0**: 0.19 baseline anchored; variance of Ash's notebook understood; local runner working. Done.
- **Phase 1**: one of our notebooks ≥ 0.35 on Kaggle (Δ ≥ +0.16 over 0.19 anchor).
- **Phase 2**: ≥ 0.45 on Kaggle (Δ ≥ +0.26).
- **Phase 3**: ≥ 0.55, top-10 on public LB (Δ ≥ +0.36).
- **Phase 4**: ≥ 0.70, beating current LB top (Δ ≥ +0.51).
