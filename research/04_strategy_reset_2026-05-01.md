# Strategy Reset — 2026-05-01 (post D2 Qwen3.6 marathon)

> **Audience**: humans + AI assistants resuming this repo from D3 onward.
> **Purpose**: re-anchor strategy on facts gathered from Kaggle code/discussion
> pages, the official ARC-AGI-3 paper (arXiv 2603.24621, ARC Prize Foundation,
> Apr 2026), and the relevant 2025-2026 RL/world-model literature. Replaces
> ad-hoc decisions in `.factory/plan.md` D5..D20 once the user signs off.
>
> This document deliberately lists hard *facts*, *literature*, and *strategy*
> separately so each is auditable without rereading the others.

---

## 1. Hard facts about the competition (verified 2026-05-01)

### 1.1 Scoring (from `arc-agi` SDK CHANGELOG and § 0.9.3)

- **`OperationMode.COMPETITION` is FORCED** on the Kaggle harness. This
  means: API-mediated only, single `make()` per env, **only level resets,
  no game resets**, `get_scorecard` of an in-flight scorecard is forbidden.
  All notebooks pretending to fully reset between tries are silently
  downgraded to level resets.
- **Per-level score is squared.** A level reached at `0.5` becomes `0.25`.
- **Per-game score = average over levels, weighted by `level_index` (1-indexed).**
  Higher levels matter much more than lower levels.
- **Average over ALL 25 environments**, even ones an agent never opens.
  Skipping a game silently zero-fills its slot.
- Score is **action-efficiency-based**: a level solved in `k_agent` actions
  vs the human baseline `k_human` yields `score = max(0, 1 − (k_agent − k_human)
  / window)` (squared after).

### 1.2 Public-LB landscape (Kaggle code page, 2026-05-01)

| Score | Notebook | Approach |
|------:|----------|----------|
| **0.42** | `ashvinsingh/ash-s-arc-agi-3-agent` (silver, 127↑) | fork of CHRONOS (FORGE-derivative) |
| **0.39** | `vyankteshdwivedi/arc-agi-3-hybrid-solver-bfs-cnn-heuristics` | BFS + CNN frame-change + heuristics |
| **0.35** | `aadigupta1601/0-35-forge-v16-trigger-aware-bfs` (silver, 91↑) | FORGE v16 + trigger-aware BFS |
| **0.35** | `marynaborovska/hybrid-search-and-learn-agent` (bronze) | hybrid search + learned policy |
| **0.33** | `karnakbaevarthur/arc-agi-3-level-by-level-logic-explanations` | per-level rule induction |
| **0.32** | `imaadmahmood/stochasticgoose-cnn-frame-change-agent` (silver, 54↑) | StochasticGoose++ with CNN frame-change |
| **0.30** | `poonszesen/redpill-zero-prior-agent-with-latent-planning` (silver, 52↑) | latent-plan zero-prior agent |
| **0.29** | trigger-aware BFS variants | |
| **0.28** | `yuriao/arc-agi-3-memoryagent` | persistent state memory |
| **0.25** | `inversion/arc3-sample-submission-stochastic-goose` (gold sample, 311↑) | **official baseline #2** |
| **0.19** | `inversion/arc3-sample-submission-just-explore` (sample, 33↑) | **our forked baseline** |
| **0.18** | `inversion/arc3-sample-submission-random-agent` (silver sample, 162↑) | random |

> **Painful observation**: our 0.19 is barely above pure Random (0.18) and
> is *6 points BELOW* the public Stochastic-Goose sample (0.25). We have
> been benchmarking against the wrong number; **the floor is 0.25, not 0.19.**

### 1.3 Public set vs private set

- 3 public dev games: `vc33`, `ls20`, `ft09`.
- 3 private comp games: `sp80`, `lp85`, `as66`.
- Each game has 8-10 levels. Public set is ~half of total; the ARC paper's
  agent solves ~30/52 levels combined median.

### 1.4 Competition timeline + prize structure

- **2026-04-22** ARC-AGI-3 paper v1 published.
- **2026-06-30** Milestone 1 (top-3: $25K / $10K / $2.5K).
- **2026-09-30** Milestone 2 (top-3: same).
- **2026-11-02** submissions close; results 2026-12-04.
- Grand Prize $700K-$1M for first 100% solver, must open-source MIT/CC0.
- **Apache-2.0 milestone obligation** for any prize-claimable submission.
- 1 successful submission/24h, 6h wall-clock cap, no internet during eval.

---

## 2. Literature digest (what's known to work)

### 2.1 Frame of reference: ARC-AGI-3 baselines

- **Frontier LLMs direct**: < 1% (Gemini 3.1 Pro 0.37%, Claude Opus 4.6 0.2%).
- **Best purpose-built agent in paper preview**: 12.58% (graph-based explorer).
- **Top public Kaggle**: 0.42 (FORGE-derived hybrid).
- **Human baseline**: 100%.

### 2.2 ARC paper preview agent (best known training-free method)

The ARC Prize Foundation's own preview-leaderboard winner combines:
1. **Frame segmentation**: parse the 64×64 grid into connected components
   (objects) → produces a structured state.
2. **State graph**: nodes = unique frame hashes; edges = `(state, action) →
   next_state` transitions observed.
3. **New-action prioritization**: in any state, prefer actions never tried
   from that state. Enumerate the **frontier** of `(state, untried_action)`
   pairs.
4. **Hierarchical action selection**: priority = function of (untried-action
   count, visual salience of the affected region, distance to frontier).
5. **Shortest-path replay**: when stuck, recompute shortest path on the
   graph back to a frontier state then explore.

This is essentially **Go-Explore** (Ecoffet 2021, *Nature* 590) ported to
ARC: cell archive + go-step (replay) + explore-step. Matches the ARC paper's
median 30/52 levels solved.

### 2.3 World-model approaches with recent traction

| Method | Reference | Sample budget | Notes |
|--------|-----------|---------------|-------|
| **MuZero** | Schrittwieser 2020 | ~10⁹ frames | Learned world model + MCTS. Atari + Go + chess. |
| **EfficientZero / BBF** | Ye 2021, Schwarzer 2023 | ~10⁵ frames | MuZero-style, Atari at human-level efficiency. |
| **Dreamer V3 / V4** | Hafner 2025 (arXiv 2509.24527) | ~10⁵-10⁷ | Latent-imagination world models, scales. |
| **AXIOM** | Heins 2026 (ICLR) | **10⁴ frames** | Object-centric, online Bayesian, **no gradients**, plays games in minutes. |

For ARC-AGI-3's tight action budget, **AXIOM-like object-centric
world-models** are the single most relevant 2025-2026 paradigm.

### 2.4 Exploration mechanisms

- **Go-Explore** (Ecoffet 2021): solved Montezuma's Revenge / Pitfall,
  archive of cells + go-step + explore-step. Detachment + derailment fixes.
- **DeepCS / Curiosity Search** (Stanton 2018): intra-life novelty grid.
- **RND** (Burda 2018): random-network-distillation intrinsic reward.
- **DTSIL** / Self-Imitation Learning (Guo 2018): replay good trajectories.

### 2.5 ARC-AGI-2 winning solutions (transferable to ARC-AGI-3)

- **NVIDIA NVARC 2025 (24% on ARC-AGI-2)**: synthetic data + TTT on 4B model.
- **Bobokhonov+ LongT5 (2026, 27% on ARC-AGI-2 semiprivate)**: transformer +
  symmetry-aware data augmentation + LoRA TTT + symmetry-aware scoring.
- **2024 Akyürek FDH**: symmetry-based scoring (multi-perspective majority
  voting on rotated/reflected outputs).

The augmentation toolbox from ARC-AGI-2:
1. **Group symmetries** (8 dihedral rotations + reflections).
2. **Color permutations** (10! permutations of the 16-color palette,
   subsampled).
3. **Cellular automata perturbations** (preserve task semantics, alter
   surface).
4. **Grid traversals** (snake, row-by-row, column-major) as alternative
   sequence views.
5. **TTT with LoRA**: lightweight LoRA adapter per task, 100-500 grad steps
   on the prompt's input-output examples.

---

## 3. Why our exp004 Qwen3.6 attempt was structurally doomed

This is the post-mortem the user asked for. Five compounding root causes:

### 3.1 Greedy decode ≠ exploration policy

A VLM with `do_sample=False` and the same image+prompt produces the *same*
first token deterministically. v9-v11 confirmed: the agent outputs ACTION1
50 times in a row on `ls20`. The anti-repeat patch in v11 only cycles
ACTION1↔ACTION2 because once the model flips to ACTION2 and that yields
no-change, the deque is filled with ACTION2 entries, releasing the gate
for ACTION1 again. **Fundamental fix: structured exploration, not better
prompting.**

### 3.2 Vision prefill bottleneck

5.13 s/action steady-state on H100 BF16. Of that, ~4.6 s is image-token
prefill of a 512×512 frame (256 patch tokens × ~17.5 ms/token). Decode is
only ~0.5 s. **Cutting decode tokens (96 → 16) saved <0.5 s.** To fix:
- Drop `_FRAME_UPSCALE` from 8 to 4 → 256×256 px → ~64 patch tokens.
- vLLM with prefix-cache reuse for the persistent system prompt.
- Or: skip vision entirely; use text-only model + symbolic frame encoding.

### 3.3 No memory across turns

Our agent's `_history` stores `(action, frame_changed)` but nothing about
*states*. There is no persistent graph of `(state_hash, action) → result`,
so the model rediscovers in turn N+1 what it already knew in turn N.
Every prompt is built from scratch. **Top public agents (memoryAgent 0.28,
graph-explorers 0.30+) all maintain state-keyed memory.**

### 3.4 No frame-change feedback signal

The prompt tells the model "ACTION1 has been tried 30 times" but does not
tell it "ACTION1 *changed nothing*". A frame-change CNN (StochasticGoose++)
takes a `Δframe` and predicts which actions actually move the world; this
is the single feature that lifts random-0.18 to goose-0.25 to
StochasticGoose++-0.32.

### 3.5 No ACTION6 coordinate prediction path

`ACTION6` is the click action and requires `(x, y) ∈ [0, 63]²`. We never
exercised this code path in dev kernel because the model never picked it.
For 3+ of the 25 games, ACTION6 is the only meaningful action; without it
we silently zero-fill those 3 games at the start.

### 3.6 What we *did* permanently gain from exp004

- 4 Kaggle-image gotchas locked in (#11 nested mounts, #12 SDK anonkey,
  #13 Pillow C-ext, #14 transformers 5.0.0 missing qwen3_5_moe).
- Kaggle CLI submission flow fully documented in `D2_EXECUTION_LOG.md`.
- `transformers 5.7.0` wheels Dataset reusable for any future kernel.
- A reusable `--target` install pattern for transformers/Pillow that
  any vision agent on this Kaggle image will need.

These survive the strategy reset. The *specific decision to use Qwen3.6
as the policy* does not.

---

## 4. New strategy (D3 onward)

### 4.1 Anchor: 0.42 in ≤ 4 weeks, then climb

The existing public ceiling is 0.42 (Ash). The gap from our anchor (0.19)
is purely a matter of porting + improving on the public CHRONOS / FORGE-v16
pipeline. We do **not** need novel ML to reach 0.42; we need to:
1. Match the public state-of-the-art reliably.
2. Add **memory + frame-change CNN + new-action prioritization** to climb
   to 0.42-0.50.
3. Layer **AXIOM-style object-centric world model + learned policy** to
   push past 0.50 toward the 12.58% paper-leaderboard ceiling and beyond.

### 4.2 Three-tier architecture

```
                  ┌─ Tier C: VLM verifier (Qwen text-only or Llama-3.1-70B)
                  │   Used only at decision points, NOT per-action.
        ┌─────────┤
        │         └─ Tier B: object-centric world model + planner
        │              (AXIOM-lite: GMM slot model + tMM transitions +
        │               rMM interactions, MCTS-lite over plans)
        │
        └─ Tier A: search agent (CHRONOS/FORGE port + memory + frame-change
                                  CNN + Go-Explore archive)
```

**Tier A is the floor we get to in week 1-2.** Tiers B and C buy headroom
in weeks 3-4 and beyond.

### 4.3 4-week milestone plan

| Week | Days | Goal | Target LB |
|------|------|------|-----------|
| 1 | D3-D7 | Port CHRONOS/FORGE-v16 + trigger-aware BFS verbatim | 0.32-0.35 |
| 2 | D8-D14 | Add state-graph memory + new-action prioritization (Go-Explore) | 0.35-0.42 |
| 3 | D15-D21 | Frame-change CNN + ACTION6 coord head | 0.42-0.48 |
| 4 | D22-D28 | AXIOM-lite world model + shallow MCTS | 0.48-0.55 |

After D28, we evaluate whether to (a) continue scaling AXIOM, (b) train a
TTT-LoRA per game offline on collected trajectories, or (c) hybridize with
a small Qwen3-7B text-only verifier for hard levels.

### 4.4 Concrete D3-D7 task list (week 1)

- **D3 (today)**: Submit a FORGE variant for variance probe (Track A from
  RUNBOOK_D2). Anchor: do we get 0.19 ± 0.03 again? Read CHRONOS source
  in detail, port to `agents/chronos_agent.py`. **No Qwen submissions.**
- **D4**: Port StochasticGoose to `agents/goose_agent.py`. Smoke-pass
  the 22-check parity test. Submit StochasticGoose; expect ~0.25.
- **D5**: Port Trigger-Aware BFS from `aadigupta1601/0-35-forge-v16-trigger-aware-bfs`
  → `agents/trigger_bfs_agent.py`. Local smoke must show ≥ 1 level on
  ls20 in ≤ 200 actions. Submit; expect 0.32-0.35.
- **D6**: Add a **state-graph wrapper** (frame hash → node, edges =
  observed transitions) reusable across all agents. Persists across
  actions in `agents/state_graph.py`. Wire it into ChronosAgent and
  TriggerBFSAgent.
- **D7**: New-action prioritization: when picking a node from the BFS
  frontier, prefer states with the most untried actions. Submit; expect
  0.36-0.40.

### 4.5 Concrete D8-D14 task list (week 2)

- **D8**: Frame-change CNN — train offline on `runs/*.json` + ARC-AGI-3
  recordings. Goal: predict P(action changes frame | state, action)
  ∈ [0, 1]. Reuses ARC dataset frames at 64×64×16 → small 100k-param CNN.
- **D9**: Wire CNN into goose-style agent → `agents/goose_pp_agent.py`.
  Submit; expect 0.30+ (matches StochasticGoose++ public 0.32).
- **D10**: Go-Explore archive: cell = downscaled 16×16 frame hash; go-step
  uses replay (we have access to `env.reset()` levels in COMPETITION mode).
- **D11**: ACTION6 coordinate head — small 2-conv CNN that predicts a
  click heatmap from the current frame. Trained from the recordings of
  public scoring sessions where ACTION6 was the winning move.
- **D12**: Combined CHRONOS + Go-Explore archive + frame-change CNN +
  ACTION6 head → `agents/forge_v20_agent.py`. Submit; target 0.42.
- **D13-D14**: Variance probe + ablation study: ablate each component
  in the dev kernel; record which contributes how much.

### 4.6 Data augmentation experiments (per user's request)

These are *training-data* augmentations for the offline-trained components
(frame-change CNN, ACTION6 coord head, Tier-B world model). Borrowed
directly from the ARC-AGI-2 winning toolbox:

#### A1. Group symmetries (8x cheap multiplier)

For every recorded `(frame, action, next_frame)` tuple, generate 8 rotated/
reflected views:
- 4 rotations × 2 reflections = D₄ dihedral group.
- Action labels are also rotated/reflected (e.g. ACTION6 click coordinate
  must follow the symmetry).
- Run with this on day 1 of CNN training; expect 8× effective dataset.

#### A2. Color permutations (subsampled from 16!)

ARC-AGI-3 uses 16 colors with no semantic ordering. Random permutations
of the palette produce equivalent-task data:
- Use 50-100 random color permutations per training tuple.
- Reduces overfitting to specific color codes (which is a known failure
  mode — see `MXF+23` in ARC-AGI-2 paper).

#### A3. Replay-buffer hindsight relabeling

For every `trajectory = (s_0, a_0, s_1, a_1, ..., s_T)` we collect on
Kaggle:
- Treat any `s_t` that was reached as a "goal that succeeded".
- Train the policy to predict the action sequence `a_0..a_{t-1}` *given*
  goal `s_t`, even if the original game's goal was elsewhere.
- This is **HER (Hindsight Experience Replay)** from Andrychowicz 2017.
- Massively expands rare positive signal in sparse-reward games.

#### A4. Cellular automata perturbations

Apply small CA perturbations (Rule 30, Rule 110, Game-of-Life-step) to a
random subset of cells per frame. Provides surface variation without
breaking task semantics. Direct port from the ARC-AGI-2 augmentation
toolbox.

#### A5. Grid traversal augmentations

For sequence-model components, present the same frame as:
- Row-major (default).
- Column-major.
- Snake (alternating row direction).
- Hilbert curve (locality-preserving).
- Z-order (Morton code).

Forces representation invariance to scan order.

#### A6. Cross-game synthetic data

Use the **6 public games** (ls20, vc33, ft09 + sp80, lp85, as66 visible
on the public preview) as seed environments. For each:
1. Record N = 1000 trajectories of an explorer agent.
2. Apply A1-A5 to each.
3. Mix into a single ~50k-tuple training set.

Expected effective dataset: 50k × 8 (A1) × 50 (A2) × 5 (A5) ≈ 100M tuples.
Per ARC-AGI-2 results, augmentation alone moved scores from 8% to 27%.

### 4.7 What we are *not* doing in week 1

- Not submitting Qwen3.6-VL again. The pure-VLM-as-policy path is killed.
  Possible future revival: text-only Qwen3-7B/32B as a Tier-C verifier of
  proposed plans, but only after Tier A reaches 0.42.
- Not training a large model from scratch. Effective compute budget per
  agent on Kaggle is ~5 h H100. Anything > 100M params is a bad bet.
- Not investing in offline BFS that ignores the squared-score-per-level.
  Discussion thread "Is Offline BFS Cheating the Spirit?" suggests the
  grader will *penalize* over-search via action-efficiency scoring. Stay
  efficient.

---

## 5. Decision for today (D3, 2026-05-01)

The user has decided **not to submit Qwen** today (correct — score
projection is 0.0-0.1, regression from 0.19).

**Recommended use of today's slot**:
- **Track A — FORGE variance probe**: re-submit the unchanged ForgeAgent
  notebook. We get a second `(0.19 ± σ)` data point from the same code,
  finally completing the variance probe we should have done on D1.
- This burns the slot harmlessly and produces the variance number that
  the "Phase 0" plan in `.factory/plan.md` was waiting on since D1.

**Background work today**:
- Read CHRONOS (`projectforty2/forge-arc-agi-3-agent`) and Ash
  (`ashvinsingh/ash-s-arc-agi-3-agent`) source line-by-line.
- Sketch out `agents/state_graph.py`.
- Scaffold offline data pipeline: `scripts/collect_recordings.py` to dump
  `(frame, action, next_frame, score_delta)` tuples from any agent run.

---

## 6. Open questions / things to confirm

1. **Is the Kaggle harness scoring the public-set 3 games, the private-set
   3 games, or all 25?** (Section 1.1 asserts "all 25 even unattempted",
   but the 25 number comes from the SDK's environment list — confirm via
   reading `ENVIRONMENTS_DIR/metadata.json`.)
2. **Does `env.reset()` in COMPETITION mode reset the level (preserving
   level index) or only the in-level state?** Per SDK docs §0.9.3: "Only
   Level Resets are permitted, Game Resets are not allowed and become
   Level Resets". We need to verify our agent is not assuming game-reset
   semantics anywhere.
3. **Is there an action-efficiency window we should publish per game?**
   Some Kaggle threads claim leaked human action counts; verify via
   discussion thread "Human scores visible to agents" by Jeroen Cottaar.
4. **Does the "Stochastic Goose 0.25" sample do anything we don't?**
   Read its source — it may be the cheapest 0.25-floor we can lock in
   tomorrow with one PR.

---

## 7. References (consulted 2026-05-01)

- ARC Prize Foundation. *ARC-AGI-3: A New Challenge for Frontier Agentic
  Intelligence*. arXiv:2603.24621v2, Apr 2026.
- Ecoffet et al. *First return, then explore.* Nature 590, 580-586 (2021).
- Heins et al. *AXIOM: Learning to Play Games in Minutes with Expanding
  Object-Centric Models.* ICLR 2026 (arXiv:2505.24784).
- Schrittwieser et al. *Mastering Atari, Go, chess and shogi by planning
  with a learned model.* Nature 588 (2020) — MuZero.
- Hafner et al. *Training Agents Inside of Scalable World Models.*
  arXiv:2509.24527 (Sep 2025).
- Schwarzer et al. *Bigger, Better, Faster: Human-level Atari with
  human-level efficiency.* ICML 2023.
- Ying et al. *Assessing Adaptive World Models in Machines with Novel
  Games.* arXiv:2507.12821 (2025).
- Andrychowicz et al. *Hindsight Experience Replay.* NeurIPS 2017.
- Pathak et al. *Curiosity-driven Exploration by Self-supervised
  Prediction.* ICML 2017.
- Bobokhonov et al. *ARC-AGI-2 Technical Report.* arXiv:2603.06590, Nov 2025.
- Top Kaggle public notebooks (April 2026):
  - `ashvinsingh/ash-s-arc-agi-3-agent` (0.42)
  - `vyankteshdwivedi/arc-agi-3-hybrid-solver-bfs-cnn-heuristics` (0.39)
  - `aadigupta1601/0-35-forge-v16-trigger-aware-bfs` (0.35)
  - `imaadmahmood/stochasticgoose-cnn-frame-change-agent` (0.32)
  - `poonszesen/redpill-zero-prior-agent-with-latent-planning` (0.30)
  - `yuriao/arc-agi-3-memoryagent` (0.28)
- Top Kaggle discussion threads:
  - Cottaar: "Simplified submission framework", "Human scores visible to
    agents", "Intended compute budget".
  - Annaswamy: "Stop searching the Action Tree, Discover the Transformation
    Graph (milestone chain)".
  - CPMP: "It is 0.66%" (clarifying that LB displays raw fractions not
    percentages).
  - Komil Parmar: "Is Offline BFS 'Cheating the Spirit' of ARC-AGI-3?"
