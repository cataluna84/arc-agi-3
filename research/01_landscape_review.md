# Landscape Review — ARC-AGI-3 (April 2026)

> First-pass synthesis of the technical paper, the dev-preview winners, and the public Kaggle notebook ecosystem. Purpose: derive the priority order of techniques to implement.

## 1. Benchmark facts (canonical)

- **Paper**: "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence", ARC Prize Foundation, arXiv:**2603.24621** (March 2026, v2 17-Apr-2026).
- **Frame**: 64×64 grid; 16 colors; turn-based; agent submits one action per turn. Frame *sequences* allowed for animations.
- **Action set**: 5 simple keys + Undo (ACTION7) + click (ACTION6 with x,y in [0,63]²). Each game exposes a *subset*.
- **States**: NOT_FINISHED / WIN / GAME_OVER. Levels chain inside a game; later levels reuse mechanics from earlier ones (compositional difficulty).
- **Datasets**: 25 public demo + 55 semi-private (LLM-API benchmarks) + 55 fully-private (Kaggle competition).
- **Anti-random guard**: each non-tutorial level has `P_random_win < 1/10000`, validated by 1M-step random sweep. Tutorials may allow random wins by design.
- **Scoring (RHAE)**: per_level = min(human/agent,1)², per_game = level-index-weighted average, final = mean across games.
- **Frontier AI** scores < 1% in March 2026; humans 100%.

## 2. Public Kaggle Leaderboard (April 2026 snapshot)

| Rank | Score | Team | Likely architecture |
| ---- | ----- | ---- | -------------------- |
| 1 | 0.68 | Redfield Rentals | Trigger-aware BFS + CNN action-change predictor (educated guess) |
| 2 | 0.66 | Barada Sahu | likely BFS + MCTS hybrid |
| 5 | 0.64 | Matthew Philip Poetker | unknown |
| 12 | 0.51 | Mon Tiger | mid-tier |
| 13 | 0.50 | Sergei Fironov | a known Kaggle Grandmaster |
| 21 | 0.43 | Ali | mid-tier |
| 22 | 0.42 | ashvin singh | author of public notebook "Ash's ARC-AGI-3 Agent" (0.42) |

**Best public-scored notebooks** (descending):

1. **Ash's ARC-AGI-3 Agent** — 0.42, 111★ Silver
2. **FORGE ARC-AGI-3 Agent** — 0.39, 124★ Gold
3. **FORGE v16 Trigger-aware BFS** — 0.35, 90★ Silver
4. **Hybrid Search-and-Learn Agent | 0.35+ LB** — 0.35, 17★
5. **StochasticGoose++ CNN Frame-Change Agent** — 0.32, 54★ Silver
6. **Redpill: Zero-Prior Agent with Latent Planning** — 0.30, 52★ Silver
7. **MCTS Solver** — 0.29, 28★
8. **memoryAgent** — 0.28, 21★
9. **Cognitive-Rungs** — 0.21, 21★

## 3. Dev-Preview winners (Aug 2025) — what they actually did

### 3.1 StochasticGoose (1st, 12.58%, Tufa Labs / Dries Smit)

- 16-channel one-hot 64×64 frame as input.
- 4-layer CNN backbone (32→64→128→256 channels).
- **Action head**: 5-way softmax over ACTION1–ACTION5.
- **Coordinate head**: convolutional 64×64 logits for ACTION6 (preserves 2D inductive bias).
- **Self-supervised target**: binary "did this (action,coord) change the next frame?" (BCE + tiny entropy regularizer).
- **Experience buffer**: 200k unique (state, action) pairs; hash-deduped.
- **Per-level reset**: clear buffer + reset weights at level transitions.
- Strategy: **bias exploration toward predicted-change actions** (stochastic sampling over sigmoid scores).

### 3.2 Blind Squirrel (2nd, 6.71%, Will Dick) — public details thin

- Used **frame segmentation** + heuristic exploration (per StochasticGoose's writeup, *"a strategy successfully used by the second-place agent"*).

### 3.3 Just Explore / Graph-Based (3rd, ~12/25, dolphin-in-a-coma)

- Method paper: arXiv **2512.24156** (Rudakov, Shock, Cowley — University of Helsinki / UCT).
- **Frame Processor**: connected-component segmentation, status-bar masking, button-likelihood priority tiers (1=highest), hash of masked grid.
- **Level Graph Explorer**: directed graph; each node is a frame-hash; per-action metadata: `priority_tier, tested?, transition_result, dest_node, distance_to_frontier`.
- Action selection: highest-priority untested action across the entire graph; only after that tier is exhausted globally, drop to the next tier.
- Performance after post-eval bug fix: **17/25 private levels median**, range 14–19. Author: this *approaches the limit of brute-force* on the preview games.

### 3.4 RGB-Agent (Read-Grep-Bash, alexisfox7)

- LLM-orchestrated agent (Claude/GPT/Gemini, optional local Qwen).
- OpenCode-style Docker sandbox; analyzer reads the prompt log with Read/Grep/Python and outputs a JSON action plan; action queue drains one per step.
- Best public action efficiency: **1,069 actions for all 3 preview games** (vs others using thousands).
- ⚠ requires API; **not directly portable to Kaggle's offline eval**, but its architecture (analyzer + action queue + score-change flush) is portable to any "small local LLM" backend.

## 4. Reference research papers (selected, by relevance to ARC-AGI-3)

| Paper | Why relevant |
| ----- | ------------ |
| Hafner et al., **DreamerV3** (Nature 2025; arxiv 2301.04104) | RSSM world model + actor-critic in imagination; few hyperparameters; strong on diverse domains, novel ones. Code: `danijar/dreamerv3` (JAX), `NM512/dreamerv3-torch` (PyTorch, ~5×faster `r2dreamer` follow-up). |
| Sekar et al., **Plan2Explore** (ICML 2020) | Self-supervised exploration via planning to expected future novelty in a learned world model. Almost matches reward-aware oracle. |
| Ermolov & Sebe, **Latent World Models for Intrinsically Motivated Exploration** (2020) | Latent representations preserving temporal distance + novelty via predictive forward error; strong on Atari hard-exploration. |
| Liao & Gu, **CompressARC** (2025, ARC Prize 2025 paper award) | 76K-param VAE-style network, MDL-guided test-time training; 20% on ARC-AGI-1. Demonstrates that *zero-pretraining* tiny networks can score significantly. |
| Sorokin & Puget (NVIDIA), **NVARC** (1st place ARC Prize 2025) | 4B model + heavy synthetic data + test-time training; 24.03% private LB on ARC-AGI-2. Code: `1ytic/NVARC`. |
| McGovern, **Test-time Adaptation of Tiny Recursive Models** (arxiv 2511.02886) | Pre-train tiny RNN on ARC tasks (700K steps on 4×H100), then 12,500-step full FT during competition → 6.67% on semi-private ARC-AGI-2. |
| Chollet et al., **ARC Prize 2025 Technical Report** (arxiv 2601.10904) | Frames "the refinement loop" as the dominant 2025 theme — per-task iterative program optimization with feedback. |
| Rudakov et al., **Graph-Based Exploration for ARC-AGI-3** (arxiv 2512.24156, AAAI'26 workshop) | The 3rd-place dev-preview agent paper — sets the bar for *training-free* methods. |

## 5. Synthesis — what works, what doesn't (yet)

**Works:**
- Frame hashing + state-graph dedup → essential for efficiency.
- Predicting "will this action change the frame?" → cuts the click-grid search space drastically.
- Per-level model/buffer reset → prevents forgetting/transfer-degradation across novel mechanics.
- Frame segmentation + saliency tiers → the cheapest, biggest win for ACTION6 click targets.

**Doesn't (yet):**
- Direct LLM-as-policy (Claude/GPT) — performs worse than dedicated brute-force search, both in raw level-completion and (especially) in actions used.
- Long-context "show all frames as images" — the per-game token budget explodes past 100s of steps.
- Pure unsupervised RL from scratch (DQN/PPO) — sample inefficiency vs the 200k-step budget per game is brutal.

**Open ground (high potential, low public score so far):**
- **DreamerV3-lite / Plan2Explore** for ARC-AGI-3 — almost zero public attempts (only one notebook "DreamerV3 + ICM" at 0.06).
- **Object-centric / slot-attention world models** + **DSL synthesis** combo — hybrid neural-symbolic.
- **Test-time training with a tiny recursive model** ported from ARC-AGI-2 (TRM family) — adapted to interactive trajectories instead of static IO pairs.
- **Refinement loop** style (paper award 2025) inside the level: synthesize a candidate program from level-1 transitions; refine when it fails.
- **Library learning (Stitch / DreamCoder)** that compresses repeated subroutines from level-1 into reusable primitives for level-2+.

## 6. Compute envelope for our notebook

- Kaggle CPU notebook ≤ 6h, GPU notebook ≤ 6h, **H100 ≤ 6h** (only on this competition).
- 110 private games × roughly 6 levels each ≈ 660 level-attempts. If we average 200 actions/level, that's 132,000 actions per submission.
- Per-action wall-clock budget: 6h / 132,000 ≈ **160 ms / action**, including model forward, plan, env step.
- **H100 throughput at 16-channel 64×64 CNN**: easily >5,000 fwd-pass/s at 4-layer 256-channel → forward is free; the bottleneck is search and env interaction.
- **Memory**: a 200K-entry hash buffer × ~5KB/entry ≈ 1 GB. Trivial on H100. CPU sustained.

## 7. Attack plan ranking (best ROI first)

1. **Baseline replay** (Stochastic Goose pinned, Just Explore pinned) → confirms infra.
2. **Trigger-aware BFS + CNN change predictor (D5 in PLAN.md)** → fastest path to 0.35.
3. **Frame seg + clickable-object library (D8)** → unlocks 0.43+ ceiling.
4. **MCTS with neural prior (D9)** → 0.45 territory.
5. **DreamerV3-lite world model (D10)** → 0.45 baseline + better long-horizon games.
6. **Test-time-trained tiny recursive model (D11)** → first real shot at 0.50+.
7. **DSL/program library (D12)** + **Slot-attention world model (D13)** → compositional generalization to deep levels.
8. **Ensemble (D15)** → final push toward 0.70+.
