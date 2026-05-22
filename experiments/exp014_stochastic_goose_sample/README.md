# exp014_stochastic_goose_sample

**D24 (2026-05-22) submission candidate**: fork of the official Kaggle sample
submission [ARC3 Sample Submission - Stochastic Goose](https://www.kaggle.com/code/inversion/arc3-sample-submission-stochastic-goose)
uploaded by `inversion` (Kaggle's bot account).

## What's in it

The official StochasticGoose CNN-based action-learning agent from Tufa Labs
(Dries Smit, lead; Jack Cole, adviser). 1st-place winner of the ARC-AGI-3
Agent Preview Competition (12.58% on the preview private set). The Kaggle
sample submission scores **LB 0.25** on the public ARC-AGI-3 competition.

## Architecture (per inlined source)

- **CNN with shared backbone** for action and coordinate prediction.
- **16-channel one-hot input** for 64x64 frames.
- **4-layer CNN** (32 → 64 → 128 → 256 channels).
- **Action head**: predicts ACTION1-ACTION5 probabilities.
- **Coordinate head**: predicts 64x64 click-position probabilities for ACTION6
  with 2D inductive bias (convolutional layers, not flattened).
- **Binary classification**: predicts if actions will cause frame changes.
- **Hierarchical sampling**: first action type, then coordinates if ACTION6.
- **Hash-deduped experience buffer** (~200k unique state-action pairs).
- **Dynamic model reset** when reaching new levels.
- **Stochastic sampling** via sigmoid probabilities.
- **Trained on the fly** during the 8-hour eval window (~100k steps available
  per game in the original preview).

## Why this is structurally different from previous experiments

| Property | exp012 (FORGE v19 fork) | exp013 (memoryAgent v6) | **exp014 (StochasticGoose)** |
|---|---|---|---|
| Core | BFS + A* + beam search | Object extraction + click heatmap | **CNN action-change predictor** |
| Click prior | Sprite permutation + MCTS click masking | Click heatmap from past changes | **CNN coordinate head (64x64 conv)** |
| Learning | Random-init CNN (private weights trap) | Memory ring buffer | **On-the-fly RL training** |
| State dedup | Hash + state graph | Frame descriptor v2 | **Hash-deduped experience buffer** |
| Memory scope | Per-level | Cross-game-type | **Reset per level** |
| Code size | 78 KB | 32 KB | **17 KB** |
| Private deps | `forge-pretrained-weights` (FAILED) | none | **none** |

## Metadata

| Field | Value |
|---|---|
| Source kernel | `inversion/arc3-sample-submission-stochastic-goose` (Kaggle official sample) |
| Source LB | **0.25** (publicly cited in our `.factory/rules/leaderboard-anchors.md`) |
| Our slug | `cataluna84/stochastic-goose-sample-comp-arc-agi-3` |
| enable_gpu | true (T4 in source) |
| enable_internet | false |
| competition_sources | `["arc-prize-2026-arc-agi-3"]` |
| dataset_sources | none |
| Inlined agent | ~17 KB, single `%%writefile` cell |

## Expected LB

Conservative band: **0.20-0.27**, with most-likely ~0.23-0.25. Source's 0.25
came from running the agent under the same Kaggle harness we'll be running,
no private dataset deps to miss. **Floor expectation 0.20+ regardless** since
StochasticGoose at random init still beats random play (0.18 anchor).

If we land 0.25+ that's a new best (+0.01 over D3 FORGE 0.24).
If we land 0.18-0.19 it means the on-the-fly RL training is unstable in our
re-roll, and the underperformance pattern from prior forks repeats.

## Inspection (D24 pre-push)

- Imports: `hashlib, logging, os, random, time, traceback, collections.deque,
  datetime, typing, numpy, torch (+nn, +nn.functional, +optim),
  agents.agent.Agent, arcengine.{FrameData, GameAction, GameState}`.
  All stdlib or Kaggle-image-shipped.
- **Zero `forge-pretrained-weights` references** — the dependency trap that
  killed exp012 / exp013 is absent here.
- **Zero hardcoded paths** to `/kaggle/input` other than the standard
  competition wheels mount.
- **Zero hardcoded game-specific data** (unlike Resonance Agent which we
  rejected for embedding hardcoded gate (x,y) coords for all 25 public games).
- Same harness wiring (`gateway:8001` + `cp ARC-AGI-3-Agents` + `python main.py
  --agent myagent`) as exp008 / exp012 / exp013.
- `class MyAgent(Agent)`, standard dummy-submission-parquet fallback.

## Attribution + license

This is the **official Kaggle sample submission** uploaded by the Kaggle bot
account `inversion`. Per Kaggle's policies, sample submissions for code
competitions are freely re-usable under permissive licensing. ARC Prize 2026
rules require 3rd-party code under permissive open-source licenses; the
StochasticGoose source repo (`github.com/DriesSmit/ARC3-solution`) is
explicitly MIT-licensed.

Attribution preserved in this README and the source URLs above.

## Process

1. `kaggle kernels pull inversion/arc3-sample-submission-stochastic-goose`
2. Created `experiments/exp014_stochastic_goose_sample/comp_kernel/` with our
   slug `cataluna84/stochastic-goose-sample-comp-arc-agi-3`.
3. `kaggle kernels push` → save-mode verification.
4. **Confirm with user before competition submit** (per saved feedback rule:
   `feedback_kaggle_slot_confirmation.md`).
