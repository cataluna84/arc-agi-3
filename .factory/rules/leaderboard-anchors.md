# Leaderboard Anchors

## Our baseline

**LB 0.19** — vanilla fork of *Ash's ARC-AGI-3 Agent*, rank **398** (submitted 2026-04-29). All score deltas in `.factory/memories.md` are measured against this number.

Notable: a **−0.23 reproduction gap** vs Ash's advertised public 0.42. See `experiments/exp001_baseline_ash/README.md` and `experiments/exp002_ash_variance_probe/README.md`.

## Public LB top (as of 2026-04 snapshot)

| Rank | Team | Score | Notes |
| --- | --- | ----- | ----- |
| 1 | Redfield Rentals | 0.68 | top of LB |
| 2 | Barada Sahu | 0.66 | |
| 3 | Kevin E R MILLE | 0.66 | |
| 4 | SVG | 0.65 | |
| 5 | Matthew Philip Poetker | 0.64 | |
| ... | ... | ... | ... |
| 21 | Ali | 0.43 | |
| 22 | ashvin singh | 0.42 | author of "Ash's ARC-AGI-3 Agent" notebook (0.42) |

## Public-notebook score landmarks (must-read references)

- **Stochastic Goose sample submission** = 0.25 (*official baseline*)
- **Random Agent** = 0.18
- **Just Explore (graph)** = 0.19
- **Ash's ARC-AGI-3 Agent** = 0.42 advertised public, **but our reproduced fork = 0.19** (rank 398) — locked in as our baseline anchor
- **FORGE ARC-AGI-3 Agent** = 0.39 (124 upvotes — Gold)
- **FORGE v16 Trigger-aware BFS** = 0.35 (90 upvotes)
- **Hybrid Search-and-Learn** = 0.35
- **StochasticGoose++ CNN** = 0.32
- **Redpill Zero-Prior Latent Planning** = 0.30 (52 upvotes)
- **MCTS Solver** = 0.29
- **memoryAgent** = 0.28
- **Cognitive-Rungs** = 0.21

> The gap from `random=0.18` → `stochastic=0.25` → `BFS-aware=0.35` → `Ash 0.42` shows clear stair-stepping. The top private LB at 0.68 likely combines **search + neural change-prediction + program library reuse**.
