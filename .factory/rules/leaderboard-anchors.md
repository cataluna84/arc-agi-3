# Leaderboard Anchors

## Our baseline

**LB 0.19** — vanilla fork of an upstream public Kaggle notebook
implementing FORGE v19, rank **398** (submitted 2026-04-29). All score
deltas in `.factory/memories.md` are measured against this number.
See `NOTICE` for upstream attribution.

Notable: a **−0.23 reproduction gap** vs the upstream notebook's
advertised public 0.42. See `experiments/exp001_baseline_forge/README.md`
and `experiments/exp002_forge_variance_probe/README.md`.

## Our scoreboard (running)

| Day | Date       | Submission                                  | LB    | Δ vs 0.19 |
| --- | ---------- | ------------------------------------------- | ----- | --------- |
| D1  | 2026-04-29 | exp001 baseline FORGE v19 (vanilla fork)    | 0.19  | +0.00     |
| D2  | 2026-04-30 | exp002 FORGE variance probe s2              | 0.00  | -0.19     |
| D3  | 2026-05-01 | exp002 FORGE variance probe s3              | 0.24  | +0.05     |
| D4  | 2026-05-02 | exp005 trigger-bfs v0 (no segmenter)        | 0.10  | -0.09     |
| D5  | 2026-05-03 | exp006 MASTER v7                            | 0.21  | +0.02     |
| D6  | 2026-05-04 | exp007 Goose CNN v1 (silent crash)          | 0.00  | -0.19     |
| D9  | 2026-05-07 | exp007 Goose CNN v2 (defensive + CPU)       | 0.17  | -0.02     |
| D15 | 2026-05-13 | exp008 trigger-bfs + frame-segmenter        | 0.12  | -0.07     |
| D16 | 2026-05-14 | exp010 FORGE variance safety resubmit       | ERROR   | -       |
| D17 | 2026-05-15 | exp008 trigger-bfs+segmenter v1 (resubmit)  | 0.12    | -0.07   |
| D18 | 2026-05-16 | (slot expired unused)                        | —       | —       |
| D19 | 2026-05-17 | exp004 Qwen Phase-1 + Path B (masked-hash + tried-clicks-in-prompt) | PENDING | -       |

Best to date: D3 (FORGE variance probe s3) at 0.24. The Goose CNN v2
result (0.17 vs random 0.18) was the moment we pivoted from training
priors → structural priors. Exp008 shows that a segmenter click prior
alone is not enough; the next structural step must include the full
GraphExplorer/action scheduler from the paper.

D16 used the slot on a low-risk FORGE variance resubmit because both
new D16 workstreams (Qwen RTX load path and GraphExplorer prototype)
were still dev-only.

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
| 22 | ashvin singh | 0.42 | author of the upstream FORGE-v19 notebook our exp001 forks |

## Public-notebook score landmarks (must-read references)

- **Stochastic Goose sample submission** = 0.25 (*official baseline*)
- **Random Agent** = 0.18
- **Just Explore (graph)** = 0.19
- **Upstream FORGE-v19 notebook** = 0.42 advertised public, **but our
  reproduced fork = 0.19** (rank 398) — locked in as our baseline anchor
- **FORGE ARC-AGI-3 Agent** = 0.39 (124 upvotes — Gold)
- **FORGE v16 Trigger-aware BFS** = 0.35 (90 upvotes)
- **Hybrid Search-and-Learn** = 0.35
- **StochasticGoose++ CNN** = 0.32
- **Redpill Zero-Prior Latent Planning** = 0.30 (52 upvotes)
- **MCTS Solver** = 0.29
- **memoryAgent** = 0.28
- **Cognitive-Rungs** = 0.21

> The gap from `random=0.18` → `stochastic=0.25` → `BFS-aware=0.35` →
> `FORGE 0.42 advertised` shows clear stair-stepping. The top private LB
> at 0.68 likely combines **search + neural change-prediction + program
> library reuse**.
