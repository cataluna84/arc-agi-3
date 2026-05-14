# ARC-AGI-3 Mechanics

> Sourced from arXiv 2603.24621 ("ARC-AGI-3 Technical Report") and `docs.arcprize.org`. Treat these as immutable facts about the environment.

## Competition snapshot

| Field | Value |
| --- | --- |
| Platform | Kaggle (Code Competition) |
| Total Prize Pool | **$850,000** |
| Final Submission Deadline | **2026-11-02** |
| Milestone 1 (open-source for prize) | **2026-06-30** |
| Milestone 2 (open-source for prize) | **2026-09-30** |
| Top Score Award (Final) | $40K / $15K / $10K / $5K / $5K |
| Grand Prize (100%) | $700K |
| Daily Submission Limit | **1 / day** (per submitter) |
| Notebook runtime cap | **CPU/GPU ≤ 9 hours** (2026-05-14 Kaggle update) |
| Internet during eval | **DISABLED** |
| Eval set | **110 private games** (50% public LB / 50% private LB) |
| Action grid | 64×64 cells, integer values 0–15 (16 colors) |
| Action set | RESET, ACTION1–ACTION5 (simple), ACTION6 (x,y), ACTION7 (often UNDO) |
| Game states | NOT_FINISHED, WIN, GAME_OVER |
| Scoring | RHAE (Relative Human Action Efficiency), per-level squared, weighted by level index, then averaged across games. Final ∈ [0, 1]. |

**Critical**: All eligible (prize-claimable) submissions must open-source
their solution. Kaggle's updated code requirements also require
notebook-based submissions, internet disabled, automatic submission-file
generation, and freely/publicly available external data (including
pretrained models). The 1/day cap means **every submission must count** —
adopt a "ship daily, learn daily" cadence.

## Frame and action semantics

- **Frame**: 64×64 grid, 16 colors (cell values 0–15), top-left = (0,0). May come as a *frame sequence* (animation) between turns.
- **Action**: discrete; up to 5 simple keys + Undo + 1 click (x,y). Internal reasoning steps **do not count** as actions, only state-affecting submissions count.
- **Win threshold**: humans solve 100%; frontier AI < 1% (per the paper).
- **Public-set:Private-set ratio is INVERTED vs ARC-AGI-2**: 25 public demo / 55 semi-private / 55 fully-private. Public set is a **demo only** — not representative of mechanics.
- **Anti-random guard**: every non-tutorial level must be unsolvable by random play with `P_win ≤ 1/10000` (validated by 1M-step random sweeps).
- **Tutorial level (level 1)**: deliberately easy — **occasional random success is acceptable by design**.
- **Difficulty ramps via composition** of mechanics introduced in earlier levels, not via obscurity.

## Scoring formula (RHAE)

```
per_level         = min(human_actions / agent_actions, 1.0)
per_level_squared = per_level ** 2
per_game          = weighted_avg(per_level_squared, weight = level_index_1based)
final             = mean_over_games(per_game)
```

**Implication**: solving levels with *fewer actions* matters more than solving more levels. Even a level-1 win at 50% efficiency contributes only 0.25.
