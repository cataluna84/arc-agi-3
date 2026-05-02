# exp005 - Trigger-Aware BFS Agent (D4 of D3-D28 SPEC)

**Submitted as an ablation data point**, knowing the result is likely a
regression vs FORGE 0.24. See `experiments/SPEC_4WEEKS.md` for the
broader plan; the choice to submit anyway was per user direction
(Option C of the D4 morning AskUser checkpoint).

## What this is

A pure state-graph + uniform-random-over-untried agent. No CNN, no
torch, no neural model. Built on `agents.state_graph.StateGraph` (also
new today) which provides the standard primitives (frame hashing,
node/edge graph, frontier, level-transition reset) used by every
later agent in the SPEC.

## What we expect

- LB: 0.18 - 0.22 (matches public Random Agent 0.18 anchor; slight lift
  from state-graph dedup avoiding redundant retries).
- Local sweep at 1000 actions per game: 1/25 games clear level 1, same
  as the RandomAgent baseline (different game though). Most games hit
  GAME_OVER from internal mechanics, not action budget.

## Why we still submit

1. **LB ablation**: we want to confirm whether state-graph dedup alone
   moves the needle vs random. This is the cleanest A/B we will ever
   get; later submissions will combine many components.
2. **Foundation**: `state_graph.py` is now reusable for D7-D27 of the
   SPEC. Even if its first agent flops, the module is the entry point
   for future search-based variants.
3. **Slot-accounting honesty**: we said D4 = submit. We submit. If the
   number is bad we record it and pivot per `experiments/SPEC_4WEEKS.md`.

## Files

```
experiments/exp005_trigger_aware_bfs/
├── README.md            (this file)
├── dev_kernel/          (placeholder; we go straight to comp_kernel)
└── comp_kernel/
    ├── kernel-metadata.json
    └── trigger_bfs_comp.ipynb   (4-cell, mirrors the FORGE structure)
```

The Kaggle kernel slug is `cataluna84/trigger-bfs-comp-arc-agi-3`.

## Smoke status (2026-05-02)

- `uv run pytest tests/test_state_graph.py`: **5/5 PASS**
- `uv run python scripts/trigger_bfs_smoke_local.py`: **22/22 PASS**
- `uv run python experiments/local_runner.py --agent agents.trigger_bfs_agent:TriggerBFSAgent --use-sdk --games <25-game sweep> --max-actions 1000`:
  - 1 of 25 games achieves level 1 (ft09 in 116 actions).
  - All other games end with GAME_OVER, no levels.
- For comparison: `agents.random_agent:RandomAgent` on the same sweep
  achieves 1 of 25 (r11l). Net parity.

## Decision rule (after LB result lands)

- If LB > 0.24: state-graph dedup is genuinely helping; keep building
  on it (per SPEC §1.5 D7+).
- If 0.18 < LB < 0.24: state graph helps modestly but isn't enough to
  beat FORGE alone. Add BFS replay (SPEC D7) to lift further.
- If LB < 0.18: regression -- something in the agent is worse than
  random. Diagnose via dev kernel logs.
