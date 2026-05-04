# exp007 - Goose CNN Agent (Track G)

Track G of `experiments/SPEC_4WEEKS.md`. Ports
[StochasticGoose](https://github.com/DriesSmit/ARC3-solution) (Tufa Labs,
1st place ARC-AGI-3 Developer Preview, **12.58% private LB**, public
Kaggle sample anchor **0.25**).

## Architecture

- **Input**: 16-channel one-hot encoded 64x64 grid (one channel per palette
  color). Preserves discrete palette structure as 2D feature maps.
- **Backbone**: 4-layer CNN, channels 32 -> 64 -> 128 -> 256, stride 1
  throughout (preserves 2D inductive bias for the coord head).
- **Two heads**:
  - **Action head**: global avg pool over 256x64x64 -> 256-d vector ->
    linear to 5 logits (sigmoid; one per ACTION1..ACTION5).
  - **Coord head**: 1x1 conv on 256x64x64 -> 1x64x64 (sigmoid; per-pixel
    P(ACTION6 click changes frame)).
- **Loss**: BCE on observed (state, action, frame_changed) tuples + light
  entropy regularization.
- **Experience buffer**: hash-deduped, capped at 50k entries. Upstream
  uses 200k; we conserve memory for the 6h Kaggle wall budget.
- **Per-level reset**: model + buffer wiped on level transition (gotcha
  #4 -- different levels can have radically different mechanics).

## What this adds over StochasticGoose 0.25

1. **State graph dedup** (shared with `agents/trigger_bfs_agent.py`):
   skip already-tried (state, action) pairs entirely, so the action
   budget is spent on novel exploration.
2. **Non-background coord prior**: when the CNN coord head is cold, ACTION6
   coords are restricted to non-background pixels (saliency-tier 0).
   Once the CNN trains it can re-rank within the non-bg set.
3. **Empirical change-score boost**: when all actions are tried at a
   state, the agent prefers actions that previously caused frame deltas
   (mirrors trigger_bfs's "highest-change edge" tie-breaker).

## Why we expect a lift

- **0.25 baseline** is for vanilla random sampling biased by CNN
  predictions. Adding state-graph dedup + non-bg prior reduces wasted
  no-op actions in the 100-step-per-level budget.
- **0.30+ target** would put us in the top 200; lift from non-bg prior
  alone is probably 0.02-0.05 based on trigger_bfs's expected 0.30-0.35
  range.

## Files

```
experiments/exp007_goose_cnn/
├── README.md            (this file)
├── dev_kernel/          (placeholder; we may use one if comp kernel is flaky on H100)
└── comp_kernel/
    ├── kernel-metadata.json
    └── goose_cnn_comp.ipynb   (4-cell, mirrors FORGE/trigger_bfs structure)
```

The Kaggle kernel slug is `cataluna84/goose-cnn-comp-arc-agi-3`.

## Smoke status (2026-05-04)

- `uv run pytest tests/test_goose_cnn.py`: **14/14 PASS**
- `uv run python scripts/goose_cnn_smoke_local.py`: **22/22 PASS**
- `uv run python experiments/local_runner.py --agent agents.goose_cnn_agent:GooseCNNAgent --games ls20-mock --max-actions 300 --seed {0..3}`:
  - 4 / 4 wins (21 - 58 actions per game).

## Decision rule (after LB result lands)

- If LB > 0.30: Track G is genuinely lifting; add to the master kernel
  ensemble (per SPEC §1.5 D7+).
- If 0.25 < LB < 0.30: matches StochasticGoose anchor; iterate on the
  CNN architecture and online-training schedule.
- If LB < 0.25: regression vs anchor. Diagnose via the dev kernel logs;
  likely culprits are (a) ACTION6 coord head not training, (b) model
  weights not actually being reset between levels, or (c) buffer cap
  too small for the longer levels.
