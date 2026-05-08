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

## Smoke status (2026-05-07 v2)

- `uv run pytest tests/test_goose_cnn.py`: **14/14 PASS**
- `uv run pytest`: **40/40 PASS** (incl. test_qwen_backbone, test_state_graph)
- `uv run ruff check .` + `uv run ruff format --check .`: clean (24 files)
- `uv run python scripts/goose_cnn_smoke_local.py`: **22/22 PASS** (incl.
  `mock end-to-end: 1 win on ls20-mock in 218 actions`).
- `uv run python experiments/local_runner.py --agent agents.goose_cnn_agent:GooseCNNAgent --games ls20-mock --max-actions 100 --seed 0`:
  completed cleanly; action distribution diverse
  (ACTION6=20 / ACTION1=16 / ACTION5=16 / ACTION3=14 / ACTION4=11 / ACTION2=11).

## v1 result (2026-05-04 D6) — LB 0.00, postmortem

D6 submission: `cataluna84/goose-cnn-comp-arc-agi-3` v1, submitted
2026-05-04 15:23 UTC. Kernel status COMPLETE. LB = **0.00**.

The downloadable kernel log shows only the 24s dev-mode `nbconvert` save,
not any agent execution, and the kernel's downloadable `submission.parquet`
contains only the dummy fallback row. Comp rerun logs are server-private
for code competitions, so the failure mode was inferred from comparing
the v1 kernel-metadata + agent code against `master_v7` (LB 0.21) and
`trigger_bfs` (LB 0.10).

| # | Hypothesis | Severity | Decision |
|---|---|---|---|
| 1 | `enable_gpu=true` differs from every kernel that scored >0 | HIGH | flip to `false` in v2 |
| 2 | `predictor.predict()` had no `try/except` | HIGH | wrap in try/except, fallback to uniform priors |
| 3 | `choose_action()` had no outer `try/except` | HIGH | wrap whole body, fallback to random non-RESET non-ACTION6 |
| 4 | Level reset triggered on transient `levels_completed=-1` | LOW | guard `cur_levels >= 0` |
| 5 | `action.set_data` API mismatch | INFO | API audited identical to random/trigger_bfs/master_v7; not the bug |
| 6 | Import chain `agents.templates.my_agent → agents.agent` | INFO | master_v7 uses identical chain; not the bug |

See `.factory/rules/gotchas.md` #17, #19 and `.factory/memories.md` D9
section for the full architectural diff.

## v2 fix (2026-05-07 D9)

```python
# 1. comp_kernel/kernel-metadata.json
"enable_gpu": false  # was true

# 2. choose_action outer guard
def choose_action(self, frame):
    try:
        return self._choose_action_inner(frame)
    except Exception as exc:
        logger.warning("graceful fallback: %r", exc)
        avail = list(getattr(frame, "available_actions", []) or [1, 2, 3, 4, 5])
        non_reset = [int(a) for a in avail if int(a) != 0 and int(a) != 6]
        if not non_reset: non_reset = [1, 2, 3, 4, 5]
        return GameAction.from_id(self._rng.choice(non_reset))

# 3. predictor.predict guard (uniform priors on torch/CUDA failure)
try:
    preds = self.predictor.predict(cur_grid)
except Exception:
    ap = np.full((NUM_SIMPLE_ACTIONS,), 0.5, dtype="float32")
    cp = np.full((GRID_SIZE, GRID_SIZE), 0.5, dtype="float32")

# 4. defensive level-transition guard
if cur_levels >= 0 and cur_levels != self._prev_levels:
    self._on_level_transition()
```

D9 submission: `cataluna84/goose-cnn-comp-arc-agi-3` v2, kernel COMPLETE
in ~40s on CPU (vs the failed GPU init). Submitted via
`kaggle competitions submit arc-prize-2026-arc-agi-3 -k cataluna84/goose-cnn-comp-arc-agi-3 -v 2 -f submission.parquet`
at 2026-05-08 00:23 UTC. Status: PENDING.

## Decision rule (after v2 LB result lands)

- If LB ≥ 0.20: v1 0.00 was a packaging bug, confirmed. Move forward to
  D10 frame-segmentation port (per `arXiv:2512.24156`) and consider
  Goose-as-ensemble-prior for master kernel.
- If 0.10 ≤ LB < 0.20: agent runs but priors do not lift over random;
  retrain CNN with longer warm-up or restrict to ACTION6 click prior.
- If LB ≤ 0.05: still failing silently. Compare cell-by-cell against the
  StochasticGoose 1st-place repo (`DriesSmit/ARC3-solution`, 12.58%
  private LB) for additional packaging differences (action timeout,
  numpy version, action enum coercion).
