# exp011 - MCTS over Trigger-BFS State Graph (Option B)

Independent build alongside exp004 Path B Qwen Phase-1 (Option A). Layers
UCB1-based selection on top of the existing `agents.state_graph.StateGraph`
to provide a principled exploration/exploitation balance over `(action_id,
click_bucket)` keys.

## Hypothesis

Trigger-BFS v1 (exp008) scored 0.12 on LB. The diagnostic was: the segmenter
provides a strong click-coord prior, but the action selector is too greedy
- once a state has been "explored" it gets stuck cycling without enough
sample-efficient breadth. MCTS with UCB1 should:

1. Force first-time exploration of every (action, bucket) combo at each
   state (Q = +inf for unvisited),
2. Then concentrate visits on the highest-Q edges (where Q = mean trigger
   `change_score`),
3. Without abandoning lower-Q edges entirely (UCB1 exploration term).

## Algorithm

- **State**: same `agents.state_graph.StateGraph` keyed by `hash_frame`.
- **Action key**: `(action_id, click_bucket)`. For non-ACTION6,
  bucket = -1. For ACTION6, bucket = index into a deterministic candidate
  list (segmenter top-K, then 5 geometric fallbacks, then a uniform-random
  sentinel - total ≤ 14).
- **Selection**: UCB1 across all enumerated keys at the current state node.
  `c = 2.0` by default.
- **Expansion + simulation + backup**: one env step per call. The next call
  computes `change_score = trigger_score(prev, cur, prev_levels, cur_levels)`
  and updates `(prev_hash, prev_action_key)` stats.
- **Level transitions**: clear state graph + MCTS stats (gotcha #4).

## Files

- `agents/trigger_bfs_mcts_agent.py` - the agent + UCB1 primitives.
- `tests/test_trigger_bfs_mcts.py` - 13 pytest cases.
- `scripts/trigger_bfs_mcts_smoke_local.py` - 22-check parity smoke.
- `comp_kernel/build_notebook.py` - inlines state_graph + frame_segmenter
  + the agent into a self-contained `my_agent.py` for the Kaggle harness.
- `comp_kernel/kernel-metadata.json` - slug
  `cataluna84/trigger-bfs-mcts-comp-arc-agi-3`, CPU-only, no internet,
  no dataset_sources.

## Expected LB band

0.18 - 0.30. Decision rule (post-submit):

- `LB >= 0.25`: MCTS+state-graph compounds → next step is per-action
  prior temperatures or learnable Q.
- `0.18 <= LB < 0.25`: UCB1 helps modestly; investigate whether
  `c_uct` is too high (over-explores) or too low (premature commit).
- `0.12 < LB < 0.18`: marginal lift over Trigger-BFS v1; consider whether
  the click-bucket discretization is too coarse.
- `LB <= 0.12`: regression; roll back to Trigger-BFS v1.

## Smoke status

Run from the repo root:

```bash
uv run pytest tests/test_trigger_bfs_mcts.py -v
uv run python scripts/trigger_bfs_mcts_smoke_local.py
uv run python experiments/exp011_trigger_bfs_mcts/comp_kernel/build_notebook.py
```

Submission is held until the parent agent green-lights the slot.
