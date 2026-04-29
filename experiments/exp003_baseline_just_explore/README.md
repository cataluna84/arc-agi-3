# exp003 — Baseline: Just Explore

**Day**: D3 (after exp001 anchor and exp002 variance probe)
**Target score**: 0.19 (matches the published Just Explore baseline notebook)
**Submission strategy**: Fork & resubmit unchanged.

---

## Why this exp

Just-Explore scores **lower** than Stochastic Goose (0.19 vs 0.25) but its agent loop is the most readable in the public notebook set. Forking it gives us:

1. A second confirmed pipeline (independent from exp001).
2. A reference **graph-state** implementation we can lift wholesale into exp004 (Trigger-Aware BFS).
3. The simplest available `Agent` class to use as a template for our own agents in `agents/`.

## Hypothesis

| Var             | Expected | Tolerance |
| --------------- | -------- | --------- |
| Public LB score | 0.19     | ±0.02     |
| Wall-clock      | < 3 h    | n/a       |

## Steps

1. On Kaggle, fork `ARC3 Sample Submission - Just Explore`.
2. Confirm dataset + wheels attached, internet OFF.
3. Run all → submit.
4. Save `score.txt` and `notebook_url.txt`.
5. Append row to `.factory/memories.md`.

## Definition of done

- [ ] Submission accepted; LB row updated.
- [ ] Score in 0.16–0.22.
- [ ] We have read the `Agent.choose_action` and `Agent.update` methods top-to-bottom and noted the data structures used (state graph, frontier, visited set).
- [ ] Notes added on any pieces we plan to reuse for exp004.

## Reuse plan

- Lift the **state-graph dict** verbatim for exp004 (rename if needed).
- Steal the **`reset_on_level_complete`** helper (we need this in every later experiment).
- Reuse the **action-int → action-name** map.
