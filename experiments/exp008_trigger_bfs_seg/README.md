# exp008 - Trigger-BFS + Frame Segmenter (D10+D11)

Track J (new) of `experiments/SPEC_4WEEKS.md`. Port of
[dolphin-in-a-coma/arc-agi-3-just-explore](https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore)
(3rd place ARC-AGI-3 Preview Challenge; 12-19 levels on the private LB;
MIT licensed). The algorithm is published as
[Rudakov et al., arXiv:2512.24156](https://arxiv.org/abs/2512.24156).

## What this submission contains

The comp notebook inlines three modules (source-of-truth in `agents/`):

1. **`agents/state_graph.py`** — `StateGraph`, `StateNode`, `hash_frame`
   (shared with trigger_bfs / goose_cnn).
2. **`agents/frame_segmenter.py`** (NEW D10) — per-color
   connected-components flood-fill + 5-tier saliency + status-bar
   detection (line + dot rules).
3. **`agents/trigger_bfs_agent.py`** with the D11 wire-up — the
   ACTION6 click-coord sampler now walks the segmenter's priority
   tiers 0..3 (skipping tier 4 status bars), picks a non-dominant
   segment within each tier, then samples a uniform pixel within that
   segment.

The MyAgent class adopts the `master_v7`-style defensive pattern:
- `enable_gpu=false`
- `try/except` wrapping the whole `choose_action()` body
- defensive fallback returning a random non-RESET non-ACTION6 action

## Algorithm (paper Algorithm 1, summarized)

The frame processor:
1. Segment the 64x64 frame into single-color connected components.
2. Identify status bars by two rules:
   - line: segment fully against an edge AND aspect ratio >= 5:1;
   - dot: segment against an edge AND has >= 3 same-shape twins also
     on the same edge.
3. Stratify segments into 5 priority tiers based on (size, salience,
   status-bar membership). Tier 0 is "salient color + medium width"
   (most likely interactive); tier 4 is "probable status bar" (least
   likely interactive).
4. Hash a status-bar-masked version of the frame (16-byte blake2b
   with shape personalization) so that frames differing only in step
   counter collide to the same hash.

Our wire-up in `_sample_click_xy`:
```python
for tier_idx in range(4):
    tier_sids = [sid for sid in tiers[tier_idx]
                 if segments[sid].area <= half_frame]
    if not tier_sids:
        continue
    chosen_sid = rng.choice(tier_sids)
    coord = mask_to_click_coords(label_map, chosen_sid, rng=rng)
    if coord is not None:
        return {"x": coord[0], "y": coord[1]}
# fallthrough: legacy non-bg sampler → uniform [0, 63]^2
```

## What this DOES NOT include (deferred to next iteration)

The paper's full method also has a **Level Graph Explorer** that
implements a hierarchical action-selection policy (Algorithm 1):
within the current priority threshold p, prefer untested actions in
the current state; if none, pick the action minimizing shortest-path
distance to a state with untested actions at priority <= p; if none,
increment p. Our exp008 uses the **existing** trigger_bfs hierarchical
selection (untried-in-current-state, then highest-change-edge) for
the action choice itself. The segmenter is used only for the ACTION6
click-coord prior. The full GraphExplorer port is a larger change and
should be a separate exp (call it D12+).

## Smoke status (2026-05-13)

- `uv run pytest tests/test_frame_segmenter.py`: **11/11 PASS**
- `uv run pytest`: **54/54 PASS**
- `uv run python scripts/trigger_bfs_smoke_local.py`: **22/22 PASS**
  (seed=1, max-actions=400 on ls20-mock; agent WINs in 237 actions)
- `uv run ruff check .` + `ruff format --check .`: clean (26 files)
- Multi-seed local_runner ls20-mock: seeds 1, 2, 3 all WIN in 199-237
  actions; seed 0 GAME_OVERs at action 100 (mock's MAX_ACTIONS_PER_LEVEL
  cap, not a runner cap).

## Submission

| When               | Slug                                                | Status   | LB |
|--------------------|-----------------------------------------------------|----------|-----|
| 2026-05-13 12:29 UTC | `cataluna84/trigger-bfs-segmenter-comp-arc-agi-3` v1 | PENDING | TBD |

(Kaggle auto-slugged the title "Trigger BFS + Segmenter comp
ARC-AGI-3" to `trigger-bfs-segmenter-comp-arc-agi-3` instead of our
configured `trigger-bfs-seg-comp-arc-agi-3`. Both `kernel-metadata.json`
id and the local docs now use the actual slug.)

## Decision rule (after LB result lands)

- LB ≥ 0.30: faithful port confirmed. Build the full GraphExplorer
  (Algorithm 1) as D12-D13 for the next iteration toward 0.36.
- 0.21 ≤ LB < 0.30: segmenter lifts but our hierarchical selection is
  weaker than the paper's; prioritize the GraphExplorer port.
- 0.10 ≤ LB < 0.21: marginal lift over goose-v2 (0.17) or trigger-bfs
  v1 (0.10). Inspect the action histogram + state graph stats to
  diagnose: probably the ACTION6 prior is correct but the agent never
  reaches click-controlled games (ls20 is arrow-controlled).
- LB < 0.10: regression; the segmenter wire-up is harming the agent
  somehow. Roll back to the legacy non-bg sampler and re-investigate.
