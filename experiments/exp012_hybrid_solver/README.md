# exp012_hybrid_solver

**D21 (2026-05-19) submission candidate**: fork of the public Kaggle notebook
[ARC-AGI-3 Hybrid Solver (BFS + CNN + Heuristics)](https://www.kaggle.com/code/vyankteshdwivedi/arc-agi-3-hybrid-solver-bfs-cnn-heuristics)
by Vyankatesh Dwivedi.

## What's in it

The notebook calls itself **"MASTER BASELINE v10"** (per the header in cell 1)
and is described as a merge of 6 top public notebooks:

- **CORE**: FORGE v19 (op_2) — A* search + game introspection heuristic,
  transient-field detection, dynamic action-rescan BFS, object-model tracking
- Beam search (FORGE v17)
- MCTS click masking
- grad_clip, intrinsic_reward
- ~78KB of inlined agent code in one `%%writefile /kaggle/working/my_agent.py`
  cell

## Structure

Identical to our `exp008_trigger_bfs_seg/comp_kernel` cell layout:

1. Cell 0: `!pip install --no-index --find-links .../arc_agi_3_wheels arc-agi python-dotenv`
2. Cell 1: `%%writefile /kaggle/working/my_agent.py` with the inlined agent
3. Cell 2: competition_rerun guard that copies the ARC-AGI-3-Agents harness
   and runs `python main.py --agent myagent`
4. Cell 3: dummy `submission.parquet` fallback for save-mode

## Metadata

| Field | Value |
|---|---|
| Source kernel | `vyankteshdwivedi/arc-agi-3-hybrid-solver-bfs-cnn-heuristics` |
| Our slug | `cataluna84/hybrid-solver-v10-comp-arc-agi-3` |
| enable_gpu | false (CPU-only, BFS-heavy) |
| enable_internet | false |
| competition_sources | `["arc-prize-2026-arc-agi-3"]` |
| dataset_sources | none |

## Expected LB

The source notebook's public LB cited in our `.factory/rules/leaderboard-anchors.md`
is **0.39** (under "FORGE ARC-AGI-3 Agent" / Hybrid Solver family).
Conservative band for our fork: **0.25-0.39**, depending on whether the
notebook's evaluation environment matches ours exactly.

Our existing best: 0.24 (D3 FORGE variance probe). Any LB result above
0.24 is a strict improvement on the 6 weeks of work since D3.

## Licensing

Public Kaggle code competition notebook. Per the source notebook's
`kernel-metadata.json`, no license field is set — Kaggle's default for public
code competition notebooks is permissive (Apache-2.0 unless overridden by
the author). ARC Prize 2026 rules require 3rd-party code under permissive
open-source licenses; this qualifies.

We attribute the work in CHANGELOG.md and the source URL above. If the
notebook lifts our LB, an open-source disclosure under the same permissive
terms will accompany our prize-eligible submission (per ARC Prize milestones).

## Process

1. Copied `arc-agi-3-hybrid-solver-bfs-cnn-heuristics.ipynb` from the source
   kernel to `comp_kernel/hybrid_solver_v10_comp.ipynb` unchanged.
2. Created our own `kernel-metadata.json` with our slug.
3. `kaggle kernels push -p experiments/exp012_hybrid_solver/comp_kernel/`
   (no `--accelerator` flag — defaults to P100 / no GPU, fine for BFS).
4. Save-mode verification.
5. Confirm with user before competition submit (per saved feedback rule:
   `feedback_kaggle_slot_confirmation.md`).

## Risks

- **License ambiguity**: if the source notebook's actual licensing is more
  restrictive than Kaggle's default, our fork may not be prize-eligible.
  For a non-prize LB probe, this is acceptable; for the September 2026 prize
  milestone, we may need to either replace the code or get explicit
  permission from the source author.
- **0.39 may be cherry-picked**: the source LB number is one run; ours might
  land in a wider band (e.g., 0.25-0.35) depending on Kaggle's per-game seed
  variance.
- **Save-mode dependencies**: cell 0 installs `arc-agi` from the competition
  wheels mount; if the wheels are mounted at a different path on the rerun
  host (gotcha #11), the install could fail. The notebook author's 0.39
  result suggests this is robust on the current Kaggle image, but worth
  watching during save-mode poll.
