# exp015_arc26_v15

**D25 (2026-05-23) submission**: fork of
[jonathanchan/arc26-3-agent-v15](https://www.kaggle.com/code/jonathanchan/arc26-3-agent-v15).

## Pre-submission inspection findings (KNOWN ISSUES)

The notebook is self-described as "Version 3 (placeholder)" in its markdown intro.
Inspection of cell 3 (`%%writefile /kaggle/working/my_agent.py`) revealed:

- **Line 557**: `__init__` has a placeholder comment `# ... (ALL YOUR ORIGINAL
  INIT CODE)` where the FORGE v19 BFS solver / CNN net / replay buffer / level
  tracking initialization should be. **Not pasted in.** Only `run_metrics`
  dict and timing fields get initialized.
- **`choose_action` references undefined methods/attributes**:
  - `s._lvl(lf)` (line 620) — method not defined.
  - `s._raw(lf)` (line 639) — method not defined.
  - `s.cl` (lines 623, 636) — attribute never initialized.
  - `s._bfs_solution`, `s._bfs_step` (line 643) — never initialized.
- **Outer `try/except Exception`** (line 664) catches all of the above and
  returns `random.choice([GameAction.ACTION1])` — list has one element, so
  EVERY action is ACTION1.

Predicted comp behavior: every action is ACTION1 across all 110 games. RHAE
score expected **0.05-0.10**, worse than our 0.12 trigger-bfs floor.

## Why we're submitting anyway

User explicit direction after seeing the inspection. Possible motivations:
- Empirical confirmation that the static analysis is correct.
- Probe how Kaggle's eval handles the always-ACTION1 collapse.
- Validate the `try/except` graceful-fallback claim from gotcha #17.

## Metadata

| Field | Value |
|---|---|
| Source kernel | `jonathanchan/arc26-3-agent-v15` |
| Source markdown self-labeled | **"Version 3 (placeholder)"** |
| Our slug | `cataluna84/arc26-v15-comp-arc-agi-3` |
| enable_gpu | false (CPU-only) |
| enable_internet | false |
| competition_sources | `["arc-prize-2026-arc-agi-3"]` |
| dataset_sources | none |
| Inlined agent | ~26 KB (but `__init__` is a placeholder) |

## Process

1. Pulled via `kaggle kernels pull jonathanchan/arc26-3-agent-v15`.
2. Inspected — flagged broken to user.
3. User re-confirmed submit.
4. Scaffolded exp015 with our slug `cataluna84/arc26-v15-comp-arc-agi-3`.
5. Save-mode push (no slot) for infrastructure verification only.
6. Submitted for D25 per user direction.

## Post-result

LB score will be recorded in `.factory/rules/leaderboard-anchors.md` once it lands.
