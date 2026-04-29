# exp002 — Ash variance probe

**Goal**: figure out whether 0.19 (rank 398, our exp001 baseline) is the real expected score for the unchanged Ash fork or just a low-variance draw from a higher-mean distribution.

**Decision rule** (from `.factory/plan.md` D1-D2):
- If `max(s2, s3) >= 0.30`: variance-dominated → exp004 will be a best-of-N seed sweep using Ash's notebook (free LB lift)
- If `max(s2, s3) < 0.25`: structural ceiling near 0.19 → pivot to FORGE Trigger-Aware BFS or Tier 4 Qwen agent

## Procedure (1 daily slot per resubmit)

```bash
# Step 1: pull our existing Ash fork to a clean folder
.venv/bin/kaggle kernels pull cataluna84/ash-s-arc-agi-3-agent -p experiments/exp002_ash_variance_probe/_pulled/ -m

# Step 2: push it back unchanged - this creates a new version and runs it
KAGGLE_API_TOKEN="$KAGGLE_KEY" .venv/bin/kaggle kernels push -p experiments/exp002_ash_variance_probe/_pulled/

# Step 3: poll status until COMPLETE (~15-25 min for Ash on the full game set)
.venv/bin/kaggle kernels status cataluna84/ash-s-arc-agi-3-agent

# Step 4: when COMPLETE, find the new version number from the push output, then submit:
.venv/bin/kaggle competitions submit-code \
  -c arc-prize-2026-arc-agi-3 \
  --kernel cataluna84/ash-s-arc-agi-3-agent \
  --kernel-version <NEW_VERSION_NUMBER> \
  -f submission.parquet \
  -m "exp002 variance probe - Ash unchanged, daily resubmit #N"
```

> **WARNING**: each `submit-code` consumes one of your daily submission slots. The CLI does NOT prevent re-submissions; the server enforces the cap.

## Logging

After each LB result lands, append to `.factory/memories.md` (top of file) a section:

```
## 2026-MM-DD - exp002 day N
- Score: 0.XX (delta vs 0.19 = +Y.YY)
- Per-game: <if visible from scorecard>
- Wall clock: ~<minutes>
- Decision: continue / pivot
```
