# exp010 - FORGE variance safety resubmit (D16)

## Why this was today's submission

D16 produced two promising but not-yet-submittable workstreams:

- Qwen now loads and generates on RTX 6000, but the direct-policy path
  still needs a guarded candidate/ranker loop before it should spend a
  competition slot.
- The GraphExplorer prototype now captures the paper's high-level
  priority-threshold scheduler, but local SDK diagnostics still solve
  **0/25 mounted public games**.

Because the user explicitly wanted a submission today, the safest use of
the D16 slot was a variance/safety resubmit of the best-known completed
kernel: `cataluna84/ash-s-arc-agi-3-agent` v2, which previously scored
**LB 0.24**.

## Brainstormed options

| Option | Expected value | Risk | Decision |
| --- | --- | --- | --- |
| Fresh/resubmitted FORGE variance probe | 0.19-0.24, chance to match high-water mark | Low | **Chosen** |
| MASTER v7 resubmit | Around 0.21 | Low-medium | Keep as backup |
| Goose v2 resubmit | Around 0.17 | Medium-low | Worse than FORGE/MASTER |
| GraphExplorer prototype | Unknown, local 0/25 levels | High | Do not submit |
| Qwen direct policy | Technically loads, policy not ready | Very high | Dev-only |

## Submission record

1. Pulled the completed FORGE kernel into `/tmp/arc_agi3_forge_d16`.
2. Pushed an unchanged v3 rerun:

   ```bash
   uv run kaggle kernels push -p /tmp/arc_agi3_forge_d16
   ```

   Kaggle accepted it as kernel version 3, but it remained queued.

3. To ensure the D16 competition slot was actually used, submitted the
   last completed best-known version:

   ```bash
   uv run kaggle competitions submit arc-prize-2026-arc-agi-3 \
     -k cataluna84/ash-s-arc-agi-3-agent \
     -v 2 \
     -f submission.parquet \
     -m "exp010 D16 (2026-05-14) FORGE variance safety resubmit; best-known completed kernel v2 scored 0.24; fresh v3 queued, using v2 to ensure today's slot is utilized"
   ```

| When | Kernel | Version | Status | LB |
| --- | --- | --- | --- | --- |
| 2026-05-14 14:46 UTC | `cataluna84/ash-s-arc-agi-3-agent` | v2 | PENDING | TBD |

## Next decision

- If D16 lands **>= 0.24**, keep a FORGE/MASTER safety-resubmit lane for
  otherwise-empty days.
- If D16 lands **0.19-0.23**, treat FORGE variance as a floor-protection
  tactic only.
- Continue debugging GraphExplorer against the reference implementation;
  do not submit it until local SDK diagnostics clear at least one real
  mounted public level.
