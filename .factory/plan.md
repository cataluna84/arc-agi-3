# PLAN.md — Daily Experiments + Definition of Done

> **North Star**: surpass the public-LB top score (currently 0.68) by 2026-09-30, and strive toward the 100% Grand Prize threshold.
>
> **Constraint**: 1 Kaggle submission/day. Each daily slot must move the score, build a foundation, or kill a hypothesis.
>
> **Locked-in baseline**: the user's existing submission (vanilla fork of an upstream public **FORGE v19** Kaggle notebook) scored **0.19** at rank 398 on 2026-04-29. Every "Δ over baseline" downstream is measured against **0.19**, not 0.25 nor 0.42. See `NOTICE` for upstream attribution.
>
> **D3+ canonical spec**: see `experiments/SPEC_4WEEKS.md` (drafted 2026-05-01) for the full file/code-level day-by-day plan from D3 through D28. The phase-level summary below is preserved for context; the SPEC overrides any conflict on day-level detail.

---

## Phase 0 — Foundation: anchor & understand variance (Days 0–3)

### [x] D0: Vanilla fork of upstream FORGE v19 (anchor)

**Goal**: anchor the LB pipeline and lock in our baseline number.

- [x] Forked the upstream FORGE-v19 notebook on Kaggle (advertised public ~0.42)
- [x] Submitted unchanged
- [x] **Result: LB 0.19, rank 398** — a −0.23 reproduction gap from the published number

**DoD**: green submission with a real LB number. (DONE)
See: `experiments/exp001_baseline_forge/`.

### [x] D1: FORGE variance probe — resubmit #1 (exp002) — slot used 2026-04-29

**Goal**: figure out whether 0.19 is "the real expected score" or just a low-variance draw.

- [x] Re-submitted (or planned to re-submit) the SAME forked FORGE notebook unchanged
- [ ] Record `s2`; compare to 0.19  (result lands ~24 h after submission, append to memories.md)

**DoD**: a second LB data point recorded; absolute diff `|s2 - 0.19|` logged in `.factory/memories.md`.

### [x] D2 (2026-04-30): Track B chosen — Qwen3.6-35B-A3B comp submission (exp004) — slot consumed

**Result**: LB **0.00**. Track B failed; postmortem in `experiments/exp004_qwen_agent/POSTMORTEM.md` (5 root causes: greedy decode, vision prefill bottleneck, no state memory, no change feedback, no ACTION6 path).

### [x] D3 (2026-05-01): Track A FORGE variance probe — slot consumed; **LB 0.24** (rank 315/715)

**Submitted**: `kaggle competitions submit arc-prize-2026-arc-agi-3 -k cataluna84/ash-s-arc-agi-3-agent -v 2 -f submission.parquet`. Result: **0.24** (+0.05 vs 0.19 baseline). Verdict per scores.json decision rule: max(0.19, 0.24) < 0.25 -> structural-floor reading; pivot to other agents per SPEC_4WEEKS.md.

**Also**: drafted `research/04_strategy_reset_2026-05-01.md` and `experiments/SPEC_4WEEKS.md` (D3-D28 detailed spec). All future days follow SPEC_4WEEKS.

### [x] D4 (2026-05-02): Combined D5+D6 — Trigger-BFS + StateGraph submitted as ablation

**User direction** at D4 morning AskUser: Option B (skip Goose, jump to D5 Trigger-BFS), strict smoke, bundle state_graph from D6. Then Option C (submit even though local sweep showed regression risk).

**Built**: `agents/state_graph.py`, `agents/trigger_bfs_agent.py`, `tests/test_state_graph.py`, `scripts/trigger_bfs_smoke_local.py`, `experiments/exp005_trigger_aware_bfs/comp_kernel/*`.

**Local sweep**: 1/25 games clear level 1 (ft09); RandomAgent also gets 1/25 (r11l). Strict ls20 gate not met (neither random nor trigger-bfs passes ls20).

**Submitted**: kernel `cataluna84/trigger-bfs-comp-arc-agi-3` v1 at 14:24 UTC. Status: PENDING. Predicted LB 0.18-0.22.

**Net daily slot tally**:
- D0 (2026-04-29): 0.19 (FORGE v1)
- D1 (placeholder; consumed by D2 logic) — slot held
- D2 (2026-04-30): 0.00 (Qwen Track B)
- D3 (2026-05-01): 0.24 (FORGE v2, Track A variance)
- D4 (2026-05-02): PENDING (exp005 Trigger-BFS ablation)

### [ ] D5+ (2026-05-03 onward): see `experiments/SPEC_4WEEKS.md`

The day-by-day plan from D5 through D28 lives in `experiments/SPEC_4WEEKS.md`.
That document is the canonical source for: file paths, code patterns,
smoke-test commands, submit decisions, exit criteria, rollback plans,
and the slot-accounting table.

The decision tree for D5 (2026-05-03) depends on the exp005 LB result:

- **If exp005 LB > 0.24**: state-graph dedup is genuinely helping; build BFS
  replay (SPEC §1.5 D7) on top of TriggerBFSAgent. Target LB 0.32-0.35.
- **If 0.18 < exp005 LB <= 0.24**: state graph helps modestly but not
  enough on its own. Add frame-change CNN + BFS replay simultaneously
  (combined SPEC D7 + D9). Target LB 0.30-0.36.
- **If exp005 LB <= 0.18**: regression vs random; diagnose via dev kernel
  logs and revert to FORGE for D5 slot. Recovery path: re-run TriggerBFSAgent
  with action-priority disabled or seed-sweep.

> The legacy `Phase 1 / Phase 2 / Phase 3 / Phase 4` sections that
> previously enumerated D5..D20 with target scores have been superseded
> by `experiments/SPEC_4WEEKS.md` (see header note above). Phase-level
> Definition of Done targets remain valid (Phase 1 = 0.35, Phase 2 = 0.45,
> Phase 3 = 0.55, Phase 4 = 0.70) and are referenced below.

### exp004 milestones (legacy, completed 2026-04-30)

The Qwen3.6-35B-A3B Track B is closed; postmortem at
`experiments/exp004_qwen_agent/POSTMORTEM.md`. The `cataluna84/qwen3-6-35b-a3b-bf16`
Kaggle Dataset (~72 GB) is parked, not deleted, in case we revive Qwen
as a Tier-C verifier per `research/04_strategy_reset_2026-05-01.md` §6.

---

## Always-on Hygiene (every day)

- **Before** burning the daily Kaggle slot:
  - [ ] Smoke test on at least 3 public games locally; agent does not crash, completes ≥ 1 level on tutorial
  - [ ] Confirm runtime extrapolation: (avg_actions_per_game × 110 games × per_action_seconds) < 5h
  - [ ] Diff vs last successful submission (`git diff` or notebook compare)
- **After** every submission:
  - [ ] Capture LB score, game-by-game scorecard, and any tracebacks → `.factory/memories.md`
  - [ ] If score regressed > 0.02, immediately roll back and bisect

## Submission slot tally (running)

| Day | Date       | Kernel slug                                  | Predicted | LB     | Rank        | Notes                          |
|----:|:-----------|:---------------------------------------------|----------:|-------:|:------------|:-------------------------------|
|  D0 | 2026-04-29 | `cataluna84/ash-s-arc-agi-3-agent` v1        |       — | 0.19   | 398/—       | FORGE v19 fork anchor          |
|  D1 | 2026-04-29 | (held)                                       |       — | —      | —           | Slot held; absorbed by D2 logic |
|  D2 | 2026-04-30 | `cataluna84/qwen-comp-arc-agi-3` v1          |     ~0.0 | 0.00   | —           | Qwen Track B — postmortem      |
|  D3 | 2026-05-01 | `cataluna84/ash-s-arc-agi-3-agent` v2        |    ~0.19 | 0.24   | 315/715     | FORGE v2, +0.05 variance lift  |
|  D4 | 2026-05-02 | `cataluna84/trigger-bfs-comp-arc-agi-3` v1   | 0.18-0.22 | PENDING | —          | exp005 Trigger-BFS ablation    |
|  D5+| 2026-05-03+| (TBD per exp005 LB result)                   | 0.30-0.36 | —      | —           | See `experiments/SPEC_4WEEKS.md` |

## Definition of Done for each phase

- **Phase 0**: 0.19 baseline anchored; variance of the upstream FORGE notebook understood; local runner working. Done.
- **Phase 1**: one of our notebooks ≥ 0.35 on Kaggle (Δ ≥ +0.16 over 0.19 anchor).
- **Phase 2**: ≥ 0.45 on Kaggle (Δ ≥ +0.26).
- **Phase 3**: ≥ 0.55, top-10 on public LB (Δ ≥ +0.36).
- **Phase 4**: ≥ 0.70, beating current LB top (Δ ≥ +0.51).

## Pointers

- Detailed D3-D28 plan: `experiments/SPEC_4WEEKS.md`
- Strategy + literature: `research/04_strategy_reset_2026-05-01.md`
- Daily memories (top of file is most recent): `.factory/memories.md`
- Hard-won gotchas (15 items): `.factory/rules/gotchas.md`
- Pre-submission verification (V1..V11): `.factory/verify.md`
