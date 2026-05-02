# exp004 Qwen3.6-35B-A3B BF16 — Postmortem

> **Status**: PARKED (not killed). Infrastructure stays; policy approach
> is replaced. See `research/04_strategy_reset_2026-05-01.md` for the new
> 4-week plan.
>
> **Date**: 2026-05-01.
> **What this doc answers**: "Why did the Qwen agent fail to clear even the
> 0.19 baseline despite the 9-iteration debugging marathon?"

---

## 1. TL;DR

We treated a vision-language model (Qwen3.6-35B-A3B) as the **policy** for
an agent in a sparse-reward, interactive grid-world. That assignment is
structurally wrong for this benchmark, and no amount of prompt tuning,
decode-budget tightening, or anti-repeat patches changes the fundamentals:

- Greedy decode gives the same first token for the same prompt — the
  agent picks the same action every turn.
- Vision prefill of a 512×512 frame is ~5 s/action on H100; the action
  budget for 25 games × 6 h cap fits roughly 150-200 actions/game, which
  on `ls20` smoke yielded 0 levels completed.
- VLMs have no built-in state memory or frame-change feedback signal.

The same model used as a **verifier of plans proposed by a symbolic
explorer** is potentially valuable, but only after the symbolic part
clears 0.42 on its own.

---

## 2. What we built

`agents/qwen_agent.py` (468 lines) — a `Qwen3.6-35B-A3B BF16` chat agent:

- Loads via `transformers 5.7.0 + accelerate` with `device_map="auto"`.
- Frame rendering: 64×64 grid → `_FRAME_UPSCALE × 8` PIL image (512×512)
  + a hex-digit text grid as parallel input.
- Prompt template: system rule "Reply with ONE action only", recent
  `(action, changed?)` history (8 frames), available actions list,
  current image.
- Action parsing: regex over the model output for `ACTION[1-7]` and
  `(x, y)` for `ACTION6`.
- Anti-repeat rotation (v11): if the same action picked 3+ times with no
  frame-change, rotate to next available action.

Smoke test (`scripts/qwen_agent_smoke_local.py`): 22 checks, all green
without GPU.

## 3. Kaggle dev-kernel iteration log (D2 06:00-09:18 UTC)

| v | UTC   | Status   | Symptom                                                                |
|---|-------|----------|------------------------------------------------------------------------|
| 1 | 06:00 | ERROR    | Legacy `/kaggle/input/<slug>/` no longer works on new image.           |
| 2 | 06:05 | ERROR    | SDK called `three.arcprize.org/api/games/anonkey` (no internet).       |
| 3 | 06:09 | ERROR    | PIL 11.3.0 `_typing.py` missing `_Ink`.                                |
| 4 | 06:13 | ERROR    | pip install pillow updated `.py` but old `_imaging.so` mismatched.     |
| 5 | 06:17 | ERROR    | Same as v4, different attempt at install.                              |
| 6 | 06:21 | ERROR    | `qwen3_5_moe` not recognized by transformers 5.0.0.                    |
| 7 | 06:46 | ERROR    | `pip install transformers` tried numpy>=1.17, no wheel offline.        |
| 8 | 06:50 | ERROR    | `tokenizers 0.23.1` exceeded transformers 5.7.0's `<=0.23.0` cap.      |
| 9 | 06:54 | COMPLETE | First green run. ls20 smoke: 50 actions in 853 s, 0 levels.            |
| 10| 07:27 | COMPLETE | Decode budget cut 96 → 16. 5.96 s/action. 0 levels.                    |
| 11| 07:47 | COMPLETE | Anti-repeat rotation. 5.13 s/action. 0 levels.                         |

Today (2026-05-01 09:18 UTC) the comp-kernel did a Save & Run All dry-run
that completed in ~30 s with `KAGGLE_IS_COMPETITION_RERUN=False`,
confirming all four infrastructure fixes are still in place. **No real
competition rerun has been triggered.**

## 4. Five compounding root causes

### 4.1 The policy is deterministic given the prompt

Greedy decode (`do_sample=False`, T=0) produces the same first token for
the same image+prompt embedding. v9-v11 confirmed: 50 actions in a row
with the same selection. The anti-repeat patch in v11 only flips
ACTION1↔ACTION2 in cycles because once the model commits to ACTION2
and the deque is dominated by ACTION2 entries, ACTION1 is allowed again.
Adding sampling (T > 0) introduces variance but no signal: the model's
sampling distribution doesn't know which actions actually move the world.

**Implication**: a VLM cannot be the policy. The policy must be a
structured exploration algorithm whose *state* includes "which actions
have been tried from this frame and what they did".

### 4.2 Vision prefill is the dominant cost

Steady-state per-action latency on H100 BF16:

```
prefill_image  ≈  256 patch tokens × 17.5 ms/token  ≈ 4.6 s
prefill_text   ≈   80 tokens       ×  6   ms/token  ≈ 0.5 s
decode_action  ≈   16 tokens       × 35   ms/token  ≈ 0.5 s
                                                    --------
total                                               ≈ 5.5 s/action
```

We tried `_DEFAULT_MAX_NEW = 96 → 16` (v10): saved ~0.5 s. Cutting
`_FRAME_UPSCALE = 8 → 4` would cut prefill by 4× to ~1.4 s/action — but
also cut visual fidelity in half. **vLLM with prefix-cache reuse is the
right answer**: the system prompt and prior turns are reused identically
across actions in the same game, so prefill cost can amortize from
~5 s/action down to ~0.5 s/action. We did not have time to integrate
vLLM in the D2 marathon.

### 4.3 No state memory across turns

The agent's `_history` is `deque[(action_name: str, changed: bool)]`.
There is no representation of *which states we have visited*, what actions
we have already tried *from this state*, or any topology of the game.
Turn N+1 cannot use the work of turn N except via prompt-included recent
history.

In contrast, the public Kaggle agents that score 0.28-0.42 all maintain
state-keyed memory:
- `yuriao/arc-agi-3-memoryagent` (0.28) — keyed memory by frame hash.
- Trigger-Aware BFS (0.35) — explicit state graph, BFS frontier ordered
  by untried-action count.
- CHRONOS / Ash (0.42) — full explore-then-replay framework.

### 4.4 No frame-change feedback signal

The prompt format includes `[ACTION1 x 30]` but does not annotate which
actions changed the world. The Stochastic-Goose-0.25 → StochasticGoose++
0.32 jump is purely from a CNN that classifies which actions in which
states actually move the grid. We never gave the model that information,
so it has no way to learn (in-context) which actions are plausible.

### 4.5 No ACTION6 click-coordinate path

`ACTION6` requires `(x, y) ∈ [0, 63]²`. The smoke kernel never picked
ACTION6 because the model's first token preference was always ACTION1.
Kaggle's 25 games include some that *only* progress via clicks (e.g.
`vc33` levels 4+). We were therefore zero-filling those games even before
considering scoring efficiency.

## 5. What we keep from exp004 (the win)

Even though Qwen-as-policy is dead, these artifacts are reusable:

### 5.1 Kaggle Datasets (permanent storage, no slot consumed to keep)

- `cataluna84/arc-agi-3-agents-pkg` (164 KB) — our agents/ package, used
  by every future kernel.
- `cataluna84/qwen3-6-35b-a3b-bf16` (71.93 GB, 26 shards) — kept around
  for a possible Tier-C verifier role. Cheap to attach, expensive to
  re-upload.
- `cataluna84/arc-agi-3-transformers-wheels` v2 (16 MB) — transformers
  5.7.0 + tokenizers 0.22.2 + 8 deps. Reusable for any future kernel
  that needs newer transformers.

### 5.2 Code artifacts

- `agents/qwen_agent.py` — kept; switched from policy to potential
  verifier role in Tier-C of the new architecture.
- `experiments/exp004_qwen_agent/dev_kernel/qwen_agent_dev.ipynb` — the
  *cell pattern* (nested-mount adapter, OFFLINE-mode setup, --target
  installs, sys.modules purge) is the template for every future kernel
  on this image.
- `scripts/qwen_agent_smoke_local.py` — 22-check parity test, useful for
  any future Qwen-on-CPU smoke.

### 5.3 Documentation

- `experiments/exp004_qwen_agent/D2_EXECUTION_LOG.md` — literal terminal
  log of every CLI command, useful for any future Kaggle CLI flow.
- `experiments/exp004_qwen_agent/RUNBOOK_D2.md` — decision tree for
  Track A/B/C selection.
- `.factory/rules/gotchas.md` #11-14 — four hard-won Kaggle image gotchas.

## 6. Recommended next steps

1. **Park exp004 here.** The dev kernel + comp kernel can be re-pushed
   any day, but the strategy is to leave it dormant until Tier A reaches
   0.42.
2. **Start exp005 (CHRONOS port).** Track week 1 of the new plan in
   `research/04_strategy_reset_2026-05-01.md`.
3. **Burn today's slot on Track A** — re-submit ForgeAgent unchanged for
   the variance probe. This gives `s2` for the original D1 plan.
4. **Update `.factory/plan.md`** to reflect the new 4-week milestones:
   D5 already targets the FORGE Trigger-Aware BFS port (0.32-0.35). Add
   the state-graph wrapper to D6, frame-change CNN to D8-D9, etc.

## 7. Lessons (canonized in `.factory/rules/gotchas.md`)

- **Lesson 11**: Kaggle image since 2026-03 mounts datasets at
  `/kaggle/input/datasets/<owner>/<slug>/`, not the legacy
  `/kaggle/input/<slug>/`.
- **Lesson 12**: The `arc-agi` SDK calls `three.arcprize.org/api/games/anonkey`
  unless `OperationMode.OFFLINE` + `environments_dir=` is set.
- **Lesson 13**: Pillow 11.3.0 on Kaggle's image has a `_typing.py`
  missing the `_Ink` symbol. `pip install pillow` updates the `.py` but
  the old `_imaging.so` C-extension wins; only `--target` install +
  `sys.path.insert(0, ...)` + `sys.modules` purge fixes it.
- **Lesson 14**: Transformers 5.0.0 (the Kaggle-image-bundled version)
  is missing `qwen3_5_moe`. Bundle 5.7.0 wheels as a Dataset and install
  with `--no-deps --target /kaggle/working/_transformers_pkg`.

(Also: the **policy ≠ VLM** lesson belongs in this list as Lesson 15.
Will add once the user confirms the strategy reset.)
