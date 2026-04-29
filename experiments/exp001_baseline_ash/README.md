# exp001 — Baseline: Ash's ARC-AGI-3 Agent (vanilla fork)

**Day**: D0 (already submitted)
**Status**: SUBMITTED
**LB score**: **0.19** (rank 398 at submission time)
**Notebook**: `Ash's ARC-AGI-3 Agent` (public, advertised score ~0.42)
**Submission strategy**: Vanilla fork — no code changes.

---

## Headline finding

We forked the public **Ash's ARC-AGI-3 Agent** notebook (advertised ~0.42 public LB) and got **0.19** on actual submission. **A −0.23 reproduction gap** that tells us something important:

- Either Ash's agent is **highly stochastic** (different random seeds → very different scores), or
- The advertised 0.42 was **cherry-picked / best-of-N**, or
- The notebook was **silently downgraded** between Ash's run and ours (e.g. environment/SDK version drift, dataset version, accelerator change), or
- The current LB has gotten harder since Ash's notebook was published (more games added).

This delta is the **single most valuable signal** we have right now. Resolving it is exp002.

## Locked-in anchor for the rest of the project

- `BASELINE_LB_ANCHOR = 0.19` (use this, not 0.25 nor 0.42, when computing improvement deltas in `.factory/memories.md`).
- Any experiment claiming "+X over baseline" is measured against **0.19**.

## What "Ash's ARC-AGI-3 Agent" actually does (to be filled in after exp002)

We have **not yet** read Ash's notebook line-by-line. exp002 starts with that — we need to understand:

1. Is there a learned model, or pure search?
2. What is the action policy? (Random / heuristic / CNN / LLM-orchestrated?)
3. How is RNG seeded? Is it deterministic across runs?
4. Does it adapt within an episode (online learning) or just replay a fixed strategy?
5. What is the wall-clock budget per game?

## Definition of done (already met)

- [x] Submission accepted on Kaggle.
- [x] LB score recorded: 0.19.
- [x] Rank recorded: 398.

## Files

```
exp001_baseline_ash/
├── README.md            # this file
├── score.txt            # 0.19
├── rank.txt             # 398
└── notebook_url.txt     # to be filled with the user's forked Kaggle notebook URL
```

## Next steps

→ exp002_ash_variance_probe: re-submit the exact same forked notebook 1–3 more times across upcoming Kaggle daily slots **without any change** to estimate the variance of the agent. If the score swings ±0.10 between identical runs, the agent is stochastic-dominated and we know we have to either fix the seed or switch agents.

→ exp003_ash_decode: read Ash's notebook in detail, write a 1-page summary into `experiments/exp003_ash_decode/README.md`, and identify reuse points for our future agents.
