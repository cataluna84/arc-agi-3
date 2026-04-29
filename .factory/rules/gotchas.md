# Hard-Won Gotchas (10)

1. **The public notebooks named "FORGE", "Trigger-Aware BFS", "Stochastic Goose"** are *forks of the same families* — read the *highest-upvoted gold/silver* one and treat lower-vote forks as iteration breadcrumbs.
2. **Score 0.66 ≠ 66% wins** — RHAE squares per-level efficiency, so a 0.66 final score corresponds to roughly 80%+ partial-level completion at near-human efficiency, or solving most levels with ~3× human action overhead. (See discussion thread "It is 0.66%" by CPMP.)
3. **Frame deltas are the lifeblood of every working agent**. If your agent cannot detect that an action did nothing (zero change), you will burn the action budget on dead ends. Every top notebook hashes the frame and checks `new_hash == prev_hash`.
4. **State-graph reset bug** — the `dolphin-in-a-coma` graph explorer leaked across levels at first; lost ~5 private levels. Always **clear state graph + hash table on level transitions**.
5. **ACTION6 click is the high-leverage action** — 64×64 = 4096 click targets. Naive uniform sampling burns the budget. Use frame segmentation + saliency tiers (StochasticGoose conv head, Just-Explore priority groups).
6. **The first level of every game is a tutorial** — random can win it. Don't measure your agent's intelligence by tutorial success.
7. **No internet in eval**. Pretrained weights must be packaged as Kaggle Datasets and loaded locally. ARC API calls only work because the runtime stubs them — read the Sample Submission notebook code carefully to see how it talks to the local environment server.
8. **Submissions are auto-generated** as long as your agent takes any action on any game. Empty/crashed runs may still produce a submission with all zeros.
9. **CPU vs GPU vs H100** — H100 only available for ARC-AGI-3 notebooks (no internet). H100 is ideal for any neural model; for non-neural (BFS/graph) prefer CPU to leave the H100 quota free.
10. **Daily 1-submission cap** is the binding constraint. Plan **A/B variants offline** (using public games as smoke tests) before burning the daily slot.

## Open questions (unresolved as of 2026-04-29)

- Can we **mirror the public game files locally** to truly reproduce the Kaggle eval offline? (`environment_files/` ships 25 of the 30+ games.)
- Is `ACTION7 == UNDO` always available? (StochasticGoose blog says UNDO was *unavailable* during dev preview; current `v0.9.3` lists ACTION7 as a `GameAction`.)
- What is the **per-game step budget** the eval enforces beyond the 6h wall clock? — flag for the Discussion thread "Intended compute budget".
- "Public-LB advertised score" of a notebook is NOT a reliable predictor of what a fork will achieve — variance is high. Why does Ash's notebook reproduce at 0.19 instead of 0.42?
