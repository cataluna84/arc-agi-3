# Strategy + Kaggle Compute Reality Check (2026-04-29)

> Combines Exa Deep-Reasoning search results, Ref-MCP Kaggle docs, the official
> Kaggle CLI source, and the actual `kernel-metadata.json` of Ash's published agent.

---

## 1. What does the ARC-AGI-3 leaderboard actually look like (April 2026)?

| Tier | Score | Examples | Status |
| ---- | ----- | -------- | ------ |
| Top private | 0.68 | Redfield Rentals (#1) | private code, presumed BFS + neural + program library |
| 2-5 private | 0.64-0.66 | Barada Sahu, Kevin E R MILLE, SVG, Matthew Philip Poetker | private |
| Top public  | 0.42 | Ash's ARC-AGI-3 Agent (FORGE v19, BFS + ForgeNet CNN) | public, we vendored as `agents/_ash_my_agent_v19.py` |
| Mid public  | 0.30-0.39 | Trigger-Aware BFS (0.35), FORGE 0.39, Hybrid Search-and-Learn 0.35, Redpill 0.30 | public |
| Low public  | 0.18-0.28 | StochasticGoose 0.25 (sample baseline), Just Explore 0.19, Random 0.18, MemoryAgent 0.28, MCTS Solver 0.29, StochasticGoose++ 0.32, Cognitive-Rungs 0.21 | public |
| **Our anchor** | **0.19** | vanilla Ash fork, rank 398 — **23-point reproduction gap from Ash's claimed 0.42** | confirmed |
| Frontier LLMs zero-shot | 0.001-0.005 | Opus 4.6 0.50%, Gemini 3.1 Pro 0.40%, GPT-5.4 0.20%, Grok-4.20 0.10% | semi-private LB; LLMs without harness are essentially random |

### What's NOT working on the LB (despite hype)

A flood of speculative architectures published April 2026 with 0 LB scores:
`Stitch Library Learner`, `Fractal MoE`, `Meta Reasoner`, `AURORA-X BFS`,
`TTT-MLP`, `LLM Visual Analyzer`, `LLM Solver Composer`, `JEPA Program Synthesis`,
`NCA World Model`, `Causal-JEPA + TRM`, `NSA Transduction`, `Slot-GNN Symbolic`,
`Resonance Sampler`, `DQN LinAlg`, `GGRoPE World Model`, `Brain MoE`,
`Classical Game Rule Induction`, `Fast ARC Agent: Parallel CNN` (= 0.19), etc.

**Most are notebooks where the author wrote code but never submitted, OR submitted but failed and didn't update.** The signal: don't spend a daily slot on a speculative architecture before reading whether someone published a real LB number for it.

### What IS working

1. **BFS over simulated game states** with state-hash dedup + trigger-aware queue ordering (FORGE family, Ash's agent).
2. **Frame-change CNN** with experience buffer + per-level reset (StochasticGoose family).
3. **Graph exploration with frontier-shortest-path** (Just Explore family — 3rd in dev preview, ~17/25 levels post-bug-fix).
4. The 0.30-0.42 cluster is **Hybrid 1+2** (BFS gates + small CNN value).
5. The mystery 0.65+ tier is **closed**, but most likely: 1+2+3 + program/DSL synthesis + per-game heuristic library.

---

## 2. What the Kaggle API actually allows (programmatic compute)

### YES — full programmatic kernel push + run + fetch

`kaggle kernels push -p <folder>` uploads a kernel **and immediately runs it**. Source: Kaggle/kaggle-cli `docs/kernels.md`, confirmed in our local `kaggle 2.1.0` install.

Workflow we can automate:

```bash
# 1. Push (uploads + runs)
kaggle kernels push -p <folder>          # folder must contain kernel-metadata.json + .ipynb/.py

# 2. Poll status
kaggle kernels status <user>/<slug>

# 3. Fetch logs and output files (e.g. submission.parquet)
kaggle kernels output <user>/<slug> -o   # downloads to cwd

# 4. Submit kernel output to a code competition (programmatic Submit)
kaggle competitions submit-code \
  --competition arc-prize-2026-arc-agi-3 \
  --kernel <user>/<slug> --kernel-version <int> \
  -f submission.parquet -m "exp003 baseline goose"
```

The CLI uses `KaggleApi.competition_submit_code(...)` under the hood; `ApiCreateCodeSubmissionRequest` carries `kernel_owner`, `kernel_slug`, `kernel_version`, `competition_name`, `file_name`, `submission_description`. Source: `Kaggle/kaggle-api/src/kaggle/api/kaggle_api_extended.py`.

### Daily submission limit is server-side

The 1/day cap is enforced by Kaggle's API, not just the UI — we cannot bypass it via tooling. **But we CAN run experimental kernels without submitting**, getting logs/scorecard/output files, useful for ablation studies that don't burn the daily slot.

### `--accelerator` options (Feb 2026 CLI list)

- `NvidiaTeslaP100` (default GPU, free tier)
- `NvidiaTeslaT4`, `NvidiaTeslaT4Highmem`
- `NvidiaTeslaA100`
- `NvidiaL4`, `NvidiaL4X1`
- **`NvidiaH100`**  ← present in the CLI but quota-gated
- `NvidiaRtxPro6000`
- `TpuV38`, `Tpu1VmV38`, `TpuV5E8`, `TpuV6E8`

> Free-account "floating" GPU quota is **30 hours/week** (Kaggle product-feedback #173129). Older docs say P100-only, but the CLI exposes 12 accelerator types — actual availability per accelerator is dynamic.
>
> H100 quota for individual users is **not publicly documented**. Several reports indicate H100 is **competition-specific** (i.e. unlocked only when a kernel is attached as `competition_sources` for an ARC-Prize-tier competition).

### Critical reality check from Ash's actual kernel-metadata.json:

```json
{
  "id": "ashvinsingh/ash-s-arc-agi-3-agent",
  "enable_gpu": true,
  "enable_internet": false,
  "machine_shape": "NvidiaTeslaT4",        ← T4, not H100!
  "docker_image": "gcr.io/kaggle-private-byod/python@sha256:00377c..."
}
```

**The 0.42-public-LB Ash agent runs on T4 — not H100.** Our local RTX 2070 (Turing sm_75, 8GB) is the **same architecture** as Kaggle's T4 (sm_75, 16GB) with half the VRAM. For BFS-bound or small-CNN agents the practical performance is in the same ballpark.

Implications:
1. The H100 path is **not what current top public agents are using**. Maybe nobody has scaled to a model big enough to need it yet, or Kaggle's H100 quota is too tight.
2. **Local development on the RTX 2070 is hardware-comparable to the actual eval target.** Don't waste time chasing H100 parity — it's not the bottleneck.
3. The `gcr.io/kaggle-private-byod/python` image is the BYOD ("Bring Your Own Docker") image attached to the comp; **not publicly pullable** but the package set is roughly the public `kaggle-images/python`. CUDA torch + numpy + arc-agi wheels (which we already have).

---

## 3. Highest-EV next steps (ranked by expected lift / cost)

### Tier 1 — kill quick hypotheses (1 submission slot each)

1. **exp002 — Ash variance probe.** Re-submit the SAME forked Ash notebook on D1 and D2. If `max(s2,s3) >= 0.30`, the 0.19 was a low-variance draw and we can squeeze free LB by a best-of-N seed sweep at exp004. If `max < 0.25`, structural ceiling at 0.19 — pivot.
2. **exp003 — orthogonal anchor.** Fork the official `ARC3 Sample Submission - Stochastic Goose` (0.25) and `Just Explore` (0.19) — submit each unchanged. Gives us two non-Ash references and confirms our submission pipeline isn't the bug.

### Tier 2 — public-notebook reproductions (each = 1-3 submission slots)

3. **exp005 — FORGE Trigger-Aware BFS** (target 0.32-0.35). Public notebook `rauffauzanrambe/trigger-aware-bfs-for-game-simulation-score-0-35` is the cleanest reference. Implement minimally: state hashing (xxhash/blake2 over `(state_grid, available_actions)`), BFS frontier ordered by **untried-action count** + **trigger score** (Δpixels, Δscore, new colors) + **distance to nearest unexplored state**. Reset state-graph between levels.
4. **exp006 — StochasticGoose++ CNN reproduction** (target 0.30). Reproduces the 1st-place dev-preview design as a non-search baseline. Useful in case BFS is fundamentally limited on private games.

### Tier 3 — Ash improvements (high EV, no Kaggle slot needed for dev)

5. **Profile Ash locally.** Our 132s/50-actions on ls20 is BFS-bound. The candidate hot paths to attack:
   - Replace `copy.deepcopy(game)` with `pickle.dumps/loads` (often **2-5x faster** for small Python objects); even better, manually copy only the mutable scalars and let immutables alias.
   - Cython-compile `_fast_deepcopy` and `perform_action` (would require source access — we already have the local game class .py files).
   - Replace `_visited_hashes` set with a `dict` keyed by `bytes(np.ascontiguousarray(grid))` to skip the per-step `xxhash` call.
   - Pre-allocate the BFS frontier as a `heapq` instead of `deque` so the trigger-priority sort is amortized.
6. **Tighten Ash's heuristics.** The CLTI demo injection and `_unproductive>=30` undo logic are tunable. A Bayesian sweep on the local box across 5 seeds × 25 games × 200-action budget would surface the best hyperparameter region without touching Kaggle.
7. **Per-game branch.** Ash's MyAgent is one-size-fits-all. Detecting game family (sokoban-like / lever-and-door / cluster-merge / etc.) and switching `_init_bfs` priors per family is a **+0.05 to +0.15 LB lift** based on landscape priors.

### Tier 4 — moonshots (don't do these until the above finish)

8. **DreamerV3 world-model agent.** PyTorch port `NM512/dreamerv3-torch`. Strong long-horizon priors but ~1 week to integrate properly.
9. **Bundled local LLM (≤7B Qwen2.5-Coder)** as an offline RGB-Agent-style reasoner. Would need to fit weights in the Kaggle dataset attached to the kernel (no internet during eval). High complexity, unclear if it beats 0.42.
10. **Program-synthesis / DSL transducer.** `Stitch Library Learner` family. Big bet, possibly the route to 0.65+.

### Recommended sequence (next 7 days, 7 submission slots)

| Day | Experiment | Slot |
| --- | ---------- | ---- |
| D1 (today) | exp002 — Ash resubmit #1 | 1 |
| D2 | exp002 — Ash resubmit #2 (variance probe) | 2 |
| D3 | exp003 — Stochastic Goose unchanged | 3 |
| D4 | (no slot) Local: profile Ash, kill 2 BFS hot paths | 0 |
| D5 | exp005 — Trigger-Aware BFS reproduction (target 0.32) | 4 |
| D6 | exp005a — Trigger-Aware BFS + Ash's hidden-field trick | 5 |
| D7 | exp006 — Ash + sped-up BFS (Tier 3 #5) | 6 |

This puts us at a known-good 0.30+ floor by D7, with two unburned Kaggle slots for ablation, and substantial local data on what's actually moving the needle.

---

## 4. What I'm NOT recommending and why

| Path | Why not |
| ---- | ------- |
| Pulling the Kaggle Docker image (`kaggle-images/python`, ~40 GB) | The actual comp image is private (401); the public alt isn't bit-perfect; our local RTX 2070 is already same-arch as the T4 eval target. Disk/time cost > marginal benefit. |
| Trying `--accelerator NvidiaH100` first | Top public agents (Ash, FORGE, Goose) all run on T4. H100 is a quota gamble and probably bottlenecked elsewhere. Test it ONLY when we have a model that is provably GPU-bound. |
| Spinning up a Kaggle Notebook just for compute | The `kaggle kernels push` flow gives us programmatic remote execution **anyway** (and counts against our quota). No advantage to manual notebook unless we need interactive debugging. |
| LLM-driven agents now | The semi-private LB shows zero-shot frontier LLMs at 0.001-0.005. Without a strong harness (RGB-Agent style), they're worse than Random. Bundling a 7B model into the kernel is doable but ~1 week and unclear ROI. |
| World-model approaches now | DreamerV3 etc. are 1+ week integrations. Not worth the slot until our search baseline is solidly above 0.30. |

---

## 5. Open questions to verify

- Q1: Does `kaggle kernels push --accelerator NvidiaH100` actually allocate an H100 for an ARC-AGI-3-attached kernel, or does it silently downgrade to T4? **Test plan:** push a tiny kernel that prints `nvidia-smi` and check the output.
- Q2: What's the actual H100-hours quota for the user's account? **No public API for this; check the user's Kaggle settings page UI.**
- Q3: Does the competition rerun environment use a different GPU than the dev kernel run? **Assumption (per `kernel-metadata.json`):** rerun uses the same `machine_shape` declared in metadata. So if Ash declared T4, the rerun is on T4.

If we ever want to verify Q1/Q3, that's a 1-line kernel push, no submission needed.

---

## 6. Sources

- Kaggle/kaggle-cli `docs/kernels.md` (`https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels.md`)
- Kaggle/kaggle-api `src/kaggle/api/kaggle_api_extended.py` (commit `c27de268`)
- DeepWiki Kaggle/kaggle-api Kernels/Notebooks API page
- Kaggle product-feedback #173129 (floating GPU quota)
- ARC Prize Foundation: arxiv 2603.24621v2 (ARC-AGI-3 technical report)
- Ash's `kernel-metadata.json`: `research/ash_notebook/kernel-metadata.json`
- ARC-AGI-3 Just-Explore (3rd dev preview): `dolphin-in-a-coma/arc-agi-3-just-explore`, arxiv 2512.24156
- StochasticGoose 1st dev preview: `DriesSmit/ARC3-solution`
- FORGE blog: `huggingface.co/blog/MiniMax-AI/forge-scalable-agent-rl-framework-and-algorithm`
- Trigger-Aware BFS notebook: `kaggle.com/code/rauffauzanrambe/trigger-aware-bfs-for-game-simulation-score-0-35`
- ARC-AGI-3 leaderboard: `kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard`
