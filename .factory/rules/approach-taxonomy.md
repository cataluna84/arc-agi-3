# Approach Taxonomy (A–G)

> Canonical reference points for ARC-AGI-3 agents. When picking a target for a new exp, name the class explicitly.

## A. Pure exploration (no learning)

- **Graph-Based Exploration** (Rudakov et al., AAAI'26 workshop, arxiv 2512.24156, repo `dolphin-in-a-coma/arc-agi-3-just-explore`): frame segmentation → priority tiers → directed state graph → frontier-shortest-path action choice. **3rd in dev preview**, post-bug-fix solves median 17/25 private levels. **Gold standard baseline for systematic exploration.**

## B. Learned action-change predictor (RL-lite)

- **StochasticGoose** (Smit @ Tufa Labs, repo `DriesSmit/ARC3-solution`): **1st place dev preview, 12.58%, 18 levels**. CNN with 16-channel one-hot input → 4-layer CNN backbone (32→64→128→256) → action head + 64×64 conv coordinate head. Binary: "will this action change the frame?". Hash-deduped 200k experience buffer. Reset model + buffer between levels.

## C. Trigger-aware BFS (top public Kaggle notebook family)

- "Trigger-aware" = detect frames where any action causes a *significant* delta (e.g., score change, color cluster appears/disappears, new region) and bias BFS into those branches. Combined with state-hash dedup, this is the source of the 0.35–0.42 cluster.

## D. Hybrid Search-and-Learn (and Redpill Latent Planning)

- BFS/MCTS skeleton + small CNN/MLP for value or change probability. Redpill (0.30) uses *zero-prior latent planning*: encodes states with a learned latent and rolls out plans in latent space.

## E. World-model based

- **DreamerV3** (Hafner et al., Nature 2025, code: `danijar/dreamerv3`, PyTorch port: `NM512/dreamerv3-torch`) — RSSM world-model + actor-critic on imagined trajectories. Few hyperparameters. Strong baseline for novel envs.
- **JEPA / GGRoPE / Causal-JEPA + TRM** — appear as low-score notebooks (~0.05–0.17) but have strong long-horizon priors.
- **NCA World Model** (neural cellular automata) — promising for grid worlds.

## F. Program / DSL synthesis (transducer)

- **Stitch Library Learner**, **MDL Program Synthesis**, **NSA Transduction**, **Classical Game Rule Induction** — synthesize compact programs from observed transitions. Underexplored in current LB but historically dominant for ARC-1/2.

## G. LLM-driven (NOT directly usable in Kaggle eval — no internet)

- **RGB-Agent** (alexisfox7) — Read/Grep/Bash + Claude/GPT — best public action efficiency (1,069 actions for 3 preview games) but requires APIs.
- **LLM Solver Composer / LLM Visual Analyzer** — same constraint.
- **Workaround**: ship a small (≤7B) quantized open-source model (Qwen2.5-Coder, DeepSeek-Coder, Llama 3) bundled into the notebook for **offline LLM reasoning**. Must fit in H100 80GB or GPU 16GB. RGB-Agent's `local-qwen` mode is the template.
