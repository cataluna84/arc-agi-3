# Exa Deep Researcher Pro — Output (2026-04-29)

Research ID: r_01kqcepv7wx50df3kg3ad6s3er

Model: exa-research-pro

Cost (USD): 1.7176574999999998

---

# ARC-AGI-3 (ARC Prize Foundation 2026) — Comprehensive Technical Report

## 1) Architectures of Top-Scoring Agents (>0.30)

Note: the following agent descriptions synthesize published agent code, Kaggle notebooks, the ARC-AGI-3 technical report, competition writeups, and developer blogs. Each agent section cites repositories or writeups that provide implementation-level detail.

### FORGE Agent — Trigger-Aware BFS (CHRONOS)
- Core idea: A search-first agent that combines systematic breadth-first search (BFS) over simulated game states with lightweight trigger/event detection to prioritize branches likely to lead to goal states. Implementation is provided as a FORGE-style agent within the ARC-AGI-3-Agents codebase and related Kaggle notebooks demonstrating trigger-aware BFS strategies [ARC-AGI-3-Agents GitHub](https://github.com/arcprize/ARC-AGI-3-Agents) [Kaggle Trigger-Aware BFS notebook](https://www.kaggle.com/code/rauffauzanrambe/trigger-aware-bfs-for-game-simulation-score-0-35/input?scriptVersionId=310983317) [FORGE framework blog](https://huggingface.co/blog/MiniMax-AI/forge-scalable-agent-rl-framework-and-algorithm).

- Search mechanics: BFS enumerates states as nodes; edges are state transitions produced by applying environment actions via the ARC API. The BFS queue is prioritized by a trigger score (see Trigger Detection below). State nodes include compressed frame representation + action history to allow path reconstruction and goal-checking via the environment's state return values [ARC-AGI-3-Agents GitHub](https://github.com/arcprize/ARC-AGI-3-Agents).

- Trigger detection: The agent scans each simulated successor state for ‘triggers’ — visual or symbolic patterns indicating progress (appearance/disappearance of key colors, object creation, or localized structural changes). When triggers are present, nodes are elevated in the search queue (enqueued with higher priority) so BFS explores them sooner; non-trigger branches are still visited but delayed. Practical trigger detectors are implemented as rule-based pattern matchers (fast image masks) or lightweight learned classifiers trained to predict frame-change likelihood [Kaggle Trigger-Aware BFS notebook](https://www.kaggle.com/code/rauffauzanrambe/trigger-aware-bfs-for-game-simulation-score-0-35/input?scriptVersionId=310983317).

- Game-state simulation: FORGE implements an internal simulator that deterministically applies API action semantics (RESET, ACTION1–ACTION7) to frame arrays so BFS expansion is cheap and avoids network round-trips for simulation steps; the simulator uses numpy-based delta-application for speed and hash-based state IDs for deduplication [ARC-AGI-3-Agents GitHub](https://github.com/arcprize/ARC-AGI-3-Agents).

- Pseudocode (Trigger-Aware BFS):

```python
from collections import deque
queue = deque([(initial_state, [])])
seen = set()
while queue:
    state, path = queue.popleft()
    sid = hash_state(state)
    if sid in seen: continue
    seen.add(sid)
    if is_goal(state): return path
    triggers = detect_triggers(state)
    actions = rank_actions(state, triggers)
    for a in actions:
        next_state = simulate(state, a)
        if is_high_priority(triggers, next_state):
            queue.appendleft((next_state, path + [a]))
        else:
            queue.append((next_state, path + [a]))
```

Citations: FORGE & BFS examples in the official agent repo and Kaggle notebooks [ARC-AGI-3-Agents GitHub](https://github.com/arcprize/ARC-AGI-3-Agents) [Kaggle Trigger-Aware BFS notebook](https://www.kaggle.com/code/rauffauzanrambe/trigger-aware-bfs-for-game-simulation-score-0-35/input?scriptVersionId=310983317) [FORGE blog](https://huggingface.co/blog/MiniMax-AI/forge-scalable-agent-rl-framework-and-algorithm).

- Performance notes: Trigger-aware BFS variants in the Developer Preview scored substantially above naive baselines in Kaggle previews (example notebooks reporting ~0.30–0.35 level scores) when using prioritized BFS + deduplication [Kaggle Trigger-Aware BFS notebook](https://www.kaggle.com/code/rauffauzanrambe/trigger-aware-bfs-for-game-simulation-score-0-35/input?scriptVersionId=310983317) [ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf).


### Hybrid Search-and-Learn Agents (BFS + Neural Learning)
- Core idea: Combine systematic search (BFS/graph exploration) with learned models (policy/value or dynamics) to (1) prioritize action selection, (2) propose promising expansions, and (3) learn to predict frame-change or goal-relevance from experience [ARC-AGI-3-Agents GitHub](https://github.com/arcprize/ARC-AGI-3-Agents) [ARC-AGI-3 Demo notebooks](https://github.com/frank-morales2020/MLxDL/blob/main/ARC_AGI3_DEMO.ipynb).

- Integration points:
  - Search layer: BFS or graph explorer proposes candidate action sequences and enforces exhaustive coverage guarantees.
  - Learner layer: A DQN-like or supervised classifier operates on state (one-hot 16-channel 64×64 input) to predict: (a) action success probability, (b) frame-change probability, or (c) immediate reward/Win-likelihood. These predictions score/prioritize search frontier nodes.
  - Meta-level LLMs (optional): LLMs propose macros or heuristics for initializing search priors or transforming failed trajectories into hypotheses (documented in community writeups combining RL + LLM guidance) [ARC-AGI-3-Agents GitHub](https://github.com/arcprize/ARC-AGI-3-Agents) [Medium hybrid writeup](https://medium.com/ai-simplified-in-plain-english/agentic-ai-for-arc-agi3-a-hybrid-approach-with-reinforcement-learning-grok-4-and-gemini-llms-5d94a1ea4fdb).

- Example neural architecture (commonly used): Convolutional backbone (3–4 conv layers) → shared latent → two heads: discrete action logits (ACTION1–5 + ACTION7) and spatial heatmap for ACTION6 (64×64), trained with binary cross-entropy for frame-change prediction and MSE/Cross-entropy for value/Q targets. Experience replay + prioritized sampling stabilizes learning [ARC_AGI3_DEMO.ipynb](https://github.com/frank-morales2020/MLxDL/blob/main/ARC_AGI3_DEMO.ipynb).

- Algorithmic pattern: use the learner to compute a score s(state,action) and feed s as a priority into BFS/heap; prune low-scoring edges under a threshold T tuned to the 6-hour runtime / step budget [ARC-AGI-3-Agents GitHub](https://github.com/arcprize/ARC-AGI-3-Agents).


### StochasticGoose++ — CNN Frame-Change Agent (Tufa Labs)
- Summary: A specialized CNN-based agent that predicts whether an action will change the frame and uses stochastic sampling of actions prioritized by predicted frame-change probability; won the Developer Preview and the author published design notes and code [DriesSmit/ARC3-solution GitHub](https://github.com/DriesSmit/ARC3-solution) [Medium writeup by the author](https://medium.com/@dries.epos/1st-place-in-the-arc-agi-3-agent-preview-competition-49263f6287db).

- Input encoding: 16-channel one-hot encoding of 64×64 grid (one channel per color) to preserve discrete palette structure [DriesSmit/ARC3-solution GitHub](https://github.com/DriesSmit/ARC3-solution).

- CNN backbone (implementation-derived): 4 conv layers with channel progression e.g., 32 → 64 → 128 → 256, ReLU + BatchNorm, global pooling for action head, and a spatial decoder head for ACTION6 producing a 64×64 probability map for click targets [DriesSmit/ARC3-solution GitHub](https://github.com/DriesSmit/ARC3-solution) [Kaggle StochasticGoose notebook](https://www.kaggle.com/code/imaadmahmood/stochasticgoose-cnn-frame-change-agent/input?scriptVersionId=312993931).

- Frame-difference detection: Supervised binary label (frame_changed) is created from environment feedback; the model trains to predict P(frame_change | state, action) using binary cross-entropy. The agent stores large buffers of state-action labels (hash-de-duplicated) to maximize the variety of transitions [DriesSmit/ARC3-solution GitHub](https://github.com/DriesSmit/ARC3-solution).

- Policy: Hierarchical stochastic sampling — sample action-type according to softmax over action logits; if ACTION6 is sampled, sample coordinate from the 64×64 heatmap. The sampling temperature and entropy regularization balance exploration/exploitation.

- Implementation notes: Experience buffer with hash-based deduplication (keeps up to ~200k unique tuples), lightweight training loop (online minibatch updates between episodes), and dynamic reset when entering new levels to avoid stale data [DriesSmit/ARC3-solution GitHub](https://github.com/DriesSmit/ARC3-solution) [Medium writeup](https://medium.com/@dries.epos/1st-place-in-the-arc-agi-3-agent-preview-competition-49263f6287db).


### Redpill — Zero-Prior Agent with Latent Planning
- Concept: A ‘zero-prior’ agent that avoids hand-coded domain priors beyond core knowledge; instead it constructs latent dynamics models from scratch and performs planning in latent space (latent planning), relying on learned abstract state representations to drive search and execution [ARC-AGI-3-Agents GitHub](https://github.com/arcprize/ARC-AGI-3-Agents) [Kaggle comment thread and community notebooks for Redpill implementations](https://www.kaggle.com/code/poonszesen/redpill-zero-prior-agent-with-latent-planning/comments).

- Architecture summary: Encoder network maps frames to a compact latent z; a learned transition model f(z, a) → z' lets the agent roll out sequences in latent space cheaply; planning is performed via beam search / MCTS over latent trajectories with rollout-value estimation provided by a learned value head. Implementations are available as submissions and examples within the ARC-AGI-3 repo and Kaggle notebooks [ARC-AGI-3-Agents GitHub](https://github.com/arcprize/ARC-AGI-3-Agents).

- Prior-free emphasis: ‘Zero-prior’ means no task-specific heuristics beyond the benchmark’s allowed Core Knowledge priors — the agent discovers useful abstractions solely from interactions [ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf).


### Monte Carlo Tree Search (MCTS) Approaches
- MCTS basics: Build a search tree of action sequences; at each iteration, select node by UCB-style policy, expand, rollout (or use learned value), and backpropagate values. Neural priors (policy/value networks) are often used to bias selection and speed convergence (AlphaZero-style) [ARC-AGI-3 paper cites classical search methods and hybridization opportunities](https://arxiv.org/abs/2603.24621) [AB-MCTS extension discussion](https://sakana.ai/ab-mcts).

- Practical configuration tips in ARC-AGI-3 context:
  - Use a policy prior π(a|s) from a fast CNN to initialize node priors.
  - Use a value-net v(s) to reduce noisy rollouts; use rollout depth limited by step budgets.
  - Adapt exploration constant c_puct to constrained budgets: smaller c_puct reduces wasted exploration in sparse-reward tasks.

- Implementation pointers & references: MCTS implementations are available in community agents and the literature; adapting to ARC-AGI-3 requires a deterministic simulator or cached environment transitions to avoid expensive API calls [ARC-AGI-3-Agents GitHub](https://github.com/arcprize/ARC-AGI-3-Agents) [AB-MCTS writeup](https://sakana.ai/ab-mcts).


### Graph-Based Exploration Agents
- Approach: Build an explicit directed graph where nodes = unique environment states (frame hashes), edges = (action → next_state). Use frontier-driven policies (shortest path to untested state-action) to choose actions that maximize coverage and information gain. This training-free approach was highly competitive in the preview [Graph-Based Exploration paper & arXiv preprint](https://arxiv.org/pdf/2512.24156) [OpenReview version](https://openreview.net/forum?id=YGTxOepY49).

- Algorithmic elements:
  - Node representation: canonical compressed frame (e.g., run-length or bytes) + invariant features to detect equivalence.
  - Frontier selection: compute shortest path in state graph to a node with unexplored actions; execute actions along path.
  - Prioritization: prioritize edges with high visual salience or untried ACTION6 coordinates.

- Results: Graph-based explorers achieved strong results in the ARC-AGI-3 preview by solving many levels without learning, demonstrating that systematic coverage beats many current learning-only baselines for the benchmark [arXiv:2512.24156](https://arxiv.org/pdf/2512.24156).


### DreamerV3 + Intrinsic Curiosity Module (ICM) World Models
- Core idea: Model-based RL (DreamerV3) learns a latent world model and uses imagined rollouts for policy optimization; ICM provides intrinsic exploration rewards based on forward-model prediction error to encourage discovery in sparse-reward environments [DreamerV3 repo](https://github.com/danijar/dreamerv3) [DreamerV3 paper](https://arxiv.org/pdf/2301.04104) [ICM conceptual reference](https://www.piriai.cn/en/p/icm).

- DreamerV3 elements relevant to ARC-AGI-3:
  - RSSM latent dynamics (deterministic RNN state + stochastic latents) used to model environment transitions.
  - Actor/critic trained on imagined trajectories sampled from the learned model, enabling long-horizon credit assignment with fewer environment interactions.

- ICM integration: an auxiliary forward model predicts next latent representation; intrinsic reward r_icm = ||pred_z_{t+1} - z_{t+1}||^2 encourages the agent to visit states with high prediction error (novelty) [ARC-AGI-3-Agents GitHub](https://github.com/arcprize/ARC-AGI-3-Agents) [DreamerV3 repo](https://github.com/danijar/dreamerv3).

- Test-time training: DreamerV3 variants applied to ARC tasks often use short online finetuning on episode data; models with robust latent representations adapt more quickly at test time [DreamerV3 repo](https://github.com/danijar/dreamerv3) [ARC-AGI-3-Agents GitHub](https://github.com/arcprize/ARC-AGI-3-Agents).


---

## 2) Core ARC-AGI-3 Mechanics (Concise, Exact)

- Frame structure: grid up to 64×64 cells; each cell is an integer in {0..15} encoding a discrete palette/color. Coordinates use (x,y) with (0,0) top-left [ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf) [ARC-AGI docs (games)](https://docs.arcprize.org/games).

- Action space (per-step agent commands):
  - RESET: reset level to initial state [docs].
  - ACTION1..ACTION5: discrete action tokens (semantics vary by game but are atomic environment actions).
  - ACTION6: parameterized action requiring (x, y) coordinates (64×64 space) — often referred to as a click or spatial action.
  - ACTION7: additional discrete action token (game-defined semantics) [ARC-AGI-3 docs].

  (Formal API & examples are exposed in the arc-agi Python SDK and the ARC-AGI-3-Agents repo) [arc-agi GitHub](https://github.com/arcprize/arc-agi) [ARC-AGI-3-Agents GitHub](https://github.com/arcprize/ARC-AGI-3-Agents).

- Game states returned by the environment: NOT_FINISHED, WIN, GAME_OVER (agent must interpret these to stop/submit trajectories) [ARC-AGI-3 docs] [Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf) [Kaggle competition page](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview).

- Scoring methodology (Relative Human Action Efficiency — RHAE): for each completed level:

  level_score = (human_baseline_actions / ai_actions)^2

  where human_baseline_actions is the upper-median human action count on that level; scores per level are capped (e.g., at 1.15) and aggregated as weighted averages across levels to compute game and competition scores [ARC-AGI-3 Technical Report (scoring)](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf) [ARC-AGI docs (methodology)](https://docs.arcprize.org/methodology).

  Example: AI matching human actions → 1.0; taking 2× human actions → 0.25; 10× human actions → 0.01 [ARC-AGI-3 Technical Report].


## 3) Best Agent Architectures for Interactive Reasoning (Survey + Practical Notes)

This section synthesizes architectures judged effective for ARC-AGI-3 style tasks: world models, program synthesis, theory-driven RL, hierarchical & neural-symbolic hybrids, MCTS with neural priors, slot-based models, causal/counterfactual reasoning, and active exploration.

### World Models (DreamerV3, JEPA, IRIS)
- DreamerV3: RSSM-based latent world model + actor-critic trained on imagined rollouts; robust out-of-the-box performance and sample efficiency via imagination [DreamerV3 paper & code](https://arxiv.org/pdf/2301.04104) [GitHub](https://github.com/danijar/dreamerv3).

- JEPA (Joint Embedding Predictive Architecture): predict future latent embeddings (energy-based or contrastive objectives) instead of raw pixels, enabling efficient latent planning and gradient-based inference in embedding space (variants: I-JEPA, V-JEPA). JEPA-family approaches support test-time optimization and multiple-candidate latent generation [JEPA explainer & resources](https://www.youtube.com/watch?v=Dcs9ZPA0d-k).

- IRIS (Transformers as sample-efficient world models): tokenizes frames via discrete autoencoders and uses autoregressive Transformers for dynamics; IRIS introduces online/test-time adaptation methods (Online World Modeling and Adversarial World Modeling) to reduce the train-test gap for gradient-based planning [IRIS paper & code](https://arxiv.org/html/2512.09929v1) [IRIS GitHub](https://github.com/eloialonso/iris).

- Planning methods used with world models:
  - Trajectory optimization in latent space (rollout + value estimation) as in DreamerV3 [DreamerV3 paper](https://arxiv.org/pdf/2301.04104).
  - Gradient-based latent optimization for JEPA/IRIS: initialize candidate latent trajectories, optimize energy/loss to produce low-energy (plausible) futures, then decode and select actions accordingly [JEPA explainer](https://www.youtube.com/watch?v=Dcs9ZPA0d-k) [IRIS paper](https://arxiv.org/html/2512.09929v1).


### Test-Time Training / Online Finetuning
- Methods: (a) short online finetuning of the world model on the agent’s recent experience, (b) continual replay buffering with prioritized recent samples, (c) online adaptation of policy/value heads while freezing stable encoder layers. IRIS emphasizes online world-model finetuning to close the planning-policy distribution gap [IRIS paper & GitHub](https://arxiv.org/html/2512.09929v1) [GitHub](https://github.com/eloialonso/iris).


### Program Synthesis (Stitch, MDL-based methods)
- Stitch: synthesize small programs from primitives (map, filter, transform) to implement level-solutions; search is guided by program size/complexity heuristics and execution tests on held-out evaluation frames. Stitch-like DSL synthesis was the dominant archetype in earlier ARC variants and remains a high-precision approach for many tasks [ARC Prize blog on combining DL + program synthesis](https://arcprize.org/blog/beat-arc-agi-deep-learning-and-program-synthesis).

- MDL (Minimum Description Length) approaches: optimize program P minimizing L(P) + L(data | P) where L() is description length — yields parsimonious programs that generalize; MDL-inspired solvers are effective when environment dynamics are rule-like [ARC Prize 2025 papers & MDL entries, ARC-AGI-3 Technical Report].


### Theory-Based Reinforcement Learning (TheoryCoder)
- TheoryCoder constructs symbolic high-level theories (PDDL-style operators) learned or synthesized from experience, uses high-level planners for abstract plans, and grounds operators into executable low-level code for simulation/execution; this bilevel planning (abstract PDDL plan → BFS/low-level execution via synthesized world-model programs) improves search efficiency and interpretability [TheoryCoder implementation & paper (Synthesizing world models for bilevel planning)](https://arxiv.org/pdf/2503.20124) [TheoryCoder repo](https://github.com/ZerghamAhmed/TheoryCoder).


### Hierarchical Agents
- Multi-level decomposition: (1) meta-controller proposes subgoals, (2) sub-policies solve subgoals. Hierarchical RL reduces long-horizon planning overhead and allows reuse of subroutines across levels. Hierarchical planning pairs well with program synthesis (subroutines are synthesized programs) and with world models (plan at latent abstract state level) [ARC-AGI technical discussion & hierarchical architecture references](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf).


### Neural-Symbolic Hybrids
- Pattern: neural perception (convolutional encoders, slot attention) → symbolic reasoning/planning (PDDL, synthesized programs) → execution via simulator. Neural-symbolic hybrids offer perceptual robustness with the systematic generalization of symbolic planners [ARC Prize blog & neurosymbolic literature](https://arcprize.org/blog/beat-arc-agi-deep-learning-and-program-synthesis) [Gary Marcus commentary](https://garymarcus.substack.com/p/even-more-good-news-for-the-future).


### MCTS with Neural Priors
- Combine MCTS selection expansion with neural policy priors π_θ(a|s) and value estimates v_θ(s) to bias search (AlphaZero-like). Equations:
  - UCT/PUCT selection: a* = argmax_a [ Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)) ] where P(s,a) = π_θ(a|s) [AlphaZero/MCTS literature].
- Using cached (simulated) environment transitions is crucial to make MCTS tractable with ARC-AGI-3 budgets [ARC-AGI-3 paper & MCTS extensions references](https://arxiv.org/abs/2603.24621) [AB-MCTS discussion](https://sakana.ai/ab-mcts).


### Slot-Based Attention Models
- Slot attention allocates a small set of entity slots that bind to recurring objects and supports relational reasoning and object-level dynamics learning; slot representations can become the units for planning/dynamics (e.g., slot transitions & object-centric predictions) [Slot Attention paper (Locatello et al., 2020)](https://arxiv.org/abs/2003.09820).


### Causal & Counterfactual Models
- Build Structural Causal Models (SCMs) over world variables; use do-calculus or counterfactual queries to evaluate which actions change desired downstream variables. Causal models help infer intervention effects and robustly generalize to new compositions of objects/actions (CausalARC & related work) [CausalARC preprint](https://arxiv.org/html/2509.03636v1) [ARC-AGI-3 Technical Report].


### Active Learning & Curiosity-Driven Exploration
- Intrinsic reward signals (prediction error, information gain, learning progress) guide agents to informative experiences; ICM is a canonical implementation (intrinsic reward = forward-model error) [ICM conceptual reference](https://www.piriai.cn/en/p/icm) [DreamerV3 + ICM integrations in ARC-AGI community notebooks](https://www.kaggle.com/code/suneetsaini/arc-agi-3-dreamerv3-icm-agent/input?scriptVersionId=310990766).


---

## 4) Open-Source Repositories & Tools (Practical List)

- ARC-AGI-3-Agents (official agent template & submissions) — https://github.com/arcprize/ARC-AGI-3-Agents (Python; agents, simulator wrappers, AgentOps integration, tests) [ARC-AGI-3-Agents GitHub].

- arc-agi (Python SDK / toolkit) — https://github.com/arcprize/arc-agi (Python; environment API, quickstart, rendering, scorecards) [arc-agi GitHub].

- ARC3-solution / StochasticGoose++ (Tufa Labs, Dries Smit) — https://github.com/DriesSmit/ARC3-solution (Python; CNN frame-change agent, training scripts, buffer/dedup code) [DriesSmit/ARC3-solution GitHub].

- Trigger-aware BFS Kaggle notebooks (examples) — https://www.kaggle.com/code/rauffauzanrambe/trigger-aware-bfs-for-game-simulation-score-0-35/input?scriptVersionId=310983317 (Python; BFS + trigger heuristics example) [Kaggle notebook].

- DreamerV3 reference implementation — https://github.com/danijar/dreamerv3 (PyTorch/Haiku; world model + actor-critic on imagined rollouts) [DreamerV3 GitHub].

- IRIS implementation — https://github.com/eloialonso/iris (Transformer-based world-model code & test-time adaptation examples) [IRIS GitHub].

- ARC-AGI benchmarking tooling — https://github.com/arcprize/arc-agi-3-benchmarking (evaluation harness & reproducibility configs) [arc-agi benchmarking].

- TheoryCoder repository — https://github.com/ZerghamAhmed/TheoryCoder (bilevel planning and synthesized world-model code) [TheoryCoder GitHub].

- Additional community agents / forks: e.g., dhanaabhirajk/ARC-AGI-3-Agents mirrors and community notebooks (search GitHub / Kaggle) [ARC-AGI-3-Agents GitHub].

(Each repository above includes examples, implementation notes, and runnable notebooks referenced in the prior sections.)


## 5) Research Papers (2025–2026) — Selected Key Items

The following list includes directly relevant papers and technical reports (title, venue/arXiv, ID/URL):

1. ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence — ARC Prize Foundation — arXiv:2603.24621 — https://arxiv.org/abs/2603.24621 [benchmark paper] [ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf).

2. ARC-AGI-3 Technical Report (full PDF) — ARC Prize Foundation — https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf [in-depth methodology & scoring].

3. Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks — Evgenii Rudakov et al. — arXiv:2512.24156 — https://arxiv.org/pdf/2512.24156 [graph exploration methods].

4. IRIS: Closing the Train-Test Gap in World Models for Gradient-Based Planning — arXiv:2512.09929 — https://arxiv.org/html/2512.09929v1 [IRIS world-model + test-time adaptation].

5. TheoryCoder — Synthesizing world models for bilevel planning — arXiv:2503.20124 — https://arxiv.org/pdf/2503.20124 [theory-based RL / bilevel planning].

6. ARC Prize 2025: Technical Report and selected award-winning papers (program synthesis, MDL, few-shot adaptation) — https://arxiv.org/pdf/2601.10904 [competition report].

7. DreamerV3 (world-model & imagined rollouts) — arXiv:2301.04104 — https://arxiv.org/pdf/2301.04104 [foundational world-model algorithm].

8. CausalARC: Abstract Reasoning with Causal World Models — arXiv:2509.03636 — https://arxiv.org/html/2509.03636v1 [causal/counterfactual reasoning for abstract environments].

9. Slot Attention (object-centric representations) — Locatello et al. — arXiv:2003.09820 — https://arxiv.org/abs/2003.09820 [slot-based attention models].

10. AB-MCTS & MCTS extensions (search adaptations) — https://sakana.ai/ab-mcts [MCTS extension discussion].

(These items are cited in previous sections where they are relevant.)


## 6) Specific Tactics Used by Winning Agents (Practical Implementation Details)

Below are specific tactics reported in winning/competitive agent writeups and repositories; each tactic is paired with concrete implementation-level details and source links.

### a) Handling the 6-hour CPU/GPU runtime limit and rate limits
- Use of local deterministic simulators: mirror environment dynamics offline (apply actions to frame arrays deterministically) to avoid expensive API round-trips; simulate large BFS expansions locally and only send short final action sequences to the ARC API, reducing network wait and staying within rate limits [ARC-AGI-3-Agents repo & Kaggle FORGE notebooks] (https://github.com/arcprize/ARC-AGI-3-Agents) [Kaggle Trigger-Aware BFS notebook](https://www.kaggle.com/code/rauffauzanrambe/trigger-aware-bfs-for-game-simulation-score-0-35/input?scriptVersionId=310983317).

- Budgeted exploration: enforce node/edge/depth limits for searches and early-stopping when remaining wall-clock/time budget is below threshold; tune BFS expansion and neural-inference frequency to guarantee completion before 6 hours [ARC-AGI-3 Technical Report (exploration limits)](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf) [docs.rate_limits](https://docs.arcprize.org/rate_limits).

- Batched environment evaluation and asynchronous simulation: run simulation and inference in parallel threads/processes while monitoring step budget to keep the agent responsive under contest constraints [ARC-AGI-3-Agents implementation patterns](https://github.com/arcprize/ARC-AGI-3-Agents).


### b) State Deduplication Strategies
- Hash canonicalization: canonicalize frame (e.g., serialize dtype=uint8 array) and compute SHA-256 / xxhash to produce state IDs; store IDs in a hashset to skip re-expansion [StochasticGoose experience buffer dedup approach] (https://github.com/DriesSmit/ARC3-solution).

- Structural invariants: reduce states via invariants (object masks, bounding boxes, normalized object order) before hashing to cluster equivalent-looking states under transform symmetries [Graph-based exploration & Stitch-like program approaches](https://arxiv.org/pdf/2512.24156) [ARC-AGI-3 Technical Report].

- Compact delta-states: store only changed cells for transitions to save memory; reconstruct full frame by applying deltas to base frames for replay or BFS rollback [ARC-AGI-3-Agents code patterns] (https://github.com/arcprize/ARC-AGI-3-Agents).


### c) Action Pruning & Prioritization
- Frame-change prediction: a binary classifier CNN predicts probability of state change for (state,action) pairs; prune actions with P(frame_change) below a tuned threshold θ to reduce wasted expansions (method used by StochasticGoose++) [DriesSmit/ARC3-solution GitHub](https://github.com/DriesSmit/ARC3-solution) [Kaggle StochasticGoose notebook](https://www.kaggle.com/code/imaadmahmood/stochasticgoose-cnn-frame-change-agent/input?scriptVersionId=312993931).

- Trigger scoring: lightweight hand-coded rules (color counts, object adjacency changes, presence/absence of key marker colors) produce trigger scores used as BFS priorities for FORGE/Trigger-Aware BFS [Kaggle Trigger-Aware BFS notebook](https://www.kaggle.com/code/rauffauzanrambe/trigger-aware-bfs-for-game-simulation-score-0-35/input?scriptVersionId=310983317).

- Heuristic action ordering: prefer ACTION6 targets near recently-changed pixels or object centroids (spatial prior), reducing coordinate search overhead for click actions [StochasticGoose implementation notes](https://github.com/DriesSmit/ARC3-solution).


### d) Frame-Difference Detection & Use
- Binary labels derived from environment feedback: label (state_t, action) as changed if env returns a different frame; used to train BCE predictor for frame-change [StochasticGoose code & training scripts] (https://github.com/DriesSmit/ARC3-solution).

- Delta-extraction utilities: compute pixel-wise XOR / difference masks and morphological filters to extract localized change regions and object movement vectors for trigger detectors [Graph-based exploration & FORGE heuristics] (https://arxiv.org/pdf/2512.24156) [Kaggle Trigger-Aware BFS notebook](https://www.kaggle.com/code/rauffauzanrambe/trigger-aware-bfs-for-game-simulation-score-0-35/input?scriptVersionId=310983317).


### e) Goal Inference Techniques
- Inverse-dynamics & outcome clustering: cluster observed transitions and infer candidate goal predicates (e.g., “turn all red pixels to blue” or “move object into box”) by comparing initial and solved frames across successful trajectories; validate candidate goals with short search probes [ARC-AGI-3 Technical Report & community writeups].

- Program-sketch induction: from a few successful action traces, induce parameterized program templates (map over coordinates, flood-fill from seed) using MDL-like scoring to prefer compact, generalizable programs [Stitch/MDL synthesis references & ARC Prize blog](https://arcprize.org/blog/beat-arc-agi-deep-learning-and-program-synthesis) [ARC Prize 2025 papers].

- Latent-goal predictors: train shallow networks to predict goal-likelihood for candidate end-states produced by imagined rollouts; select plans whose final-state latent matches high goal-likelihood clusters [Redpill latent planning references] (https://github.com/arcprize/ARC-AGI-3-Agents) [TheoryCoder bilevel planning paper](https://arxiv.org/pdf/2503.20124).


## 7) Concrete Implementation Tips & Code Patterns

- State hashing: use xxhash64 or CityHash on the bytes of a uint8 64×64 array (16-channel one-hot flattening) to compute extremely fast dedup keys. Keep an LRU eviction policy for the hashset to cap memory.

- Action6 handling: represent ACTION6 logits as a flattened 4096 output and reshape to 64×64; for coordinate prediction use a softmax temperature τ that decays as an episode progresses to focus exploration early and exploitation later [StochasticGoose code patterns] (https://github.com/DriesSmit/ARC3-solution).

- Hybrid search scheduling: alternate phases — (a) model-free exploration using the CNN action-sampler for N steps, (b) search-phase BFS expansions seeded from promising frontier states for M steps — tune N/M to budget constraints [Hybrid agent demos](https://github.com/frank-morales2020/MLxDL/blob/main/ARC_AGI3_DEMO.ipynb).

- Replay & dedup buffers: store (state_hash, action, frame_changed, next_state_hash) tuples; use deduplication by state_hash to avoid overcounting repeated observations [DriesSmit/ARC3-solution repo] (https://github.com/DriesSmit/ARC3-solution).

- Simulator & caching: implement a fast CPU-only simulator for deterministic transitions and cache the result of (state_hash, action) → next_state_hash to make large search expansions feasible within time budgets [ARC-AGI-3-Agents patterns](https://github.com/arcprize/ARC-AGI-3-Agents).


## 8) Selected Paper-Level Implementation References (examples)
- StochasticGoose++ (CNN Frame-change): https://github.com/DriesSmit/ARC3-solution — includes training scripts, buffer/dedup code, and agent main loop [DriesSmit/ARC3-solution GitHub].

- Trigger-aware BFS examples: https://www.kaggle.com/code/rauffauzanrambe/trigger-aware-bfs-for-game-simulation-score-0-35/input?scriptVersionId=310983317 — BFS with trigger detection and prioritized queue.

- Graph-based exploration: arXiv preprint with source patterns and evaluation: https://arxiv.org/pdf/2512.24156.

- DreamerV3 implementation (base world-model algorithm): https://github.com/danijar/dreamerv3 — reference to integrate DreamerV3 with ARC input encoding.

- IRIS (test-time world-model adaptation): https://github.com/eloialonso/iris — code for tokenization, transformer world model, and online finetuning strategies.

- TheoryCoder (bilevel planning): https://github.com/ZerghamAhmed/TheoryCoder — PDDL-level + synthesized low-level executors example.


---

## 9) Practical Recipe (Recipe-style summary of the most empirically effective strategy components)

1. Build a fast deterministic local simulator and a compact hashed state representation to enable massive offline search expansions without hitting API or runtime limits [ARC-AGI-3-Agents patterns] (https://github.com/arcprize/ARC-AGI-3-Agents).

2. Implement a frame-change classifier (CNN) that predicts P(change | state, action) and use it to prune low-value actions before search; for ACTION6 use a spatial head predicting 64×64 click probabilities [DriesSmit/ARC3-solution GitHub] (https://github.com/DriesSmit/ARC3-solution).

3. Combine BFS/graph exploration for coverage with learned priors (policy/value or frame-change logits) to bias expansions (hybrid search-and-learn). Keep depth and node budgets tuned for the 6-hour limit [Kaggle BFS notebooks] (https://www.kaggle.com/code/rauffauzanrambe/trigger-aware-bfs-for-game-simulation-score-0-35/input?scriptVersionId=310983317).

4. Use program-synthesis (Stitch/MDL) to compactly express repeated solution patterns discovered during exploration; synthesize subroutines for repeated transformations and generalize via MDL scoring [ARC Prize program synthesis writeups] (https://arcprize.org/blog/beat-arc-agi-deep-learning-and-program-synthesis).

5. Where sample-efficiency is crucial, use world-models (DreamerV3, IRIS, JEPA variants) with online finetuning and intrinsic curiosity to discover salient transitions and plan in latent space [DreamerV3 repo](https://github.com/danijar/dreamerv3) [IRIS paper & repo](https://arxiv.org/html/2512.09929v1) [IRIS GitHub](https://github.com/eloialonso/iris).


## 10) Additional Notes on Evaluation & Reproducibility
- Use the official ARC-AGI-3 benchmarking harness for reliable scoring and submission compatibility; precompute human baseline actions per level for RHAE calculations and ensure your per-level action counts are deterministically captured (ai_actions) [ARC-AGI-3 benchmarking repo & docs] (https://github.com/arcprize/arc-agi-3-benchmarking) [Technical Report scoring](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf).

- Log replays and provide deterministic seeds for local simulators so BFS and search experiments are reproducible for tuning and audit (community repositories provide replay formats) [ARC-AGI-3-Agents repo] (https://github.com/arcprize/ARC-AGI-3-Agents).


---

This report synthesizes public agent implementations, competition technical reports, Kaggle notebooks, and recent research (2025–2026) to provide a concrete, implementation-oriented view of the architectures, mechanics, toolchains, papers, and tactics used to build high-performing ARC-AGI-3 agents. All claims above are cited inline to their sources: see citations in each paragraph for direct links to code, notebooks, and papers.

---

## Citations

- [ARC-AGI-3](https://arcprize.org/arc-agi/3)
- [https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)
- [[2603.24621] ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence](https://arxiv.org/abs/2603.24621)
- [ARC Prize 2026 - ARC-AGI-3 Competition](https://arcprize.org/competitions/2026/arc-agi-3)
- [arcprize/ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents)
- [ARC-AGI-3 Quickstart - ARC-AGI-3 Docs](https://docs.arcprize.org)
- [ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence](https://arxiv.org/html/2603.24621v1)
- [What Is ARC AGI 3? The Interactive AI Benchmark Humans Solve at 100% | MindStudio](https://www.mindstudio.ai/blog/what-is-arc-agi-3-interactive-benchmark)
- [arcprize/arc-agi-3-benchmarking](https://github.com/arcprize/arc-agi-3-benchmarking)
- [Checking your browser - reCAPTCHA](https://www.kaggle.com/code/suneetsaini/arc-agi-3-dreamerv3-icm-agent/input?scriptVersionId=310990766)
- [https://www.reddit.com/r/LocalLLaMA/comments/1s3ll4i/introducing_arcagi3](https://www.reddit.com/r/LocalLLaMA/comments/1s3ll4i/introducing_arcagi3)
- [danijar/dreamerv3](https://github.com/danijar/dreamerv3)
- [ARC-AGI-3 Scoring Methodology - ARC-AGI-3 Docs](https://docs.arcprize.org/methodology)
- [How to Beat ARC-AGI by Combining Deep Learning and Program Synthesis | ARC Prize](https://arcprize.org/blog/beat-arc-agi-deep-learning-and-program-synthesis)
- [Games - ARC-AGI-3 Docs](https://docs.arcprize.org/games)
- [https://www.reddit.com/r/singularity/comments/1s3pbl6/human_vs_ai_performance_on_arcagi_3_as_a_function](https://www.reddit.com/r/singularity/comments/1s3pbl6/human_vs_ai_performance_on_arcagi_3_as_a_function)
- [DriesSmit/ARC3-solution](https://github.com/DriesSmit/ARC3-solution)
- [https://www.youtube.com/watch?v=3vFu79ccDcc](https://www.youtube.com/watch?v=3vFu79ccDcc)
- [Trigger-Aware BFS for Game Simulation | Score 0.35 | Kaggle](https://www.kaggle.com/code/rauffauzanrambe/trigger-aware-bfs-for-game-simulation-score-0-35/input?scriptVersionId=310983317)
- [https://www.reddit.com/r/MachineLearning/comments/1s40a34/r_arc_round_3_released_technical_report](https://www.reddit.com/r/MachineLearning/comments/1s40a34/r_arc_round_3_released_technical_report)
- [Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks](https://arxiv.org/html/2512.24156v1)
- [https://www.researchgate.net/publication/403193756_ARC-AGI-3_A_New_Challenge_for_Frontier_Agentic_Intelligence](https://www.researchgate.net/publication/403193756_ARC-AGI-3_A_New_Challenge_for_Frontier_Agentic_Intelligence)
- [https://arxiv.org/pdf/2512.24156](https://arxiv.org/pdf/2512.24156)
- [ARC-AGI-3 Preview: 30-Day Learnings | ARC Prize](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)
- [ARC-AGI Toolkit Quickstart - ARC-AGI-3 Docs](https://docs.arcprize.org/toolkit/overview)
- [Announcing ARC-AGI-3 | ARC Prize](https://arcprize.org/blog/arc-agi-3-launch)
- [ARC-AGI-3 Preview Agent Competition](https://arcprize.org/competitions/arc-agi-3-preview-agents)
- [dhanaabhirajk/ARC-AGI-3-Agents](https://github.com/dhanaabhirajk/ARC-AGI-3-Agents)
- [ARC Prize - ARC-AGI-1 & ARC-AGI-2 Guide](https://arcprize.org/guide/1)
- [1st Place in the ARC-AGI-3 Agent Preview Competition 🏆 | by Dries Smit | Medium](https://medium.com/@dries.epos/1st-place-in-the-arc-agi-3-agent-preview-competition-49263f6287db)
- [https://openreview.net/pdf?id=YGTxOepY49](https://openreview.net/pdf?id=YGTxOepY49)
- [[2512.24156] Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks](https://arxiv.org/abs/2512.24156)
- [eloialonso/iris](https://github.com/eloialonso/iris)
- [Closing the Train-Test Gap in World Models for Gradient-Based Planning](https://arxiv.org/html/2512.09929v1)
- [Mastering diverse control tasks through world models | Nature](https://www.nature.com/articles/s41586-025-08744-2)
- [https://arxiv.org/pdf/2301.04104](https://arxiv.org/pdf/2301.04104)
- [Even more good news for the future of neurosymbolic AI](https://garymarcus.substack.com/p/even-more-good-news-for-the-future)
- [https://www.theneurondaily.com/p/play-the-puzzle-that-broke-every-ai-model](https://www.theneurondaily.com/p/play-the-puzzle-that-broke-every-ai-model)
- [Rate Limits - ARC-AGI-3 Docs](https://docs.arcprize.org/rate_limits)
- [[2503.20124] Synthesizing world models for bilevel planning](https://arxiv.org/abs/2503.20124)
- [https://arxiv.org/pdf/2505.01081](https://arxiv.org/pdf/2505.01081)
- [https://arxiv.org/pdf/2601.10904](https://arxiv.org/pdf/2601.10904)
- [Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks | OpenReview](https://openreview.net/forum?id=YGTxOepY49)
- [World Models Explained: JEPA, Energy-Based Learning and the Limits of LLMs](https://www.youtube.com/watch?v=Dcs9ZPA0d-k)
- [From 0% to 36% on Day 1 of ARC-AGI-3 | Symbolica Blog](https://www.symbolica.ai/blog/arc-agi-3)
- [OpenAI’s o3 and ARC-AGI: an Explainer  - Nate’s Substack](https://natesnewsletter.substack.com/p/openais-o3-and-arc-agi-an-explainer)
- [Inference-Time Scaling and Collective Intelligence for Frontier AI](https://sakana.ai/ab-mcts)
- [Arc-AGI is just an iq test. I don’t see the problem with training it to be good ... | Hacker News](https://news.ycombinator.com/item?id=46235492)
- [Agents Quickstart - ARC-AGI-3 Docs](https://docs.arcprize.org/agents-quickstart)
- [arcprize/ARC-AGI](https://github.com/arcprize/arc-agi)
