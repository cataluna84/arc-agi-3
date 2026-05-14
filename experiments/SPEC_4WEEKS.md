# SPEC_4WEEKS.md — D3 to D28 Implementation Spec (drafted 2026-05-01)

> **Audience**: any engineer or AI agent picking this up cold.
> **Predecessor docs** to read first:
> 1. `research/04_strategy_reset_2026-05-01.md` — the why.
> 2. `experiments/exp004_qwen_agent/POSTMORTEM.md` — what failed and why.
> 3. `experiments/EXPERIMENTS.md` — the existing 1-paragraph experiment table.
>
> **Purpose**: turn the 4-week plan from §4 of the strategy reset into a
> day-by-day, file-level, code-pattern-level spec. Each day has:
> *Goal*, *Files to create/modify*, *Implementation pattern*,
> *Smoke test*, *Submit?*, *Exit criteria*, *Rollback*.
>
> **Treat this file as append-only-by-day.** Once D7 is done, append
> "Done 2026-05-05; LB=0.34" rather than rewriting D7. New questions go
> in `.factory/memories.md`.

---

## 0. Conventions

### 0.1 Repo paths used in this spec

| Path                                                      | Purpose                                                |
|-----------------------------------------------------------|--------------------------------------------------------|
| `agents/`                                                 | One `.py` per agent class                              |
| `agents/state_graph.py`                                   | NEW — shared state-graph wrapper for all search agents |
| `agents/forge_agent.py`                                   | EXISTING — FORGE v19 verbatim port                     |
| `agents/chronos_agent.py`                                 | NEW (D3) — port of public CHRONOS notebook             |
| `agents/goose_agent.py`                                   | NEW (D4) — port of Stochastic Goose                    |
| `agents/trigger_bfs_agent.py`                             | NEW (D5) — port of FORGE-v16-trigger-aware-BFS         |
| `agents/goose_pp_agent.py`                                | NEW (D9) — Goose with frame-change CNN                 |
| `agents/forge_v20_agent.py`                               | NEW (D12) — combined CHRONOS + Go-Explore + CNN        |
| `agents/world_model.py`                                   | NEW (D24) — AXIOM-lite object-centric world model      |
| `experiments/expNNN_<slug>/`                              | One folder per experiment with README + score files    |
| `experiments/expNNN_<slug>/dev_kernel/`                   | Kaggle dev kernel (notebook + kernel-metadata.json)    |
| `experiments/expNNN_<slug>/comp_kernel/`                  | Kaggle competition kernel (separate slug)              |
| `scripts/collect_recordings.py`                           | NEW (D6) — dump (frame, action, next_frame) tuples     |
| `scripts/train_frame_change_cnn.py`                       | NEW (D8) — train CNN offline                           |
| `scripts/train_action6_head.py`                           | NEW (D11) — train ACTION6 click coord head             |
| `scripts/<agent>_smoke_local.py`                          | One per new agent class, 22-check parity smoke         |
| `data/recordings/`                                        | NEW — gitignored; collected trajectories (JSONL)       |
| `data/models/`                                            | NEW — gitignored; trained CNN weights (~5 MB each)     |

### 0.2 Daily ritual (every day)

1. **Morning**: read `.factory/memories.md` top section, current plan day.
2. **Build phase** (no Kaggle slot needed):
   - Edit code locally on the dev box.
   - Run `uv run ruff check .` and `uv run ruff format .`.
   - Run smoke tests (`scripts/<agent>_smoke_local.py` and
     `experiments/local_runner.py`).
3. **Decision: submit today?**
   - Submit days are explicitly marked below. On a build day, save the
     slot.
4. **Submission flow** (only on submit days):
   - Push dev kernel → poll until COMPLETE → review smoke output.
   - If smoke green: push comp kernel → poll → `kaggle competitions
     submit-code` → wait 24 h for LB.
5. **Evening**: append result (LB, per-game scores, per-level stats) to
   `.factory/memories.md` (NEW dated section AT TOP).
6. **Update** `experiments/SPEC_4WEEKS.md` with `Done YYYY-MM-DD; LB=X`.

### 0.3 Pre-flight checklist (before EVERY Kaggle push)

```bash
# All must pass before any kernel push:
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python scripts/<agent>_smoke_local.py
uv run python experiments/local_runner.py \
    --agent agents.<agent>:<Class> \
    --games ls20-mock --max-actions 50
```

If any fail → **fix locally, do not push**. The Kaggle kernel image is
expensive feedback (~20 min/iter).

### 0.4 Submission slot rules

- **1 successful submission per 24h**, enforced by Kaggle.
- Yesterday's slot was consumed by the Qwen `submission.parquet`
  scoring 0.00 (regression; never repeat).
- Today's slot (D3, 2026-05-01) is open and earmarked for the
  ForgeAgent variance probe (Track A from RUNBOOK_D2).
- On **build days**, do not push to comp_kernel; only dev_kernel.

### 0.5 LB-target interpretation

- Targets below are **median over 3 same-day runs** if available; otherwise
  single submission.
- A target met within ±0.02 counts as green.
- A target missed by < 0.05 → keep, run ablation.
- A target missed by ≥ 0.05 → rollback, diagnose before next iteration.

---

## 1. Week 1 (D3 – D7) — Match the Public Top-5

> **Week goal**: surface the public-LB ceiling (currently 0.42 Ash) into
> our codebase. Port CHRONOS + Stochastic Goose + Trigger-Aware BFS
> verbatim, then add a state-graph wrapper. **No novel ML.**
>
> **Target LB by D7**: 0.32 – 0.40.
>
> **Submission days**: D3 (Track A variance probe), D4 (Goose),
> D5 (Trigger-Aware BFS), D7 (BFS + state-graph). Build-only days: D6.

### 1.1 D3 (2026-05-01) — Track A: FORGE variance resubmit + CHRONOS read-through

#### Goal
Burn today's slot on the existing forked FORGE notebook (no code change)
to finally complete the variance probe from `.factory/plan.md` D1.
Concurrently, *read* CHRONOS source line-by-line and start the port.

#### Files
- READ: `experiments/exp002_forge_variance_probe/_pulled/ash-s-arc-agi-3-agent.ipynb`
  (already pulled by `scripts/resubmit_forge.sh`)
- CREATE: `agents/chronos_agent.py` (skeleton only today; no logic yet)
- CREATE: `experiments/exp005_chronos_port/` directory + `README.md`

#### Implementation pattern (skeleton for chronos_agent.py)
```python
"""chronos_agent.py - Port of the public CHRONOS notebook (forty2/chronos)
which the Ash 0.42 LB is forked from. Lazy-imports torch/numpy."""

from __future__ import annotations
from . import GameAction, GameState
from .agent import Agent  # base class

class ChronosAgent(Agent):
    MAX_ACTIONS = 100
    def __init__(self, game_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.game_id = game_id
        self._step = 0
        # TODO D5: state graph injection
        # TODO D9: frame-change CNN
        # TODO D11: ACTION6 head
    def is_done(self, frames, latest_frame): ...
    def choose_action(self, frames, latest_frame) -> GameAction: ...
```

#### Smoke test
```bash
uv run python -c "from agents.chronos_agent import ChronosAgent"  # import only
```

#### Submit?
**YES** — submit the FORGE notebook (kernel `cataluna84/ash-s-arc-agi-3-agent`)
that `scripts/resubmit_forge.sh` already pushed as version 2.

```bash
.venv/bin/kaggle kernels status cataluna84/ash-s-arc-agi-3-agent  # poll until COMPLETE
.venv/bin/kaggle competitions submit-code \
    -c arc-prize-2026-arc-agi-3 \
    --kernel cataluna84/ash-s-arc-agi-3-agent \
    --kernel-version 2 \
    -f submission.parquet \
    -m "exp002 D3 variance probe - FORGE baseline unchanged"
```

#### Exit criteria
- `experiments/exp002_forge_variance_probe/scores.json` has 2 LB data points.
- `agents/chronos_agent.py` exists with skeleton; passes
  `uv run ruff check`.
- `.factory/memories.md` has D3 dated section.

#### Rollback
- If kernel ERRORs: skip submission today, proceed to D4 a day late.
- If LB radically different from 0.19 (>0.30 or <0.10): record + still
  proceed to D4 unless < 0.05 (which would mean kernel is broken).

---

### 1.2 D4 — Stochastic Goose port + submit

#### Goal
Port the official `inversion/arc3-sample-submission-stochastic-goose`
agent (LB 0.25) to `agents/goose_agent.py`, smoke it, push to a
fresh comp kernel, submit.

#### Files
- CREATE: `agents/goose_agent.py`
- CREATE: `experiments/exp006_stochastic_goose/{dev_kernel,comp_kernel,README.md}`
- CREATE: `scripts/goose_agent_smoke_local.py` (22-check parity test)

#### Implementation pattern
The Stochastic Goose policy:
```python
class StochasticGoose(Agent):
    def choose_action(self, frames, latest):
        # 1) softmax over (untried_action_count, recent_change_score)
        # 2) for ACTION6, sample (x, y) uniformly from non-background
        #    pixel locations of the latest frame.
        avail = latest.available_actions  # list[GameAction]
        if not avail: return GameAction.RESET
        # weights: 1.0 baseline; 3.0 for actions never tried in this state
        # state hash = blake2b(grid.tobytes())[:8]
        h = self._state_hash(latest.frame[-1])
        tried = self._tried_per_state.setdefault(h, set())
        weights = [3.0 if a not in tried else 1.0 for a in avail]
        a = random.choices(avail, weights=weights, k=1)[0]
        tried.add(a)
        if a == GameAction.ACTION6:
            data = self._sample_click_xy(latest.frame[-1])
        else:
            data = {}
        return a, data
```

#### Smoke test
```bash
uv run python scripts/goose_agent_smoke_local.py    # expect 22/22 PASS
uv run python experiments/local_runner.py \
    --agent agents.goose_agent:StochasticGoose \
    --games ls20-mock --max-actions 100
```

#### Submit?
**YES** — push `experiments/exp006_stochastic_goose/comp_kernel` and
submit. Kernel slug: `cataluna84/goose-comp-arc-agi-3`.

#### Exit criteria
- LB ≥ 0.22 (allow ±0.03 from public 0.25).
- Per-action latency < 100 ms (Goose has no model).
- Per-game scores recorded in `experiments/exp006_stochastic_goose/scores.json`.

#### Rollback
- If LB < 0.20: bug in the port. Diff against public notebook source.
  Most likely culprit: state_hash collision or tried-set never resetting
  on level transition.

---

### 1.3 D5 — Trigger-Aware BFS port + submit

#### Goal
Port `aadigupta1601/0-35-forge-v16-trigger-aware-bfs` to
`agents/trigger_bfs_agent.py`. Submit. Target LB 0.32-0.35.

#### Files
- CREATE: `agents/trigger_bfs_agent.py`
- CREATE: `experiments/exp005_trigger_aware_bfs/{dev_kernel,comp_kernel,README.md}`
- CREATE: `scripts/trigger_bfs_smoke_local.py`

#### Implementation pattern
Core data structures:
```python
@dataclass
class StateNode:
    state_hash: bytes               # blake2b(grid.tobytes())[:8]
    frame: np.ndarray               # 64x64 uint8
    available_actions: list[int]
    tried: dict[int, bool]          # action_id -> action_changed_world
    visit_count: int
    score_at_first_visit: float

class TriggerBFSAgent(Agent):
    def __init__(self, game_id):
        self.graph: dict[bytes, StateNode] = {}
        self.frontier: deque[tuple[bytes, int]] = deque()  # (state_hash, action_id)

    def _trigger_score(self, prev_frame, next_frame, prev_score, next_score):
        delta_pixels = np.sum(prev_frame != next_frame)
        delta_score = next_score - prev_score
        new_color = len(set(next_frame.flat) - set(prev_frame.flat))
        return delta_pixels + 5 * delta_score + 2 * new_color

    def choose_action(self, frames, latest):
        h = self._hash(latest.frame[-1])
        if h not in self.graph:
            self.graph[h] = StateNode(...)  # init
        node = self.graph[h]
        # 1) untried actions first
        untried = [a for a in node.available_actions if a not in node.tried]
        if untried:
            a = self._highest_priority(untried, node)
            node.tried[a] = None  # mark; we'll fill in changed-flag next turn
            return GameAction.from_id(a), self._action_data(a, latest.frame[-1])
        # 2) BFS frontier - replay to known frontier state
        return self._replay_to_frontier()
```

#### Smoke test
```bash
uv run python scripts/trigger_bfs_smoke_local.py
uv run python experiments/local_runner.py \
    --agent agents.trigger_bfs_agent:TriggerBFSAgent \
    --use-sdk --games ls20 --max-actions 200
# Expect: levels_completed >= 1 within 100 actions.
```

#### Submit?
**YES** — push comp kernel `cataluna84/trigger-bfs-comp-arc-agi-3`.
Submit.

#### Exit criteria
- LB ≥ 0.30.
- ls20 smoke completes ≥ 2 levels in ≤ 300 actions.
- Per-action latency < 50 ms.

#### Rollback
- If LB < 0.27: most likely the trigger formula is mis-tuned. Try
  `5*delta_score + 1*delta_pixels + 3*new_color`.
- If LB < 0.20: there's a state-hash bug (collisions or no reset on level
  transition). Read `aadigupta1601` source and diff.

---

### 1.4 D6 — `state_graph.py` shared module + recording collector

#### Goal
Build day, no submission. Extract the state-graph code from
`trigger_bfs_agent.py` into a reusable module, build a recording
collector that writes JSONL traces of any agent run.

#### Files
- CREATE: `agents/state_graph.py`
- CREATE: `scripts/collect_recordings.py`
- MODIFY: `agents/trigger_bfs_agent.py` (use state_graph.StateGraph)
- MODIFY: `agents/forge_agent.py` (gain state_graph injection point)
- CREATE: `tests/test_state_graph.py` (pytest unit tests)

#### Implementation pattern
```python
# agents/state_graph.py
import hashlib, numpy as np
from collections import deque
from dataclasses import dataclass, field

def hash_frame(frame: np.ndarray) -> bytes:
    return hashlib.blake2b(frame.astype(np.uint8).tobytes(),
                           digest_size=8).digest()

@dataclass
class StateNode:
    state_hash: bytes
    visit_count: int = 0
    untried_actions: set[int] = field(default_factory=set)
    edges: dict[int, bytes] = field(default_factory=dict)  # action_id -> next state_hash
    last_score: float = 0.0
    last_levels: int = 0

class StateGraph:
    def __init__(self):
        self.nodes: dict[bytes, StateNode] = {}
        self.frontier: deque[bytes] = deque()
        self.path_from_root: list[tuple[bytes, int]] = []

    def observe(self, prev_frame, action_id, next_frame, score, levels):
        prev_h = hash_frame(prev_frame)
        next_h = hash_frame(next_frame)
        self.nodes[next_h] = self.nodes.get(next_h, StateNode(state_hash=next_h))
        # update edge
        self.nodes[prev_h].edges[action_id] = next_h
        self.nodes[prev_h].untried_actions.discard(action_id)
        # frontier mgmt
        if next_h not in self.nodes or self.nodes[next_h].visit_count == 0:
            self.frontier.append(next_h)
        self.nodes[next_h].visit_count += 1

    def reset_on_level_transition(self, current_levels):
        """Per ARC rules, level transitions invalidate the state graph
        because mechanics may change."""
        for node in self.nodes.values():
            if node.last_levels != current_levels:
                self.nodes.clear()
                self.frontier.clear()
                self.path_from_root.clear()
                return
```

```python
# scripts/collect_recordings.py
"""Run any agent on any game and dump (frame, action, next_frame, ...)
tuples as JSONL for offline training of CNN/world-model components.

Usage:
    .venv/bin/python scripts/collect_recordings.py \
        --agent agents.trigger_bfs_agent:TriggerBFSAgent \
        --games ls20,vc33,ft09 \
        --episodes 5 \
        --out data/recordings/trigger_bfs.jsonl
"""
import argparse, json, importlib, sys, pathlib, datetime
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--games", required=True)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--max-actions", type=int, default=300)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mod_name, cls_name = args.agent.rsplit(":", 1)
    AgentCls = getattr(importlib.import_module(mod_name), cls_name)

    # ... use experiments/local_runner.py infrastructure with --use-sdk
    # write one JSONL line per (state, action, next_state) transition:
    # {"game": "ls20", "step": 7, "frame": [[...]], "action": 1,
    #  "data": {}, "next_frame": [[...]], "score": 0.0, "levels": 0,
    #  "changed": True, "wall_clock_ms": 12.4}
```

#### Smoke test
```bash
uv run pytest tests/test_state_graph.py -v
uv run python scripts/collect_recordings.py \
    --agent agents.random_agent:RandomAgent \
    --games ls20-mock --episodes 1 --out /tmp/test_rec.jsonl
wc -l /tmp/test_rec.jsonl  # > 50
```

#### Submit?
**NO** — build day. Save the slot.

#### Exit criteria
- All pytest tests pass.
- `state_graph.py` covered by ≥ 5 unit tests (hash determinism, level-reset,
  edge insertion, frontier ordering, path reconstruction).
- `collect_recordings.py` produces valid JSONL.

---

### 1.5 D7 — BFS + state-graph + new-action prioritization, submit

#### Goal
Wire `state_graph.StateGraph` into `agents/trigger_bfs_agent.py` so that
the BFS frontier is ordered by (untried-action count desc, trigger-score desc).
Submit. Target LB 0.36 – 0.40.

#### Files
- MODIFY: `agents/trigger_bfs_agent.py` to use `StateGraph`.
- CREATE: `experiments/exp007_bfs_state_graph/{dev,comp}_kernel/`

#### Implementation pattern
Replace the local `self.graph: dict[...]` with `self.graph = StateGraph()`,
and replace the frontier-pop logic with:
```python
def _next_frontier_state(self):
    # priority = (#untried_actions desc, trigger_score desc, visit_count asc)
    candidates = [
        (
            len(self.graph.nodes[h].untried_actions),
            self.graph.nodes[h].last_score,
            -self.graph.nodes[h].visit_count,
            h
        )
        for h in self.graph.frontier
    ]
    candidates.sort(reverse=True)
    return candidates[0][3]  # state_hash
```

#### Smoke test
```bash
uv run python experiments/local_runner.py \
    --agent agents.trigger_bfs_agent:TriggerBFSAgent \
    --use-sdk --games ls20 --max-actions 300
# Expect: levels_completed >= 2 within 200 actions.
```

#### Submit?
**YES**.

#### Exit criteria
- LB ≥ 0.34.
- ls20 ≥ 2 levels in 200 actions.

#### Rollback
- If LB ≤ 0.30: state_graph integration is broken. Diff vs D5's working
  trigger_bfs and bisect.

---

## 2. Week 2 (D8 – D14) — Memory + Go-Explore + Frame-change CNN

> **Week goal**: add the components that turn 0.30-LB-class agents into
> 0.40-LB-class agents: persistent state memory, Go-Explore archive,
> frame-change CNN.
>
> **Target LB by D14**: 0.40 – 0.45.
>
> **Submission days**: D9 (Goose++), D12 (combined v20). Build-only:
> D8, D10, D11, D13, D14.

### 2.1 D8 — Frame-change CNN training infra

#### Goal
Train a small CNN offline on collected recordings to predict
`P(action changes frame | state, action) ∈ [0, 1]` per (state, action).

#### Files
- CREATE: `agents/cnn_models.py` (FrameChangeCNN class)
- CREATE: `scripts/train_frame_change_cnn.py`
- CREATE: `data/recordings/` (gitignored)
- CREATE: `data/models/frame_change_v1.pt` (output)
- MODIFY: `pyproject.toml` to add `torch>=2.4`, `numpy>=1.26` if not
  already there.

#### Implementation pattern
```python
# agents/cnn_models.py
import torch
import torch.nn as nn

class FrameChangeCNN(nn.Module):
    """Input: (B, 16, 64, 64) one-hot grid + (B, 7) action one-hot.
    Output: (B, 1) sigmoid - probability the action will change the frame."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),  # 32x32
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU(), # 16x16
            nn.Conv2d(128, 128, 3, padding=1, stride=2), nn.ReLU(),# 8x8
            nn.AdaptiveAvgPool2d(1),                               # 1x1
        )
        self.action_proj = nn.Linear(7, 64)
        self.head = nn.Sequential(
            nn.Linear(128 + 64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, frame_onehot, action_onehot):
        z = self.conv(frame_onehot).flatten(1)  # (B, 128)
        a = self.action_proj(action_onehot)     # (B, 64)
        return torch.sigmoid(self.head(torch.cat([z, a], dim=1)))
```

```python
# scripts/train_frame_change_cnn.py
"""Train FrameChangeCNN on JSONL recordings.

Loss: BCE on `changed` flag.
Augmentations applied at training time:
    A1. Group symmetries (D4) - 8x multiplier (4 rots x 2 reflects).
    A2. Color permutations - sample 16-color permutation per batch.
    A4. Cellular automata perturbations on a random subset of cells.
        (small probability eps=0.02 per cell to apply Rule 30 step.)

Usage:
    .venv/bin/python scripts/train_frame_change_cnn.py \
        --recordings data/recordings/*.jsonl \
        --epochs 20 --batch 64 \
        --out data/models/frame_change_v1.pt
"""
```

#### Smoke test
```bash
# Generate a small recording set (~1k tuples) using random+goose:
uv run python scripts/collect_recordings.py \
    --agent agents.goose_agent:StochasticGoose \
    --games ls20-mock,vc33-mock,ft09-mock \
    --episodes 10 --out data/recordings/seed.jsonl
# Train for 1 epoch, just to verify pipeline:
uv run python scripts/train_frame_change_cnn.py \
    --recordings data/recordings/seed.jsonl \
    --epochs 1 --batch 16 --out /tmp/test.pt
```

#### Submit?
**NO** — build day.

#### Exit criteria
- 1-epoch training converges (loss decreases monotonically).
- Augmentation A1 (rotations) verified in unit test.
- `frame_change_v1.pt` < 5 MB.

---

### 2.2 D9 — Goose++ with frame-change CNN, submit

#### Goal
Port `imaadmahmood/stochasticgoose-cnn-frame-change-agent` to
`agents/goose_pp_agent.py`. Use the trained CNN from D8.

#### Files
- CREATE: `agents/goose_pp_agent.py`
- CREATE: `experiments/exp009_goose_pp/{dev,comp}_kernel/`
- CREATE: Kaggle Dataset for `data/models/frame_change_v1.pt` + bundle
  cnn_models.py: `cataluna84/arc-agi-3-frame-change-cnn`.

#### Implementation pattern
```python
class GoosePP(Agent):
    def __init__(self, game_id):
        super().__init__(game_id)
        self.cnn = self._load_cnn()  # from /kaggle/input/...
        self.tried = {}  # state_hash -> set[action_id]

    def choose_action(self, frames, latest):
        h = blake2b(latest.frame[-1].tobytes())[:8]
        avail = latest.available_actions
        # Compute p_change for each action
        p_change = self._cnn_predict(latest.frame[-1], avail)
        # weight = p_change * (3 if untried_in_state else 1)
        tried_set = self.tried.setdefault(h, set())
        weights = [
            p_change[a.value] * (3.0 if a not in tried_set else 1.0)
            for a in avail
        ]
        a = random.choices(avail, weights=weights, k=1)[0]
        tried_set.add(a)
        return a, self._sample_data(a, latest.frame[-1])
```

#### Smoke test
```bash
uv run python scripts/goose_pp_smoke_local.py    # 22 checks
uv run python experiments/local_runner.py \
    --agent agents.goose_pp_agent:GoosePP \
    --use-sdk --games ls20 --max-actions 200
```

#### Submit?
**YES** — push comp kernel `cataluna84/goose-pp-comp-arc-agi-3`.

#### Exit criteria
- LB ≥ 0.30 (matches public 0.32 modulo our CNN training data).
- Per-action latency < 100 ms (CNN forward pass on H100).

#### Rollback
- If LB < 0.27: most likely CNN training data is too narrow. Re-collect
  recordings on a broader agent set (random+goose+chronos).

---

### 2.2.1 D9 retrospective (Done 2026-05-07)

What actually happened on D9: not the planned `goose-pp-comp-arc-agi-3`
fresh build (the D8 frame-change CNN training was never executed because
D5-D6 burned the budget on master_v7 + Goose CNN v1). Instead D9 was a
**defensive resubmission of the existing exp007 Goose CNN v1** which had
scored **LB 0.00** on D6 due to a silent comp-rerun crash.

D9 v2 fixes (applied to both `agents/goose_cnn_agent.py` and the inlined
notebook): `enable_gpu=false`, outer `try/except` in `choose_action()`
falling back to a uniform random non-RESET non-ACTION6 action,
`try/except` around `predictor.predict()` falling back to uniform
priors, and a defensive `cur_levels >= 0` guard against transient
gateway `levels_completed=-1`. Pre-submission checks all PASS (ruff,
40/40 pytest, 22/22 goose smoke, local_runner). Pushed v2 (kernel
COMPLETE in ~40s on CPU vs failed GPU init); submitted at 2026-05-08
00:23 UTC. Result landed 2026-05-13: **LB = 0.17**.

Decision rule result:
- LB ≥ 0.20 → v1 0.00 was a packaging bug, confirmed; proceed to D10
  frame-segmenter port.
- **0.10 ≤ LB < 0.20 → agent runs but priors do not lift over random.**
  We did not retrain the CNN; we pivoted to structural priors via the
  D10+D11 frame-segmenter port.
- LB ≤ 0.05 → still failing silently; cell-by-cell diff vs StochasticGoose
  1st-place repo (`DriesSmit/ARC3-solution`, 12.58% private LB).

### 2.3 D10+D11 — Frame-segmenter port (Done 2026-05-13)

#### Goal (revised) — DONE
Build day. Port the dolphin-in-a-coma frame-segmentation algorithm from
the graph-exploration paper (Rudakov 2026, arXiv:2512.24156). The paper
reports their approach solves 19/52 levels ≈ **0.36 LB** on the private
set, while master_v7 sits at 0.21 — the implementation is small (~200
LOC) and the gap is large. Wire it as the saliency-tier-0 prior for
ACTION6 click-coord sampling in `agents/trigger_bfs_agent.py`.

(Original "Go-Explore archive" remains as a fallback if the
frame-segmenter underperforms, but it is no longer the primary D10 goal.)

#### Done 2026-05-13
- `agents/frame_segmenter.py` (~440 LOC): stateless port. Public:
  `segment_frame`, `identify_status_bars`,
  `frame_segments_to_priority_tiers`, `hash_masked_frame`,
  `mask_to_click_coords`, `salient_pixels_in_segment`.
- `tests/test_frame_segmenter.py` (11 tests, all PASS).
- `agents/trigger_bfs_agent.py` `_sample_click_xy` wired to walk tiers
  0..3 (skip tier 4 = status bars); within each tier, pick a
  non-dominant segment, then a uniform pixel within it. Defensive
  try/except.
- Smokes 22/22, full test suite 54/54, ruff clean.

#### Files
- CREATE: `agents/go_explore.py` (Archive + GoStep + ExploreStep)
- MODIFY: `agents/trigger_bfs_agent.py` to optionally use the archive

#### Implementation pattern
```python
# agents/go_explore.py
class GoExploreArchive:
    """Archive of cells. Cell = downscaled 16x16 frame hash.
    For each cell, store the BEST trajectory found (shortest path)."""
    def __init__(self, downscale=4):
        self.downscale = downscale  # 64 // 4 = 16
        self.cells: dict[bytes, Trajectory] = {}

    def cell_id(self, frame):
        small = frame[::self.downscale, ::self.downscale]  # 16x16
        return blake2b(small.tobytes())[:6]

    def add(self, frame, traj_so_far):
        c = self.cell_id(frame)
        if c not in self.cells or len(traj_so_far) < len(self.cells[c]):
            self.cells[c] = traj_so_far[:]

    def select_for_explore(self) -> bytes:
        """Probabilistic: prefer cells with low visit count and high
        novelty (untried-action count)."""
        weights = [(1.0 / (1 + cell.visit_count) * cell.untried_count)
                   for cell in self.cells.values()]
        ...
```

```python
class GoExploreAgent(Agent):
    """Composite agent: alternates between go-step (replay best trajectory
    to a selected cell) and explore-step (random/CNN-weighted exploration
    from that cell)."""
    def __init__(self, ..., explore_policy=None, max_explore_steps=20):
        self.archive = GoExploreArchive()
        self.mode = "explore"  # or "go"
        self.go_target_cell = None
        self.go_replay_step = 0
        ...
```

> **Important note about COMPETITION mode**: SDK §0.9.3 says only level
> resets, no game resets. Go-Explore's "go-step" replays the action
> trajectory; we cannot use simulator-state restore. Replay = re-issuing
> the same action sequence from the level's reset state.

#### Smoke test
```bash
uv run pytest tests/test_go_explore.py -v
uv run python experiments/local_runner.py \
    --agent agents.go_explore:GoExploreAgent \
    --use-sdk --games ls20 --max-actions 500
```

#### Submit?
**NO**.

#### Exit criteria
- Replay mechanism verified to be deterministic (same trajectory
  produces same final state).
- ≥ 5 cells discovered on ls20 in 500 actions.

---

### 2.4 D11 — ACTION6 click-coord head

#### Goal
Build day. Train a small ConvDecoder that predicts an `(H, W)` click
heatmap from the current frame, trained on (state, ACTION6 click that
moved the score).

#### Files
- CREATE: `agents/click_head.py` (ClickHeatmapNet)
- CREATE: `scripts/train_action6_head.py`
- CREATE: `data/models/click_head_v1.pt`

#### Implementation pattern
```python
# agents/click_head.py
class ClickHeatmapNet(nn.Module):
    """Input: (B, 16, 64, 64) one-hot.
    Output: (B, 1, 64, 64) - softmax click heatmap over the grid."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Conv2d(64, 1, 1)  # 1x1 conv for logits

    def forward(self, frame_onehot):
        z = self.encoder(frame_onehot)
        logits = self.decoder(z)  # (B, 1, 64, 64)
        return logits

    def sample_xy(self, frame_onehot, top_k=8):
        """Returns one (x, y) sampled from softmax(logits) flat."""
        with torch.no_grad():
            logits = self.forward(frame_onehot.unsqueeze(0)).flatten()
            top = torch.topk(logits, top_k)
            chosen = torch.multinomial(F.softmax(top.values, dim=0), 1)
            idx = top.indices[chosen.item()]
            return divmod(idx.item(), 64)  # (y, x)
```

Training data filter: only `(state, ACTION6, x, y)` tuples where
`score_after > score_before`.

#### Smoke test
```bash
uv run python scripts/train_action6_head.py \
    --recordings data/recordings/*.jsonl --epochs 5 \
    --out data/models/click_head_v1.pt
```

#### Submit?
**NO**.

---

### 2.5 D12 — `forge_v20_agent.py` = CHRONOS + state-graph + Goose++ + Go-Explore + ACTION6 head, submit

#### Goal
Combine all week-1 + week-2 components into a single agent. This is the
"target the public 0.42" agent.

#### Files
- CREATE: `agents/forge_v20_agent.py`
- CREATE: `experiments/exp012_forge_v20/{dev,comp}_kernel/`

#### Implementation pattern
```python
class ForgeV20Agent(Agent):
    def __init__(self, game_id):
        self.graph = StateGraph()
        self.archive = GoExploreArchive()
        self.cnn = FrameChangeCNN.from_pretrained("frame_change_v1.pt")
        self.click = ClickHeatmapNet.from_pretrained("click_head_v1.pt")
        self.mode = "explore"
        self._step = 0

    def choose_action(self, frames, latest):
        # 1) state graph update
        self.graph.observe(...)
        self.archive.add(latest.frame[-1], self._traj_so_far)

        # 2) decide: go or explore?
        if self._stuck() or self._step % 50 == 0:
            self.mode = "go"
            self.go_target = self.archive.select_for_explore()

        if self.mode == "go":
            return self._replay_step()  # follow archive trajectory

        # 3) explore: pick action by combined priority
        avail = latest.available_actions
        priorities = self._compute_priorities(latest.frame[-1], avail)
        a = self._sample_by_priority(avail, priorities)
        if a == GameAction.ACTION6:
            data = {"x": x, "y": y}  # via click_head
        return a, data
```

#### Smoke test
```bash
uv run python scripts/forge_v20_smoke_local.py
uv run python experiments/local_runner.py \
    --agent agents.forge_v20_agent:ForgeV20Agent \
    --use-sdk --games ls20,vc33,ft09 --max-actions 300
# Expect: median levels_completed across 3 games >= 3.
```

#### Submit?
**YES** — push `cataluna84/forge-v20-comp-arc-agi-3`. **Target LB 0.42**.

#### Exit criteria
- LB ≥ 0.40.
- Per-game scores match or exceed ash-s-arc-agi-3-agent (0.42).
- Per-action latency < 200 ms (combined model + search).
- Submission completes in < 4h Kaggle wall.

#### Rollback
- If LB < 0.35: regression vs. trigger-BFS alone (D7=0.36-0.40). Means
  one of the new components is hurting. Bisect by ablating CNN, archive,
  click head independently in next dev kernels.

---

### 2.6 D13 — Variance ablation

Build day. Submit nothing. Run the dev kernel 5 times with different
fixed seeds, record per-game stddev. Goal: identify whether
forge_v20 is variance-dominated.

### 2.7 D14 — Component ablation

Build day. Submit nothing. Run dev kernel with each component disabled
in turn. Goal: identify which components contribute most. Update spec
weights based on result.

---

## 3. Week 3 (D15 – D21) — Object-centric Perception + Adaptive Planning

> **Week goal**: extract object-centric representations from the 64×64
> grid, plug them into a learned planner, push past the public ceiling.
>
> **Target LB by D21**: 0.45 – 0.50.
>
> **Submission days**: D17, D20. Build-only: D15, D16, D18, D19, D21.

### 3.1 D15 — Object segmentation (connected components)

#### Goal
Build day. Implement per-color connected-component labeling and a
clean Object dataclass.

#### Files
- CREATE: `agents/objects.py`
- CREATE: `tests/test_objects.py`

```python
# agents/objects.py
from scipy.ndimage import label, find_objects, center_of_mass

@dataclass
class Object:
    obj_id: int
    color: int
    bbox: tuple[slice, slice]
    centroid: tuple[float, float]
    size: int                      # cell count
    shape_hash: bytes              # hash of binary mask within bbox

def segment_frame(frame: np.ndarray) -> list[Object]:
    """Per-color connected components. Returns sorted by descending size."""
    out = []
    for color in range(16):
        mask = (frame == color)
        if not mask.any(): continue
        labeled, n = label(mask)
        for cc_id in range(1, n + 1):
            cc_mask = (labeled == cc_id)
            ...
            out.append(Object(...))
    out.sort(key=lambda o: -o.size)
    return out
```

#### Smoke test
```bash
uv run pytest tests/test_objects.py -v
# Validates: 5 random frames have correct object counts, centroids,
# shape hashes are stable.
```

### 3.2 D16 — Object tracking across frames

#### Goal
Build day. Match Objects across consecutive frames (same shape_hash =
likely same object; allow centroid jitter ≤ 5 cells).

#### Files
- CREATE: `agents/object_tracker.py`

### 3.3 D17 — Object-aware ACTION6 sampling, submit

#### Goal
Replace random ACTION6 (x, y) with sampling from object centroids
weighted by `(size * saliency_tier)`. Combine with click-head as a
prior.

#### Files
- MODIFY: `agents/forge_v20_agent.py`
- CREATE: `experiments/exp013_object_aware/{dev,comp}_kernel/`

#### Submit?
**YES** — target LB 0.43.

### 3.4 D18-D19 — World-model components

#### Goal
Build days. Train a small per-object dynamics CNN: given current
Object + action, predict next Object position/state.

#### Files
- CREATE: `agents/object_dynamics.py`
- CREATE: `scripts/train_object_dynamics.py`

### 3.5 D20 — World-model-guided rollout, submit

#### Goal
Use trained dynamics model for 3-5 step lookahead beam search before
committing to an action.

#### Files
- CREATE: `agents/lookahead_planner.py`
- CREATE: `experiments/exp015_world_model_planner/{dev,comp}_kernel/`

#### Submit?
**YES** — target LB 0.46.

### 3.6 D21 — Per-game telemetry analysis

Build day. Analyze where forge_v20 + world-model wins/loses per game.
Identify games where ACTION6 dominates vs. games where it never helps.
Decide whether to add a per-game dispatcher in week 4.

---

## 4. Week 4 (D22 – D28) — AXIOM-lite + Test-time Adaptation

> **Week goal**: bring the AXIOM (Heins 2026) idea — object-centric
> Bayesian world models that grow online — into our pipeline. Add
> per-task LoRA test-time training.
>
> **Target LB by D28**: 0.48 – 0.55.
>
> **Submission days**: D24, D27. Build-only: D22, D23, D25, D26, D28.

### 4.1 D22-D23 — Bayesian slot mixture model

#### Goal
Build days. Implement a minimal AXIOM-lite: object slots tracked via
GMM + per-object piecewise-linear dynamics + interaction mixture.
Trim AXIOM heavily; keep only the parts that fit in 5 MB compiled.

#### Files
- CREATE: `agents/world_model.py` (sMM + tMM + rMM as in AXIOM Eq. 1)
- CREATE: `tests/test_world_model.py`

### 4.2 D24 — World-model-only agent, submit

#### Goal
Pure AXIOM-lite agent (no learned CNN). Tests whether Bayesian online
learning alone is enough.

#### Files
- CREATE: `agents/axiom_lite_agent.py`
- CREATE: `experiments/exp017_axiom_lite/{dev,comp}_kernel/`

#### Submit?
**YES** — target LB 0.40 (likely baseline-ish but informative).

### 4.3 D25-D26 — Test-time LoRA on tiny transformer

#### Goal
Build days. Take a 5-10M-param transformer (e.g. `microsoft/phi-2-sm`
distilled or a TRM clone). At each level transition, fine-tune via
LoRA on the trajectory of the *previous* level (HER-style, every
reached state is a goal). Predict next action given current state.

#### Files
- CREATE: `agents/ttt_transformer.py`
- CREATE: `scripts/pretrain_arc_transformer.py`

### 4.4 D27 — Combined AXIOM + TTT + forge_v20 cascade, submit

#### Goal
Per-game cascade:
- Level 1-2: forge_v20 (cheap, fast).
- Level 3+: escalate to AXIOM-lite + TTT-LoRA.

#### Files
- CREATE: `agents/cascade_agent.py`
- CREATE: `experiments/exp019_cascade/{dev,comp}_kernel/`

#### Submit?
**YES** — target LB 0.50.

### 4.5 D28 — Stocktake + week-5 plan

Build day. Read all per-game LB telemetry from D3-D27. Decide between:

- **Option A**: continue scaling AXIOM (week 5 = exp020 deeper world model).
- **Option B**: train per-game LoRA offline on collected recordings (week
  5 = pre-train phase).
- **Option C**: hybridize with a small text-only Qwen3-7B verifier
  (week 5 = revive Tier-C VLM idea but as verifier, not policy).

Update `research/05_strategy_d29plus.md` with chosen option.

---

## 5. Cross-cutting concerns

### 5.1 Kaggle Datasets to maintain

| Dataset slug                                          | Owner       | Bytes   | Refresh cadence                                      |
|-------------------------------------------------------|-------------|---------|------------------------------------------------------|
| `cataluna84/arc-agi-3-agents-pkg`                     | this repo   | 200 KB  | every time `agents/` changes (manual)                |
| `cataluna84/qwen3-6-35b-a3b-bf16`                     | exp004      | 71.93 GB| frozen (parked)                                      |
| `cataluna84/arc-agi-3-transformers-wheels`            | exp004      | 16 MB   | frozen unless transformers version bumps             |
| `cataluna84/arc-agi-3-frame-change-cnn`               | NEW (D9)    | 5 MB    | every CNN training (D8, D17, D26)                    |
| `cataluna84/arc-agi-3-click-head`                     | NEW (D11)   | 5 MB    | every click-head training (D11, D17)                 |
| `cataluna84/arc-agi-3-recordings`                     | NEW (D6)    | 100 MB  | weekly snapshot of `data/recordings/`                |

### 5.2 Data-augmentation experiment scoreboard

Track each augmentation's contribution in `experiments/AUGMENTATION_SCOREBOARD.md`.

| Aug                          | Multiplier | First introduced | LB delta vs no-aug | Notes |
|------------------------------|-----------:|-----------------:|-------------------:|-------|
| A1. D₄ symmetries            | 8×         | D8               | TBD                | Rotations + reflections. Action labels rotated. |
| A2. Color permutations       | 50×        | D8               | TBD                | Subsample of 16!. Defeats palette overfitting.   |
| A3. HER (hindsight relabel)  | ~5×        | D26              | TBD                | Goal-conditioned policy training trick.          |
| A4. CA perturbations         | 2×         | D8               | TBD                | Rule 30 / Rule 110 / GoL-step on random cells.   |
| A5. Grid traversals          | 5×         | D26              | TBD                | Snake / Hilbert / Z-order — for sequence models. |
| A6. Cross-game synthetic     | n/a        | D8               | TBD                | Recording mix of all 6 public games.             |

### 5.3 Tests to maintain

| Test                                    | Ensures                                                  |
|-----------------------------------------|----------------------------------------------------------|
| `tests/test_state_graph.py`             | hash determinism, level-reset, edge insertion, frontier  |
| `tests/test_go_explore.py`              | replay determinism, archive cell-id stability             |
| `tests/test_objects.py`                 | connected-components correctness, centroid stability      |
| `tests/test_world_model.py`             | sMM slot identity, tMM piecewise-linearity, rMM update    |
| `tests/test_augmentations.py`           | A1 invariance, A2 palette permutation correctness         |
| `scripts/<agent>_smoke_local.py` × N    | per-agent 22-check parity (no GPU)                       |

### 5.4 Pre-commit hygiene

The following are mandatory and enforced by `.pre-commit-config.yaml`:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run pre-commit run --all-files
```

If any fail at commit time, **fix locally before pushing**.

### 5.5 `.factory/` updates required by this spec

- `.factory/plan.md` — mark D2 done, add D3-D28 entries summarizing.
- `.factory/memories.md` — append daily section (TOP) per `AGENTS.md`.
- `.factory/rules/gotchas.md` — append #15 if any new gotcha emerges.
- `CHANGELOG.md` — mirror user-visible changes per day.

### 5.6 Submission-slot accounting

| Day | Submit? | Target Kernel                                  | Expected LB |
|----:|:--------|------------------------------------------------|------------:|
|  D3 | YES     | `cataluna84/ash-s-arc-agi-3-agent` v2          |        0.19 |
|  D4 | YES     | `cataluna84/goose-comp-arc-agi-3`              |        0.25 |
|  D5 | YES     | `cataluna84/trigger-bfs-comp-arc-agi-3`        |        0.32 |
|  D6 | NO      | -                                              |           - |
|  D7 | YES     | `cataluna84/bfs-state-graph-comp-arc-agi-3`    |        0.36 |
|  D8 | NO      | -                                              |           - |
|  D9 | YES     | `cataluna84/goose-cnn-comp-arc-agi-3` v2 (CPU + try/except + level>=0 guard) — Done 2026-05-07; v1=0.00, v2=**0.17** |   0.20-0.30 |
| D10 | NO      | -    (build) `agents/frame_segmenter.py` per arXiv:2512.24156 — Done 2026-05-13 (~440 LOC + 11 tests) | - |
| D11 | NO      | -    (build) trigger_bfs ACTION6 prior wired to segmenter tiers — Done 2026-05-13 (22/22 smoke) | - |
| D15 | YES     | `cataluna84/trigger-bfs-segmenter-comp-arc-agi-3` v1 (exp008; COMPLETE 2026-05-14, LB=**0.12**; next = GraphExplorer Algorithm 1, not coord-only tweak) | 0.30-0.36 |
| D12 | YES     | `cataluna84/forge-v20-comp-arc-agi-3`          |        0.42 |
| D13 | NO      | -                                              |           - |
| D14 | NO      | -                                              |           - |
| D15 | NO      | -                                              |           - |
| D16 | NO      | -                                              |           - |
| D17 | YES     | `cataluna84/object-aware-comp-arc-agi-3`       |        0.43 |
| D18 | NO      | -                                              |           - |
| D19 | NO      | -                                              |           - |
| D20 | YES     | `cataluna84/world-model-comp-arc-agi-3`        |        0.46 |
| D21 | NO      | -                                              |           - |
| D22 | NO      | -                                              |           - |
| D23 | NO      | -                                              |           - |
| D24 | YES     | `cataluna84/axiom-lite-comp-arc-agi-3`         |        0.40 |
| D25 | NO      | -                                              |           - |
| D26 | NO      | -                                              |           - |
| D27 | YES     | `cataluna84/cascade-comp-arc-agi-3`            |        0.50 |
| D28 | NO      | -                                              |           - |

11 of 26 days submit. 15 build-only. Slot-saving days are critical; the
spec deliberately leaves them empty so any agent picking up mid-stream
knows the slot is available for emergency Track A fallbacks.

---

## 6. Hard constraints (DO NOT violate)

- **Apache-2.0 obligation**: all eligible (prize-claimable) submissions
  must open-source by 2026-06-30 (Milestone 1) or 2026-09-30 (Milestone 2).
- **No internet during Kaggle eval** — every model weight, every wheel
  must be a Kaggle Dataset attached via `dataset_sources`.
- **6 h wall clock cap** on Kaggle reruns; target ≤ 5 h.
- **Pre-commit MUST PASS** before any push (gitleaks is the last line of
  defence on tokens).
- **One submission per 24 h** — never queue two on the same day; the
  second is silently rejected.
- **Vendored upstream code (`agents/_forge_v19.py`)** is APPEND-ONLY for
  attribution reasons. Local adaptations live in `agents/forge_agent.py`.

---

## 7. Spec change log

| Date       | Author | Change                                           |
|------------|--------|--------------------------------------------------|
| 2026-05-01 | Droid  | Initial draft, D3-D28 4-week plan                |

---

## 8. Open questions reserved for later

1. **Submission-slot scoring math**: exact formula not yet verified. Per
   SDK §0.9.3 it is per-level squared, per-game weighted by level index,
   averaged over all 25 envs. Re-confirm by reading
   `arc_agi/scorecard.py` source on D6.
2. **Game-level human action baselines**: discussion thread "Human scores
   visible to agents" (Cottaar) suggests these are exposed in the SDK.
   Verify on D6.
3. **Click-coord granularity**: does ACTION6 accept any (x, y) ∈ [0, 63]²
   or are there snap-to-cell rules? Verify on D11 by experimenting in
   dev kernel.
4. **State-graph reset semantics on level transition vs game reset**:
   Section 0.4 of `arc-agi` SDK; experimentally verify on D6.
