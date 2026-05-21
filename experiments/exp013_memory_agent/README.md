# exp013_memory_agent

**D23 (2026-05-21) submission candidate**: fork of the public Kaggle notebook
[ARC-AGI-3 memoryAgent (v6)](https://www.kaggle.com/code/yuriao/arc-agi-3-memoryagent)
by yuriao.

## What's in it

v6 of an iteratively-developed memory-based ARC-AGI-3 agent. Per our
`.factory/rules/leaderboard-anchors.md`, the memoryAgent family scores around
**0.28** on the public LB — above our 0.24 D3 FORGE best and meaningfully
above the 0.18-0.20 we got from the Hybrid Solver v10 fork on D21/D22.

## Architecture (per the inlined source)

- **Object extraction**: BFS connected components on the 64x64 frame (pure
  numpy), bounded by `min_size=4, max_size=3000`. Returns dicts with
  `color, cx, cy, size, w, h`.
- **Object descriptors**: canonicalize objects for memory keys.
- **Frame descriptor v2**: hashes a canonical frame representation for
  state dedup.
- **TransitionMemory** (two-buffer ring): one buffer for frame transitions,
  one for click transitions. Capacity raised to 4000/4000 in v6 (from
  1000/1000 in v5) since memory now persists across games.
- **Click heatmap**: ACTION6 prior built from click memory entries that
  caused frame changes.
- **v6 innovation**: `_SHARED_MEMORY: dict[str, TransitionMemory]`
  class-level — keyed by `game_type` (prefix before first `-` in
  `game_id`). Persists across game instances of the same type. New game
  type → fresh isolated memory.

## Why this is structurally different from exp012

| Property | exp012 Hybrid Solver v10 | exp013 memoryAgent v6 |
|---|---|---|
| Core | FORGE v19 BFS + A* + beam | Memory-mediated transition learning |
| Click prior | Sprite permutation + MCTS click masking | Click heatmap from past changes |
| State dedup | Hash + state graph | Frame descriptor v2 |
| Memory scope | Per-level | Cross-game-type (v6 upgrade) |
| GPU usage | CPU-only | T4 GPU |
| Code size | 78 KB | 32 KB |

## Metadata

| Field | Value |
|---|---|
| Source kernel | `yuriao/arc-agi-3-memoryagent` |
| Our slug | `cataluna84/memory-agent-v6-comp-arc-agi-3` |
| enable_gpu | true (T4) |
| enable_internet | false |
| competition_sources | `["arc-prize-2026-arc-agi-3"]` |
| dataset_sources | none |
| Inlined agent | ~32 KB, single `%%writefile` cell |

## Expected LB

Per `.factory/rules/leaderboard-anchors.md`: memoryAgent family ~0.28.
Conservative band for our fork: **0.20-0.32**, with same Kaggle re-roll
variance pattern as exp012.

## Inspection (D23 pre-push)

- Imports: `hashlib, logging, os, random, time, traceback, collections.deque,
  typing, numpy, torch (+ nn, nn.functional, optim), agents.agent.Agent,
  arcengine.{FrameData, GameAction, GameState}`. All stdlib / Kaggle-image-shipped.
- **Zero hardcoded paths** to `/kaggle/input`. Zero `http://`, `https://`,
  `api_key`, `urlopen`, or `requests.` calls.
- `MAX_ACTIONS = float('inf')`; relies on game-side GAME_OVER.
- Same harness wiring (`gateway:8001` + `cp ARC-AGI-3-Agents` + `python main.py
  --agent myagent`) as our exp008 / exp012 kernels.
- Same dummy-submission fallback pattern for save-mode.

## Attribution + license

Public Kaggle code competition notebook (`is_private: false`). Per the source
notebook's `kernel-metadata.json`, no license field is set — Kaggle's default
for public code competition notebooks is permissive. ARC Prize 2026 rules
require 3rd-party code under permissive open-source licenses; this qualifies.

We attribute the work in this README and the source URL. If this lifts our LB
to prize-relevant tiers, an open-source disclosure under the same permissive
terms will accompany our prize-eligible submission.

## Process

1. `kaggle kernels pull yuriao/arc-agi-3-memoryagent`.
2. Created `experiments/exp013_memory_agent/comp_kernel/` with our slug
   `cataluna84/memory-agent-v6-comp-arc-agi-3`.
3. `kaggle kernels push` → save-mode verification.
4. Confirm with user before competition submit (per saved feedback rule).
