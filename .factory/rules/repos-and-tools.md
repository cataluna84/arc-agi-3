# Critical Repos and Tools

> Must clone/study before building any new agent. The first three are official; the rest are top community references.

| Repo | What it is | Why we care |
| --- | --- | --- |
| `arcprize/ARC-AGI-3-Agents` | Official agent harness (Python, MIT, 213★) | Defines `Agent.is_done` / `Agent.choose_action`, `FrameData`, `GameAction` |
| `arcprize/ARC-AGI` | Official toolkit (`arc_agi`, `arcengine`) | `Arcade().make("ls20", render_mode="terminal")` etc. |
| `arcprize/arc-agi-3-benchmarking` | `arcagi3` harness (Plank/Anusheel) | Multi-provider LLM benchmarking driver |
| `DriesSmit/ARC3-solution` | StochasticGoose v1 (1st preview) | CNN action-change predictor blueprint |
| `dolphin-in-a-coma/arc-agi-3-just-explore` | Graph-based explorer (3rd preview) | Frame seg + priority tier + state graph |
| `alexisfox7/RGB-Agent` | LLM-orchestrated agent (Claude/GPT) | Plan-batched action queue, swarm runner |

## Where the Kaggle data lives

- 25 public game files: **`environment_files/`** in the Kaggle competition Data tab.
- `arc-agi` Python wheels: **`arc_agi_3_wheels/`** in the same Data tab.
- Both must be added as Kaggle Datasets to the offline submission notebook.

## SDK API drift

- 2026-03 release of **ARC-AGI-3-Agents v0.9.3** renames `score → levels_completed` and `win_score → win_levels`. Watch for older notebooks still using the old fields.
