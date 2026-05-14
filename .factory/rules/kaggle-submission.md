# Kaggle Submission Skeleton (offline notebook)

> The Kaggle eval is fully offline. Internet is disabled. The submission CSV is auto-built as long as actions hit the env. Always verify via the **Submissions** tab afterward.

## Skeleton

```python
# %% Cell 1 - install local wheels
!pip install --no-index --find-links=/kaggle/input/arc-agi-3-data/arc_agi_3_wheels arc-agi

# %% Cell 2 - register custom agent
import sys, os
sys.path.insert(0, '/kaggle/input/arc-agi-3-data/ARC-AGI-3-Agents')
from agents.agent import Agent
from agents.structs import FrameData, GameAction, GameState
# ...your agent class...

# %% Cell 3 - run via swarm
import subprocess
subprocess.run(['python','/kaggle/input/arc-agi-3-data/ARC-AGI-3-Agents/main.py',
               '--agent=myawesomeagent','--game=all'], check=True)
```

## Pre-flight cell (drop into every notebook)

```python
import os, sys, json
# 1. Datasets attached?
assert os.path.isdir('/kaggle/input'), 'No /kaggle/input mount'
print('Inputs:', os.listdir('/kaggle/input'))

# 2. arc-agi importable?
sys.path.insert(0, '/kaggle/input/arc-agi-3-data/ARC-AGI-3-Agents')
import arc_agi
from agents.agent import Agent
print('arc_agi version:', arc_agi.__version__)

# 3. Env files present?
ef = '/kaggle/input/arc-agi-3-data/environment_files'
assert os.path.isdir(ef), f'Missing env files at {ef}'
print('env files:', len(os.listdir(ef)))

# 4. GPU sanity (only if your agent is neural)
import torch
print('CUDA available:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
```

## Hard rules

1. **No internet calls** during eval. All weights/binaries must be packaged
   as Kaggle Datasets / Models or competition-provided inputs.
2. **Notebook-only submissions**. The Kaggle page now requires submissions
   through notebooks; the ARC-AGI-3 harness auto-generates the submission
   file when the agent interacts with the environment.
3. **Runtime cap is now 9h for CPU and GPU notebooks** (2026-05-14
   update). Leave at least 1h buffer; budget target = <=8h actual compute.
4. **External data / pretrained models must be freely and publicly
   available** for prize-relevant submissions. Private code/model datasets
   are acceptable for dev probes only, not for final prize-eligible kernels.
5. **1 submission/day per submitter** — kernel pushes and dev/probe runs do
   not burn the competition submission slot; `kaggle competitions submit`
   does.
6. **Submissions auto-generate** even on crashed runs (with zero scores).
   Always confirm a non-zero score.
7. **CPU vs GPU vs T4 vs H100 vs RTX 6000**: The Kaggle CLI exposes
   accelerator types via `--accelerator` (P100, T4, T4Highmem, A100, L4,
   L4X1, **H100**, **RtxPro6000**, several TPUs). **H100 access has been
   verified for ARC-AGI-3-attached kernels on this account** — pushing with
   `--accelerator NvidiaH100` allocates a real H100 80 GB HBM3 (sm_90,
   Hopper) running `gcr.io/kaggle-gpu-images/python` (Python 3.12, torch
   2.10.0+cu128, CUDA 12.8 / driver 13.0). **RTX 6000 (`g4-standard-48`) is
   now explicitly announced for ARC-AGI-3-only notebooks with internet
   disabled**; probe before relying on exact VRAM/package stack. See
   `runs/h100_probe/` for the prior H100 probe results and
   `experiments/exp004_qwen_agent/rtx6000_probe_kernel/` for the new RTX
   probe scaffold. Important caveats:
   - The H100 image SHA is the SAME as the `gcr.io/kaggle-private-byod/python` image referenced in some kernels' metadata (just mirrored under different registry namespaces).
   - `machine_shape` in `kernel-metadata.json` (e.g. the upstream FORGE notebook declares `NvidiaTeslaT4`) is metadata-level and stale; the actual allocation is what `--accelerator` requests.
   - Use H100/RTX for compute-heavy paths (bundled LLMs, DreamerV3, etc). Use CPU/T4 for BFS-bound agents (FORGE, Trigger-Aware BFS) where the bottleneck is Python-level state-space search, not GPU FLOPS.
   - Quota for individual users is not publicly documented; the probe ran in ~14s without issue. Test small before committing.
