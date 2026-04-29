# exp004 - Qwen3.6-35B-A3B vision-language agent

**Goal**: a vision-language agent driven by `Qwen/Qwen3.6-35B-A3B` (BF16,
~70 GB) that ingests the ARC-AGI-3 frame as an image AND a hex-grid text
dump in the same prompt, and emits one action per turn.

## Why this config (locked decisions)

- **Precision: bf16** (chosen by user 2026-04-29). 35.95B BF16 params = ~72 GB.
  Fits in Kaggle's 80 GB H100 with mild headroom; zero quantization-induced
  quality loss.
- **Framework: `transformers 5.0.0`** (pre-installed on Kaggle's H100 image -
  see `runs/h100_probe/`). No vLLM / SGLang / flash-attn install needed for
  v1; can be added later if latency budget pushes back.
- **Frame encoding: image + text grid + available_actions** (chosen by user).
  Image gives spatial cues, text grid gives exact cell values, action list
  forces the model to pick within the env's filtered set.
- **Bundling: private Kaggle Dataset** (no `kaggle.com/refs/hf-model/`,
  that bridge URL returns 404). The bundler kernel does
  `huggingface_hub.snapshot_download` then `kagglehub.dataset_upload` from
  inside Kaggle's network (gigabit) so we never upload 72 GB from home.

## Open trade-offs deferred to later experiments

- `exp005`: same code, swap weights to **Qwen3.5-VL-30B-A3B-Thinking** and
  **Qwen3-Next-80B-A3B-INT4**. Compares quality at the 80 GB H100 limit.
- `exp006`: switch to vLLM if transformers latency is too high.
- `ACTION6` (click) coordinate parsing: v1 accepts "ACTION6 (x, y)"; if the
  model ignores the format we'll add a coord-only second-pass call.

## File map

| Path | Role |
| --- | --- |
| `bundle_qwen_kernel/` | Kaggle dev kernel that downloads HF weights and pushes them as a private Dataset. Run **once**. |
| `dev_kernel/` | Kaggle dev kernel that mounts the Dataset and runs `QwenAgent` on a single ARC game. Smoke test, not a competition submission. |
| `notes.md` | Per-day notes during the experiment (created lazily). |

The agent class itself lives at `agents/qwen_agent.py`. The pure-Python
prompt-build + parser smoke test lives at `scripts/qwen_agent_smoke_local.py`
and runs without any GPU.

## End-to-end runbook

### Step 0 - local prompt/parser smoke (no GPU, no model)

```bash
.venv/bin/python scripts/qwen_agent_smoke_local.py
```

Verifies that `build_prompt()` and `parse_action()` produce sensible output
on a synthetic 64x64 grid. Does NOT load Qwen weights.

### Step 1 - bundle the model into a Kaggle Dataset (one-time, ~45-60 min)

```bash
cd /home/cataluna84/Workspace/arc-agi-3
.venv/bin/kaggle kernels push -p experiments/exp004_qwen_agent/bundle_qwen_kernel
.venv/bin/kaggle kernels status cataluna84/qwen-bundle-arc-agi-3
# wait until COMPLETE (~30-50 min)
```

The bundler kernel writes a private Kaggle Dataset at
`cataluna84/qwen3-6-35b-a3b-bf16` (slug auto-derived). Verify with:

```bash
.venv/bin/kaggle datasets files cataluna84/qwen3-6-35b-a3b-bf16 | head -5
```

### Step 2 - dev kernel smoke run (no submission, no daily slot)

The dev kernel's `kernel-metadata.json` references the bundled Dataset via
`dataset_sources`. Inside the kernel, weights appear at
`/kaggle/input/qwen3-6-35b-a3b-bf16/` and `QWEN_MODEL_PATH` defaults to that.

```bash
.venv/bin/kaggle kernels push -p experiments/exp004_qwen_agent/dev_kernel
.venv/bin/kaggle kernels status cataluna84/qwen-dev-arc-agi-3
.venv/bin/kaggle kernels output cataluna84/qwen-dev-arc-agi-3 -p runs/qwen_dev/
```

Pass criteria for v1 (smoke):
- No OOM (model loads in BF16 on the 80 GB H100).
- Per-action latency < 10 s wall-clock.
- Picks valid actions (in `available_actions`) on at least 90% of turns.
- Reaches at least 1 level on `ls20` within 200 actions.

### Step 3 - decide whether to promote to a competition kernel

If Step 2 looks healthy, we either:
- (a) Edit the dev kernel's metadata to remove `enable_internet` and queue
  for a competition submission (uses the daily slot).
- (b) Iterate on the prompt template / max_new_tokens and re-run Step 2.

## Kaggle hardware constraints (verified 2026-04-29 via dev probe)

- 1 x H100 80 GB HBM3, compute capability 9.0, FP8 native.
- 31.4 GB system RAM (NOT enough for CPU offload of 70 GB weights -
  the model MUST fit fully on GPU).
- `/kaggle/working` only **19.5 GB** free (cannot cache model here).
- `/tmp` 1.2 TB free (used by HF cache during dev kernel runs).
- Kaggle Dataset cap: 500 GB public, 100 GB private. 72 GB fits private.
