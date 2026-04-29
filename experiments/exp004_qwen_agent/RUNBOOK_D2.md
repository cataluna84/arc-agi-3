# RUNBOOK_D2 - exp004 Qwen3.6-35B-A3B - tomorrow morning, copy-paste edition

> **Read order**: this file first. If you need deeper context: `.factory/memories.md` (top section) and `experiments/exp004_qwen_agent/README.md`.
>
> **Total time budget tomorrow**: ~1.5-2 h of clock time, but only ~30 min of active attention (the bundler runs ~45-60 min unattended).

---

## 0. Pre-flight (5 min, the moment you sit down)

```bash
cd /home/cataluna84/Workspace/arc-agi-3
set -a && source .env && set +a
export KAGGLE_API_TOKEN="$KAGGLE_KEY"          # KGAT_ tokens use Bearer auth

# Sanity: verify auth works
.venv/bin/kaggle kernels status cataluna84/h100-probe-arc-agi-3
# expect: ... has status "KernelWorkerStatus.COMPLETE"
```

**Check today's (D1) submission outcome before anything else:**

```bash
.venv/bin/kaggle competitions submissions arc-prize-2026-arc-agi-3 | head -20
```

Find the row from yesterday (2026-04-29). Note the score. If it's already in the leaderboard, log it to `.factory/memories.md` (new top section) before starting D2 work.

---

## 1. Decide today's Kaggle slot (5 min)

**The single daily slot is precious.** Use this decision tree:

| Yesterday's `s2` outcome | Today's action |
|---|---|
| `s2 >= 0.30` (much higher than 0.19) | **Track A**: Ash resubmit #2 (variance probe). High-variance hypothesis is alive; one more sample resolves it. |
| `0.10 <= s2 <= 0.30` | **Track C**: hold the slot. We have enough info from `(0.19, s2)` - mean ~0.19, std small. Spend the day on exp004 dev kernels. |
| `s2 < 0.10` | **Track A**: resubmit (something went wrong; rule out infrastructure failure). |
| Submission still pending / errored / didn't grade | Track C: hold slot, advance exp004. Burn the slot only if there's a clear submission to make. |

If you choose Track A, run `bash scripts/resubmit_ash.sh` and skip to section 6 (it'll take its own day; no Qwen progress).

For everything else, continue with Track B/C below (Qwen advancement, no slot consumed).

---

## 2. Bundle the local `agents/` package as a Kaggle Dataset (one-time, ~2 min)

This is the missing dependency the dev kernel needs - it imports `from agents.qwen_agent import QwenAgent` from `/kaggle/input/arc-agi-3-agents-pkg/agents/`.

```bash
cd /home/cataluna84/Workspace/arc-agi-3

# stage just the agents/ dir (no .venv, no data, no runs)
mkdir -p /tmp/agents_pkg_stage
rm -rf /tmp/agents_pkg_stage/*
cp -r agents /tmp/agents_pkg_stage/

# write dataset metadata
cat > /tmp/agents_pkg_stage/dataset-metadata.json <<'JSON'
{
  "title": "ARC-AGI-3 agents package (cataluna84)",
  "id": "cataluna84/arc-agi-3-agents-pkg",
  "licenses": [{"name": "apache-2.0"}],
  "subtitle": "Local agents/ directory for ARC-AGI-3 dev kernels",
  "description": "Mirrors arc-agi-3/agents/ for offline import in Kaggle kernels.",
  "isPrivate": true,
  "keywords": ["arc-agi", "agents"]
}
JSON

# create the Dataset (first version; uses --dir-mode tar so the agents/ tree is preserved)
.venv/bin/kaggle datasets create -p /tmp/agents_pkg_stage --dir-mode tar
# expect: "Your private Dataset is being created. ... cataluna84/arc-agi-3-agents-pkg"

# verify
.venv/bin/kaggle datasets files cataluna84/arc-agi-3-agents-pkg | head -20
# expect to see: agents/qwen_agent.py, agents/__init__.py, agents/agent.py, etc.
```

If you ever change `agents/` later and want to refresh this Dataset, run:
```bash
.venv/bin/kaggle datasets version -p /tmp/agents_pkg_stage --dir-mode tar -m "agents pkg sync"
```

---

## 3. Push the bundler kernel and let it run (~45-60 min unattended)

```bash
cd /home/cataluna84/Workspace/arc-agi-3
.venv/bin/kaggle kernels push -p experiments/exp004_qwen_agent/bundle_qwen_kernel
# expect: "Kernel version 1 successfully pushed."
```

Now poll until `COMPLETE`. Run this loop and walk away for a coffee:

```bash
KERNEL=cataluna84/qwen-bundle-arc-agi-3
while true; do
  S=$(.venv/bin/kaggle kernels status "$KERNEL" 2>&1 | head -1)
  echo "[$(date +%H:%M:%S)] $S"
  echo "$S" | grep -q "COMPLETE\|ERROR\|FAIL\|CANCELLED" && break
  sleep 60
done
```

When it says COMPLETE, fetch the log and verify the upload succeeded:

```bash
mkdir -p runs/qwen_bundle && rm -rf runs/qwen_bundle/*
.venv/bin/kaggle kernels output "$KERNEL" -p runs/qwen_bundle/

.venv/bin/python <<'PY'
import json
log = open('runs/qwen_bundle/qwen-bundle-arc-agi-3.log').read()
events = json.loads(log)
out = ''.join(e['data'] for e in events if e.get('stream_name') == 'stdout')
# print only the SUMMARY block + any error lines
print(out[-4000:])
PY
```

**Pass criteria for the bundler:**
- Final SUMMARY block prints `uploaded ok      : True`
- `total bytes` is between **65 and 75 GB** (sanity check - 26 shards * ~2.7 GB)
- Download time < 50 min, upload time < 50 min

**Verify the Dataset exists:**
```bash
.venv/bin/kaggle datasets files cataluna84/qwen3-6-35b-a3b-bf16 | head -10
# expect: model-00001-of-00026.safetensors, ..., tokenizer.json, config.json
```

If the bundler failed, see Troubleshooting (section 7) before re-running.

---

## 4. Push the dev kernel and review the smoke output (~10-20 min)

This is the expensive step in terms of clock time per iteration: ~3-5 min for the H100 to load 70 GB of bf16 shards from `/kaggle/input/`, plus ~50 actions * ~3-10 s each = 5-15 min of inference.

```bash
cd /home/cataluna84/Workspace/arc-agi-3
.venv/bin/kaggle kernels push -p experiments/exp004_qwen_agent/dev_kernel
# expect: "Kernel version 1 successfully pushed."

# poll
DEV_KERNEL=cataluna84/qwen-dev-arc-agi-3
while true; do
  S=$(.venv/bin/kaggle kernels status "$DEV_KERNEL" 2>&1 | head -1)
  echo "[$(date +%H:%M:%S)] $S"
  echo "$S" | grep -q "COMPLETE\|ERROR\|FAIL\|CANCELLED" && break
  sleep 60
done

# fetch outputs
mkdir -p runs/qwen_dev && rm -rf runs/qwen_dev/*
.venv/bin/kaggle kernels output "$DEV_KERNEL" -p runs/qwen_dev/

# print the smoke summary
.venv/bin/python <<'PY'
import json
log = open('runs/qwen_dev/qwen-dev-arc-agi-3.log').read()
events = json.loads(log)
print(''.join(e['data'] for e in events if e.get('stream_name') == 'stdout')[-4000:])
print('--- smoke json ---')
print(open('runs/qwen_dev/qwen_smoke.json').read())
PY
```

---

## 5. Pass criteria for the dev kernel smoke run

Read the smoke JSON and the kernel log. Mark each:

| Check | Pass threshold | If fail, do this |
|---|---|---|
| Model loaded without OOM | log contains "loaded ... in N.Ns" with N < 600 | switch to fp16 (`QWEN_DTYPE=fp16` in the notebook) and re-push |
| No transformers AutoProcessor errors | no `KeyError` / `OSError` in stderr | check that `tokenizer.json` and `preprocessor_config.json` are in the Dataset; if missing, re-bundle with broader allow_patterns |
| Per-action latency < 10 s | mean `dt` in the per-step lines | reduce `QWEN_MAX_NEW_TOKENS` from 96 to 48; consider FP8 via torchao in exp005 |
| All emitted actions in `available_actions` | inspect `action_histogram` | parser failure - investigate `qwen_trace.log` (saved at `/tmp/qwen_trace.log`, exfil via writing to `/kaggle/working`) |
| `levels_completed >= 1` on `ls20` | `qwen_smoke.json[\"levels_completed\"] >= 1` | this is the real ML signal. If 0, prompt rework is needed - go to section 6 prompt-iteration loop |

Document the result by appending to `.factory/memories.md` (NEW top section):

```
## 2026-04-30 - exp004 dev kernel D2

- Model load:    XXXs (target < 600s, got: ___)
- Per-action:    XX.Xs (target < 10s, got: ___)
- Valid actions: XX/50 (target == 50, got: ___)
- Levels done:   X / Y win_levels on ls20 (target >= 1, got: ___)
- Decision:      [iterate prompt | promote to comp kernel | pivot]
```

---

## 6. Iteration loop (no slot consumed, do this until smoke is healthy)

If smoke fails one criterion (latency, validity, or level completion), iterate **on dev kernels only**:

1. Edit `agents/qwen_agent.py` locally.
2. Re-bundle the agents Dataset (section 2 - just the version step):
   ```bash
   rm -rf /tmp/agents_pkg_stage/agents
   cp -r agents /tmp/agents_pkg_stage/
   .venv/bin/kaggle datasets version -p /tmp/agents_pkg_stage --dir-mode tar -m "qwen prompt iter"
   ```
3. Re-push the dev kernel: `.venv/bin/kaggle kernels push -p experiments/exp004_qwen_agent/dev_kernel`.
4. Re-poll, re-review.

Each iteration is ~10-15 min wall clock. **No daily slot consumed.** Iterate until the smoke is in the green.

---

## 7. Troubleshooting

### Bundler kernel fails at Step 1 (HF download)
Likely cause: HF hub temporary outage OR allow_patterns excluded a critical file.
Fix: edit `bundle_qwen.ipynb`, broaden `allow_patterns` to `["*.json", "*.txt", "*.md", "*.py", "*.safetensors", "tokenizer*", "preprocessor*"]`, re-push.

### Bundler kernel fails at Step 3 (Kaggle upload)
- If `kagglehub.dataset_upload` errors with auth: the kernel doesn't have ambient Kaggle creds. The fallback to CLI runs automatically; check its `stderr`.
- If both fail with "dataset already exists": rename the slug in `bundle_qwen.ipynb` (e.g. `qwen3-6-35b-a3b-bf16-v2`) and re-push, OR use `kaggle datasets version` from a manual kernel.

### Dev kernel: `Qwen dataset not mounted at /kaggle/input/qwen3-6-35b-a3b-bf16`
Fix: open the dev kernel on Kaggle web UI, confirm "Add Data" lists the bundled Qwen Dataset. If missing, edit `kernel-metadata.json -> dataset_sources` to use the actual slug from `kaggle datasets list -m`. The slug is case-sensitive.

### Dev kernel: OOM during model load
70 GB bf16 + ~3-8 GB activations + KV cache overhead can push past 80 GB.
Fix: in `qwen_agent_dev.ipynb`, set `os.environ['QWEN_DTYPE'] = 'fp16'` (saves ~no memory but uses different code path) OR more reliably, install bitsandbytes and switch to NF4 quant:
```python
# add to dev kernel before importing the agent:
import subprocess, sys
# enable_internet=true would be needed for this; better: pre-bundle bnb wheel
subprocess.run([sys.executable, '-m', 'pip', 'install', 'bitsandbytes==0.45.0'])
```
**Cleaner option for D3+**: build exp005 with `Qwen3-Next-80B-A3B-INT4` from Kaggle Models registry (no bnb needed, native INT4 weights).

### Dev kernel: `transformers.AutoModelForImageTextToText` missing
Means transformers < 5.0 in the kernel container.
Fix: print `transformers.__version__` early in the notebook; if < 5.0, fall back to `AutoModelForCausalLM` (text-only mode, drop the image part of the prompt for now).

### Per-action latency > 30 s
The model is generating too many reasoning tokens or KV cache is paging.
Fix: drop `QWEN_MAX_NEW_TOKENS` to 32; cap context length explicitly via processor `max_length`.

---

## 8. Promotion to a competition kernel (only if dev smoke is healthy)

When dev smoke meets all pass criteria on at least 3 sampled games (`ls20`, plus two others - edit `SMOKE_GAME` env var and re-run dev kernel), promote:

```bash
# copy the dev kernel to a new comp-kernel folder
cp -r experiments/exp004_qwen_agent/dev_kernel experiments/exp004_qwen_agent/comp_kernel

# edit experiments/exp004_qwen_agent/comp_kernel/kernel-metadata.json:
#   - id: "cataluna84/qwen-comp-arc-agi-3"
#   - title: "Qwen Comp ARC-AGI-3"
#   - enable_internet: "false"  (already false)
#   - keep dataset_sources, competition_sources

# edit experiments/exp004_qwen_agent/comp_kernel/qwen_agent_dev.ipynb:
#   - replace the single-game smoke loop with the standard competition loop
#     (iterate over arc_env.list_games(), produce submission.parquet)
#   - remove os.environ['QWEN_DEBUG_PROMPTS'] = '1'  (slow + leaks log file)

# push and submit
.venv/bin/kaggle kernels push -p experiments/exp004_qwen_agent/comp_kernel
# wait for COMPLETE
.venv/bin/kaggle competitions submit-code \
    -c arc-prize-2026-arc-agi-3 \
    --kernel cataluna84/qwen-comp-arc-agi-3 \
    --kernel-version 1 \
    -f submission.parquet \
    -m "exp004 D3 - Qwen3.6-35B-A3B BF16 first submission"
# THIS BURNS THE DAILY SLOT.
```

Do NOT submit unless every dev pass criterion is green. The slot is more valuable than the urge to ship.

---

## 9. End-of-day ritual

Whatever happened, append a dated section to the **top** of `.factory/memories.md`:
- Today's daily-slot decision (which track) and why.
- LB number if a submission landed.
- Smoke results (latency, validity, levels).
- Decision for D3: continue iterating exp004 prompt? promote? pivot to exp005 (INT4 80B)?

If you discovered a new Kaggle / Qwen / SDK gotcha, also add a one-liner to `.factory/rules/gotchas.md`.

---

## Quick reference - file map

| File | Edit when... |
|---|---|
| `agents/qwen_agent.py` | iterating prompt template, parser, history, generation params |
| `experiments/exp004_qwen_agent/bundle_qwen_kernel/bundle_qwen.ipynb` | changing what gets downloaded from HF |
| `experiments/exp004_qwen_agent/dev_kernel/qwen_agent_dev.ipynb` | changing the smoke loop, env vars passed to QwenAgent |
| `experiments/exp004_qwen_agent/dev_kernel/kernel-metadata.json` | adding/removing attached Datasets |
| `scripts/qwen_agent_smoke_local.py` | adding more parser / prompt edge cases |

## Quick reference - kernel slugs to remember

| Slug | Purpose |
|---|---|
| `cataluna84/h100-probe-arc-agi-3` | reusable H100 probe (already exists) |
| `cataluna84/qwen-bridge-probe-arc-agi-3` | Qwen env probe (already exists) |
| `cataluna84/qwen-bundle-arc-agi-3` | bundler kernel (run once tomorrow) |
| `cataluna84/qwen-dev-arc-agi-3` | smoke dev kernel (run + iterate) |
| `cataluna84/qwen-comp-arc-agi-3` | competition kernel (only when ready) |
| `cataluna84/arc-agi-3-agents-pkg` | Dataset bundling our agents/ dir |
| `cataluna84/qwen3-6-35b-a3b-bf16` | Dataset with Qwen weights (created by bundler) |
| `cataluna84/ash-s-arc-agi-3-agent` | our Ash fork (Track A resubmit target) |
