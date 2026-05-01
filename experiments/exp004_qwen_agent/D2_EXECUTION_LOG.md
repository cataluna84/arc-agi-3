# D2 Execution Log - exp004 Qwen3.6-35B-A3B BF16 on H100

> **Companion doc to `RUNBOOK_D2.md`.** The runbook explains the *why* and the
> *decision logic*; this file is the *literal terminal log* of the actual D2
> execution: every command we ran, the output we saw, what we decided, and
> why.
>
> If you are a contributor onboarding to this repo and you want to understand
> "how do I push a Kaggle kernel and submit it from the CLI for ARC-AGI-3?",
> read this file top-to-bottom. It is meant to be reproducible: copy each
> command, run it, and you should see the same output (modulo timestamps and
> Kaggle slug versions).
>
> **Date executed**: 2026-04-30 (D2)
> **Operator**: Droid (Claude Opus 4.7) running in Factory CLI on the user's
> WSL2 box (`linux 6.6.87.2-microsoft-standard-WSL2`, Python 3.12, uv-managed
> venv at `.venv/`).

---

## 0. Why we are doing this via CLI at all

Up until D1, every Kaggle submission was done by hand: open the notebook on
the Kaggle web UI, edit cells, click "Save & Run All", click "Submit to
Competition". That works for one-off submissions but it is not reproducible
and it is impossible to version-control. The CLI flow:

1. Stages everything we need (weights, agents/ package, code) as Kaggle
   Datasets (`kaggle datasets create / version`).
2. Pushes a notebook + its `kernel-metadata.json` (`kaggle kernels push`).
   Kaggle reruns it on the requested accelerator. **No daily slot consumed.**
3. Polls the kernel via `kaggle kernels status` until COMPLETE / ERROR.
4. Fetches outputs via `kaggle kernels output` for review.
5. Iterates 2-4 until a kernel version is healthy.
6. **Only then** calls `kaggle competitions submit-code --kernel <slug>
   --kernel-version N`. This single call is what consumes the daily slot.

Documenting commands like this also means we have a literal regression
target: if any step's output drifts in a future D2-equivalent execution, the
diff is obvious.

---

## 1. Pre-checks

The runbook (`RUNBOOK_D2.md` § 0) requires five pre-checks before anything
else: P1 auth, P2 yesterday's submission, P3 today's slot, P4 local smoke,
P5 kernel-metadata review.

### P1 - Kaggle CLI auth (KGAT_ Bearer)

The user's Kaggle API token is a "KGAT_" key (introduced 2025) which uses
**Bearer auth** instead of the older Basic auth. The Kaggle CLI auto-detects
this only when the env var `KAGGLE_API_TOKEN` is set; if you only set
`KAGGLE_USERNAME` + `KAGGLE_KEY`, the CLI still tries Basic and gets 401.

```bash
cd /home/cataluna84/Workspace/arc-agi-3
set -a && source .env && set +a
export KAGGLE_API_TOKEN="$KAGGLE_KEY"
.venv/bin/kaggle kernels status cataluna84/h100-probe-arc-agi-3
```

Expected:
```
... has status "KernelWorkerStatus.COMPLETE"
```

(See `.factory/rules/gotchas.md` for the full Bearer-vs-Basic gotcha.)

### P2 + P3 - Submissions table

```bash
.venv/bin/kaggle competitions submissions arc-prize-2026-arc-agi-3 | head -20
```

We use this for two things at once:
- P2: read yesterday's score and decide Track A / B / C from
  `RUNBOOK_D2.md` § 1.
- P3: confirm we have not already used today's slot. Kaggle enforces 1
  successful submission per 24h.

### P4 - Local smoke

If local smoke is broken, pushing a kernel is wasted - the same code will
break on Kaggle. Re-run all three layers:

```bash
uv run ruff check .
uv run python scripts/qwen_agent_smoke_local.py
uv run python experiments/local_runner.py \
    --agent agents.random_agent:RandomAgent \
    --games ls20-mock --max-actions 30
```

### P5 - Kernel metadata review

```bash
cat experiments/exp004_qwen_agent/bundle_qwen_kernel/kernel-metadata.json
cat experiments/exp004_qwen_agent/dev_kernel/kernel-metadata.json
```

Required fields to verify by eye:
- `id` matches the slug we expect to push to (`cataluna84/qwen-bundle-...`,
  `cataluna84/qwen-dev-...`).
- `enable_internet` is `true` for the bundler (HF download) and `false` for
  the dev kernel (mirrors the offline competition runtime).
- `accelerator` is `gpu` for the dev kernel - this is what gives us the
  H100 attach. (Kaggle's `kernel-metadata.json` accelerator vocabulary is
  documented in
  `https://github.com/Kaggle/kaggle-api/blob/main/src/kaggle/api/kaggle_api.py`.)
- `dataset_sources` includes `cataluna84/arc-agi-3-agents-pkg` (the
  agents/ Dataset we create in Step 1) and, for the dev kernel,
  `cataluna84/qwen3-6-35b-a3b-bf16` (the weights Dataset the bundler
  creates in Step 2).
- `competition_sources` includes `arc-prize-2026-arc-agi-3` so the eval
  data files are mounted at `/kaggle/input/arc-prize-2026-arc-agi-3/`.

---

## 2. Execution log (literal commands + outputs)

> This section is appended to as the run progresses. Each subsection is
> dated and time-stamped UTC.

### 2.1 - 2026-04-30 05:22 UTC - P1 auth check

```bash
cd /home/cataluna84/Workspace/arc-agi-3
set -a && source .env && set +a
export KAGGLE_API_TOKEN="$KAGGLE_KEY"
.venv/bin/kaggle kernels status cataluna84/h100-probe-arc-agi-3
```

Output:
```
cataluna84/h100-probe-arc-agi-3 has status "KernelWorkerStatus.COMPLETE"
```

**Result: PASS.** Bearer auth works; the existing `h100-probe` kernel is
visible and reports COMPLETE.

### 2.2 - 2026-04-30 05:22 UTC - P2 / P3 submissions table

```bash
.venv/bin/kaggle competitions submissions arc-prize-2026-arc-agi-3 | head -20
```

Output:
```
fileName            date                        description                                  status                     publicScore  privateScore
------------------  --------------------------  -------------------------------------------  -------------------------  -----------  ------------
submission.parquet  2026-04-29 02:26:19.467000  Notebook Ash's ARC-AGI-3 Agent  | Version 1  SubmissionStatus.COMPLETE  0.19
```

**Result: PASS.** Yesterday's baseline scored **0.19** as recorded.
**No submission today** -> daily slot is open.

Decision tree (`RUNBOOK_D2.md` § 1): the only confirmed score is 0.19 from
the baseline; we have not yet collected an `s2` resubmit. The decision tree
says `0.10 <= s2 <= 0.30` -> **Track C**: hold the slot, advance exp004.
This is what we proceed with.

### 2.3 - 2026-04-30 05:22 UTC - P4 local smoke

```bash
uv run ruff check .                                   # lint
uv run ruff format --check .                          # format check
uv run python scripts/qwen_agent_smoke_local.py       # 22-check QwenAgent smoke
uv run python experiments/local_runner.py \
    --agent agents.random_agent:RandomAgent \
    --games ls20-mock --max-actions 30                # offline mock-env smoke
```

Output (tails):
```
All checks passed!
11 files already formatted
[PASS] e2e: every emitted action was in available_actions - valid=30/30
[PASS] e2e: history buffer populated within cap - history_len=8/8
ALL OK
[local_runner] totals: {"games": 1, "wins": 0, "actions_taken": 30, "wall_clock_s": 0.0011}
```

**Result: PASS.** All four local-smoke layers green.

### 2.4 - 2026-04-30 05:22 UTC - P5 kernel-metadata review

```bash
cat experiments/exp004_qwen_agent/bundle_qwen_kernel/kernel-metadata.json
cat experiments/exp004_qwen_agent/dev_kernel/kernel-metadata.json
.venv/bin/kaggle --version
.venv/bin/kaggle kernels push --help | grep -i accelerator
```

`bundle_qwen_kernel/kernel-metadata.json`:
```json
{
  "id": "cataluna84/qwen-bundle-arc-agi-3",
  "title": "Qwen Bundle ARC-AGI-3",
  "code_file": "bundle_qwen.ipynb",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "false",
  "enable_internet": "true",
  "dataset_sources": [],
  "competition_sources": []
}
```

`dev_kernel/kernel-metadata.json`:
```json
{
  "id": "cataluna84/qwen-dev-arc-agi-3",
  "title": "Qwen Dev ARC-AGI-3",
  "code_file": "qwen_agent_dev.ipynb",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "false",
  "dataset_sources": [
    "cataluna84/qwen3-6-35b-a3b-bf16",
    "cataluna84/arc-agi-3-agents-pkg"
  ],
  "competition_sources": [
    "arc-prize-2026-arc-agi-3"
  ]
}
```

Kaggle CLI version: `Kaggle CLI 2.1.0`. Help output confirms
`--accelerator ACC` is a valid flag for `kaggle kernels push` -> we pass
`--accelerator NvidiaH100` when pushing the dev kernel.

**Result: PASS.** Both metadata files have the required fields. The two
Datasets the dev kernel mounts (`qwen3-6-35b-a3b-bf16`,
`arc-agi-3-agents-pkg`) do not yet exist -> Steps 1 and 2 will create
them.

---

## 3. Pre-checks summary

| # | Check | Result |
|---|-------|--------|
| P1 | Kaggle CLI auth (KGAT_ Bearer) works | PASS |
| P2 | Yesterday's submission landed at 0.19 | PASS |
| P3 | Today's daily slot still available | PASS |
| P4 | Local smoke (ruff + qwen + local_runner) green | PASS |
| P5 | Both kernel-metadata.json files reviewed | PASS |

All five pre-checks are green. Proceeding with Step 1.

---

## 4. Step 1 - Bundle agents/ as a Kaggle Dataset

### 4.1 Why this Step exists

The dev kernel runs with `enable_internet=false`. It cannot `pip install`
our local code from the working directory; it has to `import` from
`/kaggle/input/<slug>/`. So we mirror the local `agents/` directory as a
private Kaggle Dataset and attach it via `dataset_sources` in
`kernel-metadata.json`.

This Dataset is **versioned**: when `agents/qwen_agent.py` changes during
prompt iteration, we run `kaggle datasets version` (not `create`) to bump
it.

### 4.2 - 2026-04-30 05:24 UTC - Stage agents/ to /tmp

```bash
rm -rf /tmp/agents_pkg_stage
mkdir -p /tmp/agents_pkg_stage
cp -r agents /tmp/agents_pkg_stage/

# Important: strip __pycache__ - those bytecode files can leak old class
# names from before the FORGE rename, and they bloat the upload.
find /tmp/agents_pkg_stage -name __pycache__ -type d -exec rm -rf {} +
find /tmp/agents_pkg_stage -name "*.pyc" -delete
```

After cleanup, `/tmp/agents_pkg_stage/` had 164 KB across 7 .py files
(`__init__.py`, `_forge_v19.py`, `agent.py`, `forge_agent.py`,
`greedy_explore_agent.py`, `qwen_agent.py`, `random_agent.py`).

### 4.3 - 2026-04-30 05:24 UTC - Write dataset-metadata.json

```bash
cat > /tmp/agents_pkg_stage/dataset-metadata.json <<'JSON'
{
  "title": "ARC-AGI-3 agents package (cataluna84)",
  "id": "cataluna84/arc-agi-3-agents-pkg",
  "licenses": [{"name": "apache-2.0"}],
  "subtitle": "Local agents/ directory for ARC-AGI-3 dev kernels",
  "description": "Mirrors arc-agi-3/agents/ for offline import in Kaggle kernels.",
  "isPrivate": true,
  "keywords": ["arc-agi", "agents", "arc-prize-2026"]
}
JSON
```

The `id` must be `<owner>/<slug>` exactly. The slug becomes
`cataluna84/arc-agi-3-agents-pkg` and is referenced verbatim in the dev
kernel's `dataset_sources`.

### 4.4 - 2026-04-30 05:26 UTC - Create the Dataset

```bash
.venv/bin/kaggle datasets create -p /tmp/agents_pkg_stage --dir-mode tar
```

Output:
```
Starting upload for file agents.tar
100%|##########| 150k/150k [00:01<00:00, 93.3kB/s]
Upload successful: agents.tar (150KB)
The following are not valid tags and could not be added to the dataset: ['arc-agi', 'agents', 'arc-prize-2026']
Your private Dataset is being created. Please check progress at https://www.kaggle.com/datasets/cataluna84/arc-agi-3-agents-pkg
```

`--dir-mode tar` packs the staged dir as a single tarball - this preserves
the `agents/` directory tree on the Kaggle side so `from agents.qwen_agent
import QwenAgent` resolves correctly. (The default mode flattens the
directory.)

The keyword warning is cosmetic; the Dataset still exists. Tags can be
added later via the Kaggle web UI.

### 4.5 - 2026-04-30 05:26 UTC - Verify

```bash
sleep 10
.venv/bin/kaggle datasets files cataluna84/arc-agi-3-agents-pkg
.venv/bin/kaggle datasets status cataluna84/arc-agi-3-agents-pkg
```

Output:
```
name                      size  creationDate
-----------------------  -----  --------------------------
__init__.py               3340  2026-04-30 05:26:35.115000
_forge_v19.py            99427  2026-04-30 05:26:35.159000
agent.py                  4190  2026-04-30 05:26:35.115000
forge_agent.py            8689  2026-04-30 05:26:35.115000
greedy_explore_agent.py   3383  2026-04-30 05:26:35.139000
qwen_agent.py            17152  2026-04-30 05:26:35.109000
random_agent.py           1638  2026-04-30 05:26:35.092000
ready
```

**Result: PASS.** All 7 .py files uploaded; dataset status is `ready`.

---

## 5. Step 2 - Push the bundler kernel

### 5.1 Why this Step exists

Qwen3.6-35B-A3B BF16 is ~70 GB - too large to ship in our repo, and the
competition kernel cannot fetch it from HuggingFace because
`enable_internet=false` is enforced during eval. The bundler kernel runs
**once** with `enable_internet=true` to:
1. `huggingface_hub.snapshot_download` the weights into the kernel's `/tmp/`
2. Upload them as a private Kaggle Dataset
   (`cataluna84/qwen3-6-35b-a3b-bf16`).

After this Step finishes, the dev/comp kernels can mount the Dataset under
`/kaggle/input/qwen3-6-35b-a3b-bf16/` with no internet required.

### 5.2 - 2026-04-30 05:27 UTC - Push the kernel

```bash
.venv/bin/kaggle kernels push -p experiments/exp004_qwen_agent/bundle_qwen_kernel
```

Output:
```
Kernel version 1 successfully pushed.
Please check progress at https://www.kaggle.com/code/cataluna84/qwen-bundle-arc-agi-3
```

The push itself is fast (the .ipynb + metadata are <10 KB). Kaggle
schedules it on a CPU worker (because `enable_gpu=false`).

### 5.3 - 2026-04-30 05:27..05:47 UTC - Poll until COMPLETE

```bash
KERNEL=cataluna84/qwen-bundle-arc-agi-3
for i in $(seq 1 60); do
  S=$(.venv/bin/kaggle kernels status "$KERNEL" 2>&1 | head -1)
  echo "[$(date -u +%H:%M:%S)] iter=$i $S"
  if echo "$S" | grep -q "COMPLETE\|ERROR\|FAIL\|CANCELLED"; then break; fi
  sleep 60
done
```

Iterations 1..18 reported `RUNNING`; iteration 19 (`05:47:15` UTC)
reported `COMPLETE`. **Total wall clock: ~19 minutes** (faster than the
runbook's 45-60 min projection - HF mirror was hot, network was fast).

### 5.4 - 2026-04-30 05:49 UTC - Fetch outputs and inspect SUMMARY

```bash
mkdir -p runs/qwen_bundle && rm -rf runs/qwen_bundle/*
.venv/bin/kaggle kernels output cataluna84/qwen-bundle-arc-agi-3 -p runs/qwen_bundle/
uv run python -c "
import json
log = open('runs/qwen_bundle/qwen-bundle-arc-agi-3.log').read()
events = json.loads(log)
out = ''.join(e['data'] for e in events if e.get('stream_name') == 'stdout')
print(out[-3000:])
"
```

Tail of stdout (`=== SUMMARY ===` block):
```
HF source        : Qwen/Qwen3.6-35B-A3B
local cache      : /tmp/qwen36_bf16
total bytes      : 71.93 GB
download time    : 6.4 min
upload time      : 12.4 min
Kaggle Dataset   : cataluna84/qwen3-6-35b-a3b-bf16
uploaded ok      : True
```

### 5.5 Pass-criteria check (`RUNBOOK_D2.md` § 3)

| Check | Threshold | Observed | Result |
|-------|-----------|----------|--------|
| `uploaded ok` is True | True | True | PASS |
| Total bytes 65-75 GB | 65-75 GB | 71.93 GB | PASS |
| Download time < 50 min | < 50 min | 6.4 min | PASS |
| Upload time < 50 min | < 50 min | 12.4 min | PASS |

### 5.6 - 2026-04-30 05:49 UTC - Verify weights Dataset

```bash
.venv/bin/kaggle datasets status cataluna84/qwen3-6-35b-a3b-bf16
.venv/bin/kaggle datasets files cataluna84/qwen3-6-35b-a3b-bf16 | head -40
```

Status: `ready`. Files listed include all 26 safetensors shards
(`model-00001-of-00026.safetensors` ... `model-00026-of-00026.safetensors`)
plus `config.json`, `generation_config.json`, `tokenizer.json` (via
`merges.txt` / `vocab.json`), `chat_template.jinja`, and
`model.safetensors.index.json`.

**Result: PASS.** The dev kernel can now mount these weights.

---

## 6. Step 3 - Push the dev kernel (smoke test on H100)

### 6.1 Why this Step exists

The dev kernel is where we validate that QwenAgent can:
- Load the bundled Qwen3.6-35B-A3B BF16 weights from `/kaggle/input/...`
- Run inference on H100 (~30s model load, < 10s/action target)
- Emit valid `GameAction` enums into the offline arc-agi SDK loop
- Make at least 1 level of progress on `ls20`

**This step does NOT submit to the competition.** It produces
`/kaggle/working/qwen_smoke.json` with per-action latency stats.

### 6.2 - 2026-04-30 05:50..06:22 UTC - Iteration log

#### v1 (push 05:50, error 05:57 - 6 min in)

```bash
.venv/bin/kaggle kernels push -p experiments/exp004_qwen_agent/dev_kernel \
  --accelerator NvidiaH100
```

Error: `Qwen dataset not mounted at /kaggle/input/qwen3-6-35b-a3b-bf16`.
Available `/kaggle/input/` entries were `competitions` and `datasets` -
**the kernel image is using a new nested layout we hadn't seen before**.

Lesson: Kaggle now mounts datasets at
`/kaggle/input/datasets/<owner>/<slug>/` (nested) and competitions at
`/kaggle/input/competitions/<comp-slug>/` rather than the flat
`/kaggle/input/<slug>/` layout that older docs describe. Adapter code must
try both.

#### v2 (push 06:00, error 06:01 - 1 min)

After fixing the nested-layout adapter in `qwen_agent_dev.ipynb`
(`_CANDIDATES = [legacy, nested, recursive]`), the weights, agents-pkg,
and competition env files were all found correctly:

```
Qwen weights at /kaggle/input/datasets/cataluna84/qwen3-6-35b-a3b-bf16
copied agents pkg from /kaggle/input/datasets/cataluna84/arc-agi-3-agents-pkg -> /kaggle/working/agents
ENVIRONMENTS_DIR = /kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files
```

But `Arcade()` tried to call `https://three.arcprize.org/api/games/anonkey`
and got `NameResolutionError` because `enable_internet=false`.

Lesson: the arc-agi SDK ships with `OperationMode.NORMAL` by default,
which calls the public ARC API. For the offline competition runtime we
need `OperationMode.OFFLINE` plus an explicit `environments_dir` pointing
at the mounted competition env files.

#### v3 (push 06:05, error 06:06 - 1 min)

After setting `OPERATION_MODE=offline` and `ENVIRONMENTS_DIR=...` and
constructing `Arcade(operation_mode=OperationMode.OFFLINE,
environments_dir=ENV_DIR)`:

```
ENVIRONMENTS_DIR = /kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files
  contains 25 game dirs
INFO:arc_agi.scorecard:Initialized ScorecardManager with idle_for=0:15:00 ...
2026-04-30 06:05:36 | INFO | Created new scorecard: ...
2026-04-30 06:05:36 | INFO | Found latest version of ls20: ls20-9607627b
2026-04-30 06:05:36 | INFO | Successfully loaded game class Ls20
starting smoke run on ls20, max_actions=50
```

Then: `transformers` import triggered `torchvision` -> `PIL.ImageDraw` ->
`PIL.ImageText` -> `from ._typing import _Ink` ->
`ImportError: cannot import name '_Ink' from 'PIL._typing'`.

Lesson: the Kaggle H100 image ships **Pillow 11.3.0** which has a
packaging bug - `ImageText.py` imports a name not present in `_typing.py`.
This breaks every `transformers` model that uses an image processor (which
includes anything `AutoModelForImageTextToText` touches).

#### v4 (push 06:09, error 06:14 - 5 min)

Tried `pip install --no-index --upgrade --force-reinstall --no-deps
pillow` from the bundled wheels (which include `pillow-12.2.0`).
Pip *did* install Pillow 12.2.0 but PIL still errored because the C
extension `_imaging.cpython-312-x86_64-linux-gnu.so` was still the old
11.3.0 build (the system image keeps the .so at a path pip can't replace
cleanly):

```
ImportError: The _imaging extension was built for another version of Pillow or PIL:
Core version: 11.3.0
Pillow version: 12.2.0
```

Lesson: Kaggle's image has Pillow's Python sources at
`/usr/local/lib/python3.12/dist-packages/PIL/` but the C extension is
preserved across pip operations because the dist-info / RECORD doesn't
fully cover the .so path on this image. `pip uninstall pillow` removes
some files; the next `pip install pillow` adds the new .py + new .so to
the same dir, but Python's already-cached `_imaging.so` path still wins.

#### v5 (push 06:17, error 06:18 - 1 min)

Same C-extension-mismatch error - confirmed our hypothesis.

#### v6 (push 06:21, error 06:22 - 1 min)

**Fix that finally worked for PIL**: install Pillow into a separate
target directory and prepend it to `sys.path` so import-time resolution
finds the new build first, before reaching the broken system PIL:

```python
PILLOW_TARGET = '/kaggle/working/_pillow_pkg'
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',
                f'--find-links={WHEELS}', '--upgrade',
                '--target', PILLOW_TARGET, 'pillow'], check=False)
sys.path.insert(0, PILLOW_TARGET)
for mod in [m for m in list(sys.modules) if m.startswith('PIL')]:
    del sys.modules[mod]
```

Result:
```
PIL ok: 12.2.0 at /kaggle/working/_pillow_pkg/PIL/__init__.py
```

But the **next** error appeared - and this one is structural:
```
ValueError: The checkpoint you are trying to load has model type
`qwen3_5_moe` but Transformers does not recognize this architecture.
```

The Kaggle image ships **transformers 5.0.0** (per H100 probe v2). The
Qwen3.6-35B-A3B model uses architecture `qwen3_5_moe` which was
introduced in a **post-5.0.0** transformers release. The bundled
competition wheels (`/kaggle/input/competitions/.../arc_agi_3_wheels/`)
do **not** include a transformers wheel.

### 6.3 - 2026-04-30 06:22 UTC - Diagnosis: hard blocker for D2

The dev kernel is now blocked on:
- Kaggle image's `transformers 5.0.0` does not recognize `qwen3_5_moe`.
- The competition wheel bundle has no transformers upgrade.
- We cannot `pip install transformers --upgrade` from the internet
  because the dev kernel has `enable_internet=false`.

To unblock, we need one of:

1. **Bundle a newer transformers wheel as a separate Kaggle Dataset.**
   Run a one-shot bundler kernel with `enable_internet=true` that
   downloads `transformers >= 5.x-with-qwen3_5_moe-support` (and any new
   deps), uploads as `cataluna84/transformers-qwen3-compat-bundle`, then
   attach that Dataset to the dev kernel.

2. **Switch the model to one supported by transformers 5.0.0.** Likely
   candidates that fit in 80 GB BF16 / INT4 on a single H100:
   - `Qwen/Qwen2.5-VL-7B-Instruct` (vision, supported).
   - `Qwen/Qwen3-32B` (text-only, supported, no MoE).
   - `Qwen/Qwen3-Next-80B-A3B-INT4` from Kaggle Models registry (MoE,
     INT4 native, may also need newer transformers - check).

3. **Burn the daily slot on Track A (FORGE resubmit) instead.** This
   doesn't progress exp004 but it does collect the exp002 variance probe
   data point we should have had on D1, and it costs zero compute on
   our side.

### 6.4 What we DID achieve in Step 3 (kept, not wasted)

Even though the dev kernel did not produce a green smoke, we
**permanently fixed**:

- Path-discovery code that handles both legacy (`/kaggle/input/<slug>/`)
  and nested (`/kaggle/input/datasets/<owner>/<slug>/`) Kaggle layouts -
  **needed for every future kernel push on the new image**.
- arc-agi SDK offline-mode setup (`OperationMode.OFFLINE` +
  `environments_dir=...` + the `OPERATION_MODE` env var).
- Pillow `--target` install + `sys.path.insert` fix - **applies to every
  vision-model kernel pushed to this image** until Kaggle ships a fixed
  Pillow build. Also recorded as a gotcha.

The dev_kernel notebook now contains all these fixes for the next push.

---

## 7. Status as of D2 06:22 UTC and decision point

| Item | Status |
|------|--------|
| Pre-checks P1..P5 | DONE (all PASS) |
| Step 1: agents-pkg Dataset | DONE (PASS) |
| Step 2: weights bundler kernel | DONE (PASS, 71.93 GB uploaded) |
| Step 3: dev kernel smoke | BLOCKED (transformers 5.0.0 < qwen3_5_moe) |
| Step 4-6: comp kernel + submit | NOT REACHED |

Daily Kaggle slot: **still open** as of 06:22 UTC. Operator decision
needed - see § 6.3 options.

**Decision: Track B (bundle newer transformers as Dataset)**.

---

## 8. Track B - bundle transformers 5.7.0 as Kaggle Dataset

### 8.1 First attempt - Kaggle bundler kernel (failed at API)

We initially built `experiments/exp004_qwen_agent/transformers_bundle_kernel/`
- a one-shot internet-enabled kernel that does `pip download transformers
tokenizers accelerate huggingface_hub safetensors regex filelock fsspec
pyyaml tqdm` into `/tmp/transformers_pkg/wheels/`, writes
`dataset-metadata.json`, and uploads via `kagglehub.dataset_upload()`.

```bash
.venv/bin/kaggle kernels push -p experiments/exp004_qwen_agent/transformers_bundle_kernel
```

The kernel ran in ~1 min and uploaded all 10 wheels (visible in stdout as
"Upload successful" rows). But **finalizing the Dataset version 403'd**
on `https://api.kaggle.com/v1/datasets.DatasetApiService/CreateDatasetVersion`
- this API endpoint requires a Kaggle login the kernel context cannot
provide. The CLI fallback (`kaggle datasets create`) hit the title
already-in-use-by-a-notebook conflict (the bundler kernel itself reserves
that title).

Lesson: bundler-kernel-builds-Dataset is a fragile path because the kernel
can upload files into a Dataset but cannot create a *new* Dataset - that
requires a CreateDatasetVersion call from a logged-in user context.

### 8.2 Pivot: bundle locally (we have internet on dev box)

Faster, simpler path: download the wheels with our local `pip` and
upload via `kaggle datasets create`. Steps:

```bash
# 1. install pip into our venv (uv-managed venvs don't ship pip by default)
uv pip install pip --python .venv/bin/python

# 2. download wheels for cp312 / manylinux into a staging dir
.venv/bin/python -m pip download \
    --dest /tmp/transformers_pkg/wheels \
    --python-version 3.12 \
    --platform manylinux_2_28_x86_64 \
    --platform manylinux_2_27_x86_64 \
    --platform manylinux2014_x86_64 \
    --only-binary=:all: \
    --no-deps \
    transformers tokenizers accelerate huggingface_hub safetensors \
    regex filelock fsspec pyyaml tqdm

# 3. write dataset-metadata.json
cat > /tmp/transformers_pkg/dataset-metadata.json <<'JSON'
{
  "title": "ARC-AGI-3 transformers wheels (cataluna84)",
  "id": "cataluna84/arc-agi-3-transformers-wheels",
  "licenses": [{"name": "apache-2.0"}],
  "subtitle": "transformers 5.7.0 + qwen3 deps for offline ARC-AGI-3 kernels",
  "description": "Mirrored wheels of transformers + tokenizers + accelerate + huggingface_hub + safetensors + ...",
  "isPrivate": true
}
JSON

# 4. create the Dataset
.venv/bin/kaggle datasets create -p /tmp/transformers_pkg --dir-mode tar
```

Output: `Your private Dataset is being created. Please check progress at
https://www.kaggle.com/datasets/cataluna84/arc-agi-3-transformers-wheels`.

After ~10 sec the status was `ready` with all 10 wheels listed.

### 8.3 Dataset v2 - re-pin tokenizers

The first bundle had `tokenizers-0.23.1`, but `transformers 5.7.0`
requires `tokenizers>=0.22.0,<=0.23.0`. Re-download with the constraint:

```bash
rm -f /tmp/transformers_pkg/wheels/tokenizers-*.whl
.venv/bin/python -m pip download \
    --dest /tmp/transformers_pkg/wheels \
    --python-version 3.12 \
    --platform manylinux_2_28_x86_64 ... --only-binary=:all: --no-deps \
    'tokenizers>=0.22.0,<=0.23.0'
# downloaded tokenizers-0.22.2

.venv/bin/kaggle datasets version -p /tmp/transformers_pkg --dir-mode tar \
    -m "tokenizers pinned to <=0.23.0 for transformers 5.7.0 compat"
```

### 8.4 Final dev kernel install pattern

```python
TX_TARGET = '/kaggle/working/_transformers_pkg'
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',
                f'--find-links={TX_WHEELS}', '--upgrade',
                '--target', TX_TARGET, '--no-deps',
                'transformers', 'tokenizers', 'accelerate',
                'huggingface_hub', 'safetensors',
                'regex', 'filelock', 'fsspec', 'pyyaml', 'tqdm'])
sys.path.insert(0, TX_TARGET)
for mod in [m for m in list(sys.modules)
            if m.split('.')[0] in {'transformers', 'tokenizers', 'accelerate',
                                   'huggingface_hub', 'safetensors'}]:
    del sys.modules[mod]
```

Critical flags:
- `--no-deps`: pip would otherwise try to resolve `numpy>=1.17` against
  `--no-index` and fail.
- `--target <dir>`: user-site doesn't beat system site-packages on this
  image; only an explicit target dir + `sys.path.insert(0, ...)` wins.
- `--no-index --find-links=<mount>`: ensures we install from the Dataset
  mount, not PyPI (which is unreachable).

### 8.5 Iteration log v6..v9

| Push | UTC | Status | Symptom |
|------|-----|--------|---------|
| v6 | 06:21 | ERROR | qwen3_5_moe not recognized (transformers 5.0.0) |
| v7 | 06:46 | ERROR | pip install of transformers tried to resolve numpy>=1.17 (forgot --no-deps) |
| v8 | 06:50 | ERROR | tokenizers 0.23.1 > transformers 5.7.0's <=0.23.0 cap |
| v9 | 06:54 | **COMPLETE** at 07:11 (16.5 min) | smoke run on ls20, 50 actions, 853s |

### 8.6 v9 - the green run

```
Qwen weights at /kaggle/input/datasets/cataluna84/qwen3-6-35b-a3b-bf16
Total size: 71.926853455 GB
PIL ok: 12.2.0 at /kaggle/working/_pillow_pkg/PIL/__init__.py
transformers ok: 5.7.0 at /kaggle/working/_transformers_pkg/transformers/__init__.py
copied agents pkg from /kaggle/input/datasets/cataluna84/arc-agi-3-agents-pkg
ENVIRONMENTS_DIR = /kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files
  contains 25 game dirs
INFO:arc_agi.scorecard:Initialized ScorecardManager ...
2026-04-30 06:54:29 | INFO | Successfully loaded game class Ls20 from ...
starting smoke run on ls20, max_actions=50
  [step 0]  ACTION1  state=NOT_FINISHED  level=0  dt=579.50s   (model load)
  [step 1]  ACTION1  state=NOT_FINISHED  level=0  dt=5.47s
  ...
  [step 49] ACTION1  state=NOT_FINISHED  level=0  dt=5.73s

=== SMOKE SUMMARY ===
  game             : ls20
  actions taken    : 50
  levels_completed : 0
  win_levels       : 7
  final state      : GameState.NOT_FINISHED
  wall clock       : 853.5s (17.07s/action)
saved /kaggle/working/qwen_smoke.json
```

### 8.7 What v9 proves

- Track B works end-to-end: transformers 5.7.0 loads from
  `/kaggle/working/_transformers_pkg/`, qwen3_5_moe is recognized,
  Qwen3.6-35B-A3B BF16 model loads on H100 (~580s including weights from
  Dataset mount), and inference runs at ~5.5s/action steady-state.
- All four Kaggle-image gotchas (#11 nested mount, #12 SDK anonkey,
  #13 PIL C-ext, #14 transformers 5.0.0) have working fixes in the
  dev kernel notebook.
- 50 actions on ls20 yielded 0 levels completed - the agent always took
  ACTION1, which is consistent with the prompt template not yet biasing
  the model toward action diversity. This is a *prompt/agent-logic*
  problem, not an *infrastructure* problem.

### 8.8 Cost projection for full submission

50 actions x 5.5 s/action steady-state = ~275s after model load.
Full submission: 25 games x ~250-500 actions/game (eval may cap at
~300 actions/game given 6h wall) = 7500-12500 actions. At 5.5s/action
that is 11-19 hours - **above the 6h cap**.

The model load (580s) is amortized across all games in one kernel run
since the model stays loaded. So the per-action steady-state matters most.

To fit in 6h:
- Action budget: 6h * 3600 / 5.5 = 3927 actions across 25 games
- ~157 actions/game average
- Need to either reduce per-action latency (smaller image, kv-cache reuse,
  shorter prompts) or accept fewer actions per game.

---

## 9. Status as of D2 07:11 UTC after v9 green

| Item | Status |
|------|--------|
| Pre-checks P1..P5 | DONE (all PASS) |
| Step 1: agents-pkg Dataset | DONE (PASS) |
| Step 2: weights bundler kernel | DONE (PASS, 71.93 GB) |
| Track B: transformers wheels Dataset | DONE (PASS, v2) |
| Step 3: dev kernel smoke | DONE (v9 PASS, 50 actions in 853s) |
| Step 4: comp kernel push | OPEN |
| Step 5: submit-code | OPEN (slot still available) |

---

## 10. v10/v11 quality + speed iteration (07:27..08:03 UTC)

After v9 confirmed end-to-end infrastructure, two issues remained:
- **Speed**: 17 s/action wall-clock (incl. model load amortized) - projected
  11h+ for full eval, above 6h cap.
- **Quality**: agent picked ACTION1 50 times in a row on ls20, 0 levels
  completed.

### 10.1 v10 - tighten decode budget + drop text grid (07:27 UTC)

```python
# agents/qwen_agent.py
_DEFAULT_MAX_NEW = 16        # was 96 - decode-time = N tokens * ~57 ms
# build_prompt(): drop the "Text grid (rows top-to-bottom, hex digits):"
# block from user_text. The image already encodes the grid; the text grid
# duplicates it and ~quadruples input tokens.
# Tighten system prompt to "Reply with ONE action only, NO explanation".
```

`scripts/qwen_agent_smoke_local.py` updated to assert the new prompt
shape (drop "Text grid" check). All 21/21 checks pass. Bumped agents-pkg
Dataset to v2 with `kaggle datasets version`.

**v10 result**: per-action 5.96 s steady-state (only ~0.5s improvement over
v9's 5.5s). Conclusion: **decode-time was NOT the bottleneck** - prefill on
the upscaled 512x512 image dominates. Cutting `max_new_tokens` from 96 to
16 only changed ~1s of decode.

The agent still picked ACTION1 50 times in a row.

### 10.2 v11 - anti-repeat exploration (07:47 UTC)

The greedy decode locks Qwen into the first-feasible action because the
prompt + frame embedding drive the same first generated token every turn.
Added a deterministic post-processor in `QwenAgent.choose_action`:

```python
STUCK_THRESHOLD = 3
recent_no_change = [name for name, changed in self._history if not changed]
n_recent_same = sum(1 for n in recent_no_change[-STUCK_THRESHOLD:] if n == action.name)
if n_recent_same >= STUCK_THRESHOLD and avail:
    sorted_avail = sorted(avail)
    idx = sorted_avail.index(int(action.name.replace("ACTION", "")))
    next_id = sorted_avail[(idx + 1) % len(sorted_avail)]
    action = GameAction.from_id(next_id)
```

If the agent has tried the same action 3+ times recently AND the frame
didn't change, rotate to the next available action.

**v11 result** (08:03 UTC):
- Action distribution: ACTION1 x 31, ACTION2 x 19 (no longer monotonic).
- Per-action steady-state: 5.13 s (small improvement, prefill still
  dominates).
- 0 levels completed in 50 actions on ls20.

The anti-repeat rotation cycles only between ACTION1 and ACTION2 because
the rotation is per-action, not per-orbit; once the agent flips to
ACTION2 and that yields no-change, the recent_no_change deque is mostly
ACTION2 entries, so the model's next ACTION1 prediction is allowed.
Rotates ACTION1 -> ACTION2 -> back to ACTION1 in cycles.

### 10.3 Speed math after v10/v11

| Metric | v9 | v11 |
|--------|----|----|
| First call (model load) | 579.5 s | 528.9 s |
| Steady-state per-action | 5.50 s | 5.13 s |
| Total for 50 actions on ls20 | 853 s | 780 s |
| Levels completed | 0 | 0 |

Projection for full submission (25 games):
- 25 games x 50 actions x 5.1 s + 528 s model load = 6900 s = **1.92 h**
- 25 games x 200 actions x 5.1 s + 528 s = 26 028 s = 7.23 h - over cap
- 25 games x 150 actions x 5.1 s + 528 s = 19 653 s = 5.46 h - tight but OK

In other words: with v11 speed, **~150 actions/game** fits the 6h cap.
But the smoke shows 0 levels in 50 actions, so 150 actions probably
nets 0-1 level/game. Expected score: 0.0-0.1 - **below the 0.19 baseline**.

### 10.4 What v11 proved vs what's missing

Proved:
- Track B infrastructure works end-to-end on H100 (latest transformers,
  qwen3_5_moe loads, BF16 generation runs).
- Anti-repeat rotation prevents the trivial monotonic-action failure
  mode.

Missing for D2 submission:
- **Action diversity**: only ACTION1/2 explored. Need anti-repeat to
  cycle through all actions when stuck, or a temperature > 0.0 sampling
  pass.
- **Vision-side speed**: prefill is the bottleneck; need
  - smaller image resolution (drop _FRAME_UPSCALE 8 -> 4 -> 256x256 px)
  - vllm/sglang (per HF model card recommendation) instead of
    transformers.generate
  - kv-cache persistence between turns (prompt prefix is identical
    across calls in the same game)
- **ACTION6 (click) coordinates**: untested in v11 because the model
  never picked it.
- **Hybrid policy**: model-driven on 2-3 games, fast random/greedy on
  the rest, to bank a 0.19+ score before iterating Qwen.

### 10.5 Decision for today's daily slot

The dev kernel is green at the infrastructure level but the agent's
expected LB score is below 0.19. Submitting today as-is regresses our
LB position.

Recommended path:
- DO NOT submit Qwen today.
- Consider Track A (re-submit ForgeAgent) for variance baseline data.
- Spend tomorrow on speed (vllm or smaller model) + diversity (better
  anti-repeat, bigger action set).
