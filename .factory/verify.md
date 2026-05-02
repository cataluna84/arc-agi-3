# VERIFY.md — exact commands to prove things work

> Run these *every time* before burning a Kaggle submission slot.
> If any block fails, fix the cause **before** uploading.
>
> **Quick path** for a routine D5+ submission (skip V1-V6 once they have
> run green once on this machine):
>
> ```bash
> uv run ruff check . && uv run ruff format --check .
> uv run pytest tests/ -v
> uv run python scripts/<your_agent>_smoke_local.py
> uv run python experiments/local_runner.py \
>     --agent agents.<your_agent>:<YourClass> \
>     --use-sdk --games <2-3 game ids> --max-actions 300
> # All green? Then proceed to V10 + V7.
> ```

---

## V1. Repo + Memory layout sanity

```bash
cd /home/cataluna84/Workspace/arc-agi-3

# Layout matches the Factory.ai canonical pattern
ls AGENTS.md .factory/plan.md .factory/memories.md .factory/verify.md
ls .factory/rules/*.md   # expect 7 rules files

# Documentation extracted (plain-text mirrors of the MHTMLs)
ls documentation/kaggle/*_extracted.txt | wc -l   # expect 5
```

Expected: 5 extracted txt files, all 4 memory files exist, no errors.

## V2. Local environment setup (Python 3.12)

```bash
python3 --version                       # expect Python 3.12.x
python3 -m pip --version                # confirm pip available
python3 -m pip install --user uv numpy pillow tqdm   # base toolkit
```

If `pip` is missing:
```bash
python3 -m ensurepip --default-pip && python3 -m pip install --upgrade pip
```

## V3. ARC-AGI-3 toolkit local install

```bash
# Requires internet on the dev machine; in-Kaggle this is replaced by --find-links
python3 -m pip install --user arc-agi
python3 -c "import arc_agi; from arcengine import GameAction; print(GameAction.ACTION1)"
```

Expected: prints something like `GameAction.ACTION1`.

## V4. Smoke test custom agent on a public game

```bash
# After cloning ARC-AGI-3-Agents (only needed locally)
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
cd ARC-AGI-3-Agents
cp .env.example .env
# Optional: set ARC_API_KEY=... for online mode; leave empty for offline anonymous
ONLINE_ONLY=False uv run main.py --agent=random --game=ls20 || \
    python3 main.py --agent=random --game=ls20
```

Expected: agent runs, terminates with `WIN` or `GAME_OVER`, prints scorecard JSON.

## V5. Custom agent skeleton compiles

Create `experiments/sanity_agent.py`:

```python
from agents.agent import Agent
from agents.structs import FrameData, GameAction, GameState
import random
class SanityAgent(Agent):
    def is_done(self, frames, latest):
        return latest.state is GameState.WIN
    def choose_action(self, frames, latest):
        if latest.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET
        a = random.choice([x for x in GameAction if x is not GameAction.RESET])
        if a.is_complex():
            a.set_data({"x": random.randint(0,63), "y": random.randint(0,63)})
        return a
```

Run:
```bash
python3 -c "from experiments.sanity_agent import SanityAgent; a=SanityAgent(); print(a)"
```

Expected: no ImportError, no syntax error.

## V6. Pre-submission Kaggle notebook checklist

Run inside the Kaggle notebook (or commit a "pre-flight" cell):

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
print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')

# 5. Estimate runtime: dry-run 10 actions on 1 game
# ...drop in your agent here, time 10 calls...
```

Expected: every assertion passes, GPU shows "H100" if H100 attached.

## V7. Submission via Kaggle CLI (code competition pattern)

> **Critical CLI gotcha (#15)**: there is **no** `kaggle competitions submit-code`
> subcommand in the pinned CLI 2.1.0. Several public docs reference it.
> Use plain `submit` with `-k <kernel> -v <version> -f <output_file>`.

```bash
# Pre-flight: confirm yesterday's slot is closed (need a >= 24 h gap)
.venv/bin/kaggle competitions submissions arc-prize-2026-arc-agi-3 | head -3

# 1. Push your kernel (this just creates the version; does NOT submit)
.venv/bin/kaggle kernels push -p experiments/expNNN_<slug>/comp_kernel/

# 2. Wait for the kernel to finish save-mode run (typically 20-60 s for
#    pure-Python agents; 5-30 min for agents with model loading).
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    s=$(.venv/bin/kaggle kernels status cataluna84/<your-kernel-slug> 2>&1 | tail -1)
    echo "[poll $i] $s"
    if echo "$s" | grep -qE "(COMPLETE|ERROR|CANCEL)"; then break; fi
    sleep 30
done

# 3. Sanity-check the kernel output (save-mode log + dummy parquet)
mkdir -p /tmp/kernel_out
.venv/bin/kaggle kernels output cataluna84/<your-kernel-slug> -p /tmp/kernel_out
ls -la /tmp/kernel_out/                    # expect submission.parquet (~2.6 KB save-mode placeholder)
tail -20 /tmp/kernel_out/<your-kernel-slug>.log    # confirm no tracebacks

# 4. Submit (BURNS DAILY SLOT — exactly once per 24 h on Kaggle)
.venv/bin/kaggle competitions submit arc-prize-2026-arc-agi-3 \
    -k cataluna84/<your-kernel-slug> \
    -v <kernel-version-number> \
    -f submission.parquet \
    -m "expNNN <date> <one-line-rationale>"

# 5. Confirm acceptance
.venv/bin/kaggle competitions submissions arc-prize-2026-arc-agi-3 | head -3
# Expect a new row with status PENDING and your description.
```

LB result lands ~24 h later. Until then status is PENDING.

After the LB result lands:

1. Open the **Submissions** tab on the Kaggle competition page (or use
   `kaggle competitions submissions` to see it from the CLI).
2. Confirm "Public score" appears.
3. Compare to the previous best in `.factory/memories.md`.
4. If `Δscore < -0.02`, immediately:
   - **Rollback**: re-submit the previous best notebook version
   - **Bisect**: identify which change degraded the score
5. Append a new dated entry to `.factory/memories.md` with:
   - Score delta vs prior best
   - Per-game breakdown (download from the Submissions detail view)
   - Top 3 failure modes observed (look at the replays)
   - Next-step decisions
6. Update `experiments/expNNN_<slug>/scores.json` with the new
   `data_points[]` entry; populate `summary_when_resolved` if it's the
   final probe of an experiment.

## V8. Reproducibility log

For every successful submission save:

```
experiments/expNNN_<slug>/
├── notebook.ipynb         # frozen at the time of submission
├── notes.md               # 1-page rationale
├── score.txt              # public-LB score
├── per_game_scores.csv    # per-game RHAE
└── replays/               # replay JSONs from arcprize replay viewer (optional)
```

The intent is that 6 months from now you can `cd` into any experiment folder and *exactly* replay it.

## V9. Kaggle data + SDK setup

```bash
# 9a. .env exists and is populated
test -f /home/cataluna84/Workspace/arc-agi-3/.env && echo "[ok] .env present" || echo "[fail] copy .env.example to .env and fill in creds"

# 9b. Dev deps installed (kaggle, kagglehub, python-dotenv)
python3 -c "import kaggle, kagglehub, dotenv; print('ok')"

# 9c. Auth round-trip (lists comp metadata; needs network + valid token + accepted rules)
python3 -c "
from dotenv import load_dotenv; load_dotenv('.env')
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
print('[ok] authenticated as:', api.config_values['username'])
"

# 9d. Competition data downloaded (idempotent; reads .env automatically)
python3 scripts/download_kaggle_data.py

# 9e. Verify expected layout (~25 game dirs + N wheels)
ls data/kaggle/arc-prize-2026-arc-agi-3/environment_files/ | wc -l
ls data/kaggle/arc-prize-2026-arc-agi-3/arc_agi_3_wheels/*.whl 2>/dev/null | wc -l

# 9f. Install arc-agi SDK from local wheels (offline-equivalent of the Kaggle notebook step)
python3 scripts/install_arc_agi_sdk.py

# 9g. Round-trip the SDK with a real game via local_runner
python3 experiments/local_runner.py \
    --agent agents.random_agent:RandomAgent \
    --use-sdk \
    --games <real-id-from-environment_files> \
    --max-actions 50
```

Expected: every block exits 0; final `local_runner` returns valid stats (WIN or GAME_OVER, not crash).

## V10. Trigger-BFS / state-graph smoke (post-D4)

The state-graph foundation introduced 2026-05-02 must stay green for any
agent that depends on `agents.state_graph`. Run after every change to
`agents/state_graph.py`, `agents/trigger_bfs_agent.py`, or any future
agent that imports `StateGraph`/`hash_frame`.

```bash
# State-graph unit tests (5 tests, expect 5/5 PASS)
uv run pytest tests/test_state_graph.py -v

# 22-check parity smoke (pure-Python; no GPU; no network)
uv run python scripts/trigger_bfs_smoke_local.py

# 25-game SDK sweep (small budget; minutes-class)
uv run python experiments/local_runner.py \
    --agent agents.trigger_bfs_agent:TriggerBFSAgent \
    --use-sdk \
    --games ar25,bp35,cd82,cn04,dc22,ft09,g50t,ka59,lf52,lp85,ls20,m0r0,r11l,re86,s5i5,sb26,sc25,sk48,sp80,su15,tn36,tr87,tu93,vc33,wa30 \
    --max-actions 300 --seed 0 --quiet-sdk-logs \
    --json-out /tmp/trigger_bfs_sweep.json

# Quick stats on the sweep
uv run python -c "
import json
d=json.load(open('/tmp/trigger_bfs_sweep.json'))
games=d['games']
total=sum(g['levels_completed'] for g in games)
any_=[g for g in games if g['levels_completed']>=1]
print(f'25-game sweep: total levels={total}, games with >=1 level={len(any_)}')
"
```

Expected (as of 2026-05-02 baseline):
- pytest: 5 / 5 PASS in <2 s.
- 22-check smoke: 22 / 22 PASS, mock end-to-end ls20 win in ~21 actions.
- 25-game sweep: 1 / 25 games clear level 1 (ft09 with seed=0). Net
  parity with RandomAgent's 1/25 (r11l). If you see 0 / 25, something
  broke; bisect agent changes.

## V11. scores.json schema (post-D2)

Every submitting experiment owns a `scores.json` file. Validate it with:

```bash
uv run python -c "
import json, sys
from pathlib import Path

bad = []
for fp in Path('experiments').glob('exp*/scores.json'):
    d = json.loads(fp.read_text())
    required = {'experiment','kernel_slug','competition','data_points'}
    missing = required - set(d)
    if missing:
        bad.append((fp, f'missing keys: {missing}'))
        continue
    for i, p in enumerate(d['data_points']):
        for k in ('label','submitted_at_utc','status'):
            if k not in p:
                bad.append((fp, f'data_point[{i}] missing {k}'))
        if p.get('status') == 'COMPLETE' and 'score' not in p:
            bad.append((fp, f'data_point[{i}] is COMPLETE but score absent'))

if bad:
    for fp, msg in bad: print(f'FAIL {fp}: {msg}')
    sys.exit(1)
print('all scores.json valid')
"
```

Expected: `all scores.json valid`. If a probe entry is `COMPLETE` but
missing a `score`, fill it in from `kaggle competitions submissions ...`
before merging.
