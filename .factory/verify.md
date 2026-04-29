# VERIFY.md — exact commands to prove things work

> Run these *every time* before burning a Kaggle submission slot.
> If any block fails, fix the cause **before** uploading.

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

## V7. Post-submission verification

After committing the notebook:

1. Open the **Submissions** tab on the Kaggle competition page.
2. Confirm "Public score" appears within ~30 minutes.
3. Compare to the previous best in `.factory/memories.md`.
4. If `Δscore < -0.02`, immediately:
   - **Rollback**: re-submit the previous best notebook version
   - **Bisect**: identify which change degraded the score
5. Append a new dated entry to `.factory/memories.md` with:
   - Score delta vs prior best
   - Per-game breakdown (download from the Submissions detail view)
   - Top 3 failure modes observed (look at the replays)
   - Next-step decisions

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
