"""Build qwen_phase1_dev.ipynb for exp004 Phase 1 dev/smoke (D17).

Run:
    uv run python experiments/exp004_qwen_agent/dev_kernel_v3/build_notebook.py

Produces:
    experiments/exp004_qwen_agent/dev_kernel_v3/qwen_phase1_dev.ipynb

Same cells 0-2 as comp_kernel_v2 (pip install + offline overlay + inlined
agent). Cells 3+ replace the competition-rerun guard with a 4-game
sequential smoke loop on ls20 + ft09 + vc33 + lp85 (100 actions each),
writing `/kaggle/working/qwen_phase1_smoke.json` with per-game + aggregate
metrics that gate the D17 comp push.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent

# Re-use the comp_kernel_v2 inlined-agent cells.
sys.path.insert(0, str(REPO / "experiments/exp004_qwen_agent/comp_kernel_v2"))
from build_notebook import (  # type: ignore
    CELL0_PIP,
    CELL1_OVERLAY,
    CELL5_DUMMY,
    cell2_writefile,
    make_cell,
)


CELL3_MARKDOWN = ["Phase-1 dev smoke: load model once, run 4 games sequentially, write metrics."]

CELL4_SMOKE = """\
import json, os, statistics, sys, time, traceback
from pathlib import Path


def find_qwen_model_path():
    \"\"\"Resolve mounted Kaggle dataset path for Qwen weights (gotcha #11).\"\"\"
    env_p = os.environ.get('QWEN_MODEL_PATH')
    if env_p and Path(env_p).exists() and any(Path(env_p).glob('config.json')):
        return env_p
    for root in [
        Path('/kaggle/input/qwen3-6-35b-a3b-bf16'),
        Path('/kaggle/input/datasets/cataluna84/qwen3-6-35b-a3b-bf16'),
        Path('/kaggle/input'),
    ]:
        if not root.exists():
            continue
        for cfg in root.rglob('config.json'):
            text = cfg.read_text(errors='ignore')[:4000].lower()
            path_text = str(cfg.parent).lower()
            if 'qwen' in text or 'qwen' in path_text:
                return str(cfg.parent)
    return None


qwen_path = find_qwen_model_path()
if qwen_path is None:
    raise RuntimeError('Could not locate Qwen weights under /kaggle/input/')
os.environ['QWEN_MODEL_PATH'] = qwen_path
os.environ['OPERATION_MODE'] = 'offline'   # gotcha #12
os.environ['QWEN_MAX_NEW_TOKENS'] = '24'
os.environ['QWEN_HISTORY_LEN'] = '6'
print(f'QWEN_MODEL_PATH = {qwen_path}')

sys.path.insert(0, '/kaggle/working')
from my_agent import QwenAgent           # noqa: E402

# Pick up the official harness env mounts.
env_dir = None
for candidate in [
    '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files',
    '/kaggle/input/arc-prize-2026-arc-agi-3/environment_files',
]:
    if os.path.isdir(candidate):
        env_dir = candidate
        break

print(f"env_dir = {env_dir}")
GAMES = ['ls20', 'ft09', 'vc33', 'lp85', 'r11l', 's5i5']
ACTIONS_PER_GAME = 100
RESULTS = {'games': [], 'env_dir': env_dir}

# Pre-load the model once so the load time is amortized.
agent = QwenAgent(seed=0)
model_load_t0 = time.time()
try:
    agent._ensure_model_loaded()
    model_load_s = round(time.time() - model_load_t0, 1)
    RESULTS['model_load_s'] = model_load_s
    print(f"model loaded in {model_load_s}s")
except Exception as exc:
    RESULTS['model_load_error'] = f"{type(exc).__name__}: {exc}"
    traceback.print_exc()
    with open('/kaggle/working/qwen_phase1_smoke.json', 'w') as f:
        json.dump(RESULTS, f, indent=2)
    raise

try:
    from arc_agi import Arcade, OperationMode
    if env_dir is not None:
        arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=env_dir)
    else:
        arc = Arcade(operation_mode=OperationMode.OFFLINE)
except Exception as exc:
    RESULTS['arcade_init_error'] = f"{type(exc).__name__}: {exc}"
    traceback.print_exc()
    with open('/kaggle/working/qwen_phase1_smoke.json', 'w') as f:
        json.dump(RESULTS, f, indent=2)
    raise


def hash_grid(g):
    if g is None:
        return None
    try:
        return hash(tuple(tuple(int(v) for v in r) for r in g))
    except Exception:
        return None


for game_id in GAMES:
    print(f"\\n=== {game_id} ===")
    agent.reset_counters()
    per_game = {
        'game': game_id,
        'steps_taken': 0,
        'levels_completed': 0,
        'valid_action_count': 0,
        'latencies': [],
        'no_change_count': 0,
        'action_hist': {},
        'fallback_invocations': 0,
        'parse_failure_count': 0,
        'final_state': None,
        'error': None,
    }
    try:
        env = arc.make(game_id)
        frame = env.observation_space
        prev_hash = hash_grid(frame.frame[0]) if frame.frame else None
        for step_i in range(ACTIONS_PER_GAME):
            if frame.state.name in ('WIN', 'GAME_OVER'):
                break
            t0 = time.time()
            action = agent.choose_action(frame)
            elapsed = time.time() - t0
            per_game['latencies'].append(elapsed)
            aname = action.name
            per_game['action_hist'][aname] = per_game['action_hist'].get(aname, 0) + 1
            avail = [int(a) for a in (frame.available_actions or [])]
            if int(action.value) in avail:
                per_game['valid_action_count'] += 1
            # arcengine LocalEnvironmentWrapper.step signature (per memories):
            # step(action, data=..., reasoning=...) -> FrameDataRaw | None
            data = getattr(action, '_data', None) or {}
            if not data:
                ad = getattr(action, 'action_data', None)
                if ad is not None and action.is_complex():
                    data = {'x': int(getattr(ad, 'x', 32)), 'y': int(getattr(ad, 'y', 32))}
            nxt = env.step(action, data=data, reasoning=None)
            if nxt is None:
                per_game['final_state'] = 'NULL_RETURN'
                break
            cur_hash = hash_grid(nxt.frame[0]) if nxt.frame else None
            if cur_hash is not None and cur_hash == prev_hash:
                per_game['no_change_count'] += 1
            prev_hash = cur_hash
            per_game['levels_completed'] = max(
                per_game['levels_completed'],
                int(getattr(nxt, 'levels_completed', 0)),
            )
            frame = nxt
        per_game['final_state'] = (
            frame.state.name if hasattr(frame.state, 'name') else str(frame.state)
        )
    except Exception as exc:
        per_game['error'] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    per_game['steps_taken'] = len(per_game['latencies'])
    per_game['fallback_invocations'] = agent._fallback_count
    per_game['parse_failure_count'] = agent._parse_failure_count
    if per_game['latencies']:
        per_game['per_action_latency_p50'] = round(statistics.median(per_game['latencies']), 3)
        if len(per_game['latencies']) >= 5:
            sorted_lat = sorted(per_game['latencies'])
            p95_idx = max(0, int(0.95 * len(sorted_lat)) - 1)
            per_game['per_action_latency_p95'] = round(sorted_lat[p95_idx], 3)
        else:
            per_game['per_action_latency_p95'] = round(max(per_game['latencies']), 3)
    per_game['valid_action_rate'] = (
        per_game['valid_action_count'] / max(1, per_game['steps_taken'])
    )
    per_game['no_change_rate'] = per_game['no_change_count'] / max(1, per_game['steps_taken'])
    # Trim the raw latency list before dump.
    per_game.pop('latencies', None)
    per_game.pop('valid_action_count', None)
    per_game.pop('no_change_count', None)
    most_common = sorted(per_game['action_hist'].items(), key=lambda kv: -kv[1])
    per_game['most_common_action'] = most_common[0][0] if most_common else None
    RESULTS['games'].append(per_game)
    print(json.dumps(per_game, indent=2))

# Aggregate gates.
all_steps = sum(g['steps_taken'] for g in RESULTS['games'])
total_valid = sum(int(g['valid_action_rate'] * g['steps_taken']) for g in RESULTS['games'])
total_fb = sum(g['fallback_invocations'] for g in RESULTS['games'])
total_nc = sum(int(g['no_change_rate'] * g['steps_taken']) for g in RESULTS['games'])
all_lat = []
for g in RESULTS['games']:
    p95 = g.get('per_action_latency_p95')
    if p95 is not None:
        all_lat.append(p95)
RESULTS['total_steps'] = all_steps
RESULTS['max_levels_completed'] = max(
    (g['levels_completed'] for g in RESULTS['games']), default=0
)
RESULTS['aggregate_valid_action_rate'] = total_valid / max(1, all_steps)
RESULTS['aggregate_p95_latency'] = round(max(all_lat), 3) if all_lat else None
RESULTS['fallback_ratio'] = total_fb / max(1, all_steps)
RESULTS['aggregate_no_change_rate'] = total_nc / max(1, all_steps)

# Gate evaluation summary (5-gate threshold per D17 plan).
gates = {
    'levels_ge_1': RESULTS['max_levels_completed'] >= 1,
    'p95_le_7': (RESULTS['aggregate_p95_latency'] or 999) <= 7.0,
    'valid_rate_ge_0_95': RESULTS['aggregate_valid_action_rate'] >= 0.95,
    'fallback_le_0_20': RESULTS['fallback_ratio'] <= 0.20,
    'no_change_le_0_50': RESULTS['aggregate_no_change_rate'] <= 0.50,
}
RESULTS['gates'] = gates
RESULTS['all_gates_pass'] = all(gates.values())

with open('/kaggle/working/qwen_phase1_smoke.json', 'w') as f:
    json.dump(RESULTS, f, indent=2)
print(json.dumps(RESULTS, indent=2))
"""


def main() -> None:
    cells = [
        make_cell("code", CELL0_PIP),
        make_cell("code", CELL1_OVERLAY),
        make_cell("code", cell2_writefile()),
        make_cell("markdown", CELL3_MARKDOWN),
        make_cell("code", CELL4_SMOKE),
        make_cell("code", CELL5_DUMMY),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    out_path = REPO / "experiments/exp004_qwen_agent/dev_kernel_v3/qwen_phase1_dev.ipynb"
    with out_path.open("w") as f:
        json.dump(notebook, f, indent=1)
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
