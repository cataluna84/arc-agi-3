"""local_runner.py - offline smoke test harness for ARC-AGI-3 agents.

Goal: never burn a Kaggle daily slot on a smoke test.

Usage:
    .venv/bin/python experiments/local_runner.py \\
        --agent agents.random_agent:RandomAgent \\
        --games ls20,ft09 \\
        --max-actions 200 \\
        --seed 0 \\
        --use-sdk \\
        --json-out runs/local/$(date +%s).json

Modes:
    --use-sdk       : use the real arc-agi SDK (requires the wheels to have been
                      installed via scripts/install_arc_agi_sdk.py).
    (default)       : use the offline `MockGame` -- a tiny deterministic env
                      that responds to RESET / ACTION1..ACTION7 with frame
                      deltas. Useful for unit-testing the agent loop end-to-end
                      without internet or the SDK.

Output JSON shape:
    {
      "runner_version": "2.0",
      "timestamp": "2026-04-29T...",
      "agent": "agents.random_agent:RandomAgent",
      "games": [
        {
          "game_id": "ls20",
          "backend": "sdk" | "mock",
          "actions_taken": 17,
          "levels_completed": 1,
          "win_levels": 7,
          "final_state": "WIN" | "GAME_OVER" | "NOT_FINISHED",
          "wall_clock_s": 0.42,
          "action_histogram": {"ACTION1": 3, "ACTION6": 11, "RESET": 3}
        }
      ],
      "totals": {"games": 1, "wins": 1, "actions_taken": 17, "wall_clock_s": 0.42}
    }
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from typing import Any

# Make `agents/` discoverable when run from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agents import SDK_AVAILABLE, GameAction, GameState, MockFrame  # noqa: E402

# ---------------------------------------------------------------------------
# Mock environment (offline, no SDK required)
# ---------------------------------------------------------------------------


class MockGame:
    """Tiny deterministic mock that pretends to be an arc-agi-3 game.

    Behaviour:
      - 64x64 grid, palette 0..15.
      - The "win condition" is: pixel (10, 10) == 7.
      - ACTION1 increments pixel (10,10) by 1 (mod 16).
      - ACTION2 decrements pixel (10,10) by 1 (mod 16).
      - ACTION6(x, y) sets pixel (x, y) to 7.
      - All other simple actions are no-ops on the grid.
      - Each successful WIN increments levels_completed; up to MAX_LEVELS levels.
    """

    MAX_LEVELS = 3
    MAX_ACTIONS_PER_LEVEL = 100

    def __init__(self, game_id: str, seed: int = 0) -> None:
        self.game_id = game_id
        self._rng = random.Random(seed)
        self._state: Any = GameState.NOT_FINISHED
        self._levels_completed: int = 0
        self._actions_this_level: int = 0
        self._grid: list[list[int]] = [[0] * 64 for _ in range(64)]

    # ---- public API matching arc_agi.EnvironmentWrapper as closely as reasonable

    @property
    def observation_space(self) -> MockFrame:
        return self._frame()

    @property
    def action_space(self) -> list[GameAction]:
        return [GameAction.from_id(i) for i in (1, 2, 3, 4, 5, 6, 7)]

    def reset(self) -> MockFrame:
        self._grid = [[0] * 64 for _ in range(64)]
        self._actions_this_level = 0
        self._state = GameState.NOT_FINISHED
        return self._frame(full_reset=True)

    def step(
        self, action: GameAction, data: dict | None = None, reasoning: Any | None = None
    ) -> MockFrame:
        del reasoning  # unused in mock
        # Real GameAction is a plain Enum; .value holds the int.
        action_id = int(action.value) if hasattr(action, "value") else int(action)

        # RESET handling
        if action_id == GameAction.RESET.value:
            return self.reset()

        self._actions_this_level += 1

        # Simple actions
        if action_id == GameAction.ACTION1.value:
            self._grid[10][10] = (self._grid[10][10] + 1) % 16
        elif action_id == GameAction.ACTION2.value:
            self._grid[10][10] = (self._grid[10][10] - 1) % 16
        elif action_id == GameAction.ACTION6.value:
            # Pull (x, y) either from the supplied `data` dict or from the
            # GameAction's bound action_data attribute (which set_data() updates).
            x = y = 0
            if data:
                x = int(data.get("x", 0))
                y = int(data.get("y", 0))
            else:
                ad = getattr(action, "action_data", None)
                if ad is not None:
                    x = int(getattr(ad, "x", 0))
                    y = int(getattr(ad, "y", 0))
            if 0 <= x < 64 and 0 <= y < 64:
                self._grid[y][x] = 7

        # Win check
        if self._grid[10][10] == 7:
            self._levels_completed += 1
            if self._levels_completed >= self.MAX_LEVELS:
                self._state = GameState.WIN
            else:
                # Reset the working grid for the next level but keep level count.
                self._grid = [[0] * 64 for _ in range(64)]
                self._actions_this_level = 0
                self._state = GameState.NOT_FINISHED
        elif self._actions_this_level >= self.MAX_ACTIONS_PER_LEVEL:
            self._state = GameState.GAME_OVER

        return self._frame()

    def _frame(self, full_reset: bool = False) -> MockFrame:
        return MockFrame(
            game_id=self.game_id,
            state=self._state,
            levels_completed=self._levels_completed,
            win_levels=self.MAX_LEVELS,
            available_actions=[1, 2, 3, 4, 5, 6, 7],
            frame=[[row[:] for row in self._grid]],
            guid=None,
            full_reset=full_reset,
        )


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------


def load_agent(spec: str):
    """Spec format: 'package.module:ClassName'."""
    if ":" not in spec:
        raise ValueError(f"--agent spec must look like 'pkg.module:ClassName', got: {spec!r}")
    mod_path, cls_name = spec.split(":", 1)
    module = importlib.import_module(mod_path)
    return getattr(module, cls_name)


# ---------------------------------------------------------------------------
# SDK-backed environment
# ---------------------------------------------------------------------------


def maybe_load_sdk_env(game_id: str) -> tuple[Any, str]:
    """Try to load the real arc_agi SDK env for `game_id`.

    Returns (env_obj, kind) or (None, "") if SDK isn't available locally.
    """
    if not SDK_AVAILABLE:
        return None, ""
    try:
        from arc_agi import Arcade  # type: ignore
    except ImportError:
        return None, ""
    try:
        arcade = Arcade()
        env = arcade.make(game_id)
        return env, "sdk"
    except Exception as e:
        print(
            f"[local_runner] SDK present but failed to create {game_id}: {e}",
            file=sys.stderr,
        )
        return None, ""


# ---------------------------------------------------------------------------
# Run a single game
# ---------------------------------------------------------------------------


def _action_data_dump(action: GameAction) -> dict[str, Any]:
    """Pull the action's bound action_data into a plain dict for env.step().

    For simple actions this is empty; for complex (ACTION6) it contains x/y.
    """
    ad = getattr(action, "action_data", None)
    if ad is None:
        return {}
    dump = getattr(ad, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            pass
    # Fallback for the offline mock IntEnum stand-in
    return getattr(action, "_data", {}) or {}


def run_game(
    game_id: str,
    AgentCls,
    *,
    max_actions: int,
    seed: int,
    use_sdk: bool,
) -> dict[str, Any]:
    env: Any = None
    backend = "mock"

    if use_sdk:
        env, kind = maybe_load_sdk_env(game_id)
        if env is not None:
            backend = kind  # "sdk"

    if env is None:
        env = MockGame(game_id, seed=seed)
        env.reset()

    # Initial observation: real SDK exposes `observation_space` as the latest
    # FrameDataRaw; our MockGame mirrors the same property.
    frame = env.observation_space

    # Some agents (e.g. ForgeAgent) need access to the running env to look up
    # game source files for their planner. Pass it through if the constructor
    # accepts it; otherwise fall back to the simple (seed,) signature.
    import inspect as _inspect

    try:
        sig = _inspect.signature(AgentCls.__init__)
        if "arc_env" in sig.parameters:
            agent = AgentCls(seed=seed, arc_env=env, game_id=game_id)
        else:
            agent = AgentCls(seed=seed)
    except (ValueError, TypeError):
        agent = AgentCls(seed=seed)
    agent_name = getattr(agent, "name", AgentCls.__name__)

    histogram: Counter = Counter()
    actions_taken = 0
    final_state: Any = getattr(frame, "state", GameState.NOT_FINISHED)
    levels_completed: int = getattr(frame, "levels_completed", 0)
    win_levels: int = getattr(frame, "win_levels", 1)

    t0 = time.time()
    for _ in range(max_actions):
        # Allow the agent to short-circuit (WIN, etc.)
        if hasattr(agent, "is_done") and agent.is_done(frame):
            break

        action = agent.choose_action(frame)
        if not isinstance(action, GameAction):
            # Be permissive: accept raw int as a hint. The real GameAction is
            # a plain Enum (not IntEnum), so we must go through from_id().
            action = GameAction.from_id(int(action))

        histogram[action.name] += 1
        actions_taken += 1

        try:
            next_frame = env.step(
                action,
                data=_action_data_dump(action),
                reasoning=None,
            )
        except Exception as e:
            print(
                f"[local_runner] env.step({action.name}) raised: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            break

        if next_frame is None:
            print(
                f"[local_runner] env.step({action.name}) returned None; stopping {game_id}.",
                file=sys.stderr,
            )
            break

        frame = next_frame
        final_state = frame.state
        levels_completed = getattr(frame, "levels_completed", levels_completed)
        win_levels = getattr(frame, "win_levels", win_levels)

        # WIN/GAME_OVER from the env terminate the loop unconditionally.
        if frame.state in (GameState.WIN, GameState.GAME_OVER):
            break

    # Normalize state to its string value for JSON output.
    final_state_str = final_state.value if hasattr(final_state, "value") else str(final_state)

    return {
        "game_id": game_id,
        "agent": agent_name,
        "backend": backend,
        "actions_taken": actions_taken,
        "levels_completed": levels_completed,
        "win_levels": win_levels,
        "final_state": final_state_str,
        "wall_clock_s": round(time.time() - t0, 4),
        "action_histogram": dict(histogram),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Offline smoke runner for ARC-AGI-3 agents.")
    p.add_argument("--agent", required=True, help="'pkg.module:ClassName' of the agent to run.")
    p.add_argument("--games", default="ls20-mock", help="Comma-separated list of game IDs.")
    p.add_argument(
        "--max-actions", type=int, default=200, help="Max actions per game (safety cap)."
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--use-sdk",
        action="store_true",
        help="Use the real arc-agi SDK (requires the wheels installed).",
    )
    p.add_argument("--json-out", default=None, help="If set, write the full report to this path.")
    p.add_argument(
        "--quiet-sdk-logs", action="store_true", help="Silence INFO-level logs from the SDK."
    )
    args = p.parse_args(argv)

    if args.quiet_sdk_logs:
        logging.basicConfig(level=logging.WARNING)
        # arcengine uses loguru-style logging via the standard root logger.
        for name in ("arc_agi", "arc_agi.scorecard", "arcengine"):
            logging.getLogger(name).setLevel(logging.WARNING)

    AgentCls = load_agent(args.agent)
    games = [g.strip() for g in args.games.split(",") if g.strip()]

    results: list[dict[str, Any]] = []
    for gid in games:
        res = run_game(
            gid,
            AgentCls,
            max_actions=args.max_actions,
            seed=args.seed,
            use_sdk=args.use_sdk,
        )
        results.append(res)
        print(json.dumps(res, indent=2))

    totals = {
        "games": len(results),
        "wins": sum(1 for r in results if r["final_state"] == GameState.WIN.value),
        "actions_taken": sum(r["actions_taken"] for r in results),
        "wall_clock_s": round(sum(r["wall_clock_s"] for r in results), 4),
    }

    report = {
        "runner_version": "2.0",
        "timestamp": datetime.now(UTC).isoformat(),
        "agent": args.agent,
        "games": results,
        "totals": totals,
    }

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\n[local_runner] wrote {args.json_out}")

    print("\n[local_runner] totals:", json.dumps(totals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
