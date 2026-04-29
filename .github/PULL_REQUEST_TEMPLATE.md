<!--
Thanks for contributing to arc-agi-3!

Before submitting:
- [ ] Run `uv run pre-commit run --all-files`
- [ ] Run `uv run pytest`
- [ ] Run `uv run python scripts/qwen_agent_smoke_local.py` if you touched any agent code
- [ ] Update CHANGELOG.md under [Unreleased] if this is user-visible
-->

## Summary

<!-- One-line description of WHAT this PR does. -->

## Why

<!-- Link the issue this closes (Closes #123) or describe the rationale. -->

## What changed

<!-- Bullet list of the concrete changes. Skip files that only got reformatted. -->

-

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change (changes existing behaviour)
- [ ] New agent (`agents/<name>.py`)
- [ ] New experiment (`experiments/expNNN_<slug>/`)
- [ ] Documentation only
- [ ] Tooling / CI / dev experience
- [ ] Refactor / cleanup

## How I tested it

<!-- Paste the smoke output, ruff result, etc. Specify which tests pass. -->

```text
# uv run pytest
# uv run python scripts/qwen_agent_smoke_local.py
# uv run python experiments/local_runner.py --agent ... --games ls20-mock
```

## Kaggle impact (if applicable)

- [ ] This change touches a Kaggle competition kernel
- [ ] This change is dev-only (no Kaggle daily slot impact)
- [ ] This change requires a new Kaggle Dataset / Model
- LB delta vs 0.19 baseline (if measured): __

## Checklist

- [ ] My code follows the project's style (`uv run ruff check .` and `ruff format --check .` are clean)
- [ ] I have added/updated tests where appropriate
- [ ] I have updated relevant documentation (`README.md`, `AGENTS.md`, `CONTRIBUTING.md`, or `.factory/`)
- [ ] I have added an entry to `CHANGELOG.md` under `[Unreleased]`
- [ ] I have NOT committed any secrets, model weights, or large data files
- [ ] I am licensing my contribution under Apache 2.0 (the project's license)
