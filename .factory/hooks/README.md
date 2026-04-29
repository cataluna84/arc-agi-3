# .factory/hooks/

Project-scoped Factory.ai hook scripts live here.

- Hooks fire on lifecycle events: `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `SessionStart`, `Stop`, etc.
- Docs: https://docs.factory.ai/cli/configuration/hooks-guide
- Reference: https://docs.factory.ai/reference/hooks-reference
- Wired up in `~/.factory/settings.json` (or project-level overrides) under `"hooks"`.

## Candidate hooks for this project

- `memory-capture.py` — on `UserPromptSubmit`, intercept messages starting with `#` and append them to `.factory/memories.md` with today's date (per Factory's recommended pattern in https://docs.factory.ai/guides/power-user/memory-management).
- `lint-on-write.py` — on `PostToolUse` for `Edit|Create|ApplyPatch`, run `python3 -m py_compile` on touched `.py` files; abort on syntax error.
- `score-validator.py` — on `PostToolUse` for any write to `experiments/expNNN_*/score.txt`, validate the value parses as a float in [0, 1] and is monotonic vs prior recorded scores.

(All empty for now — drop executable scripts in this folder when needed and reference them in `~/.factory/settings.json`.)
