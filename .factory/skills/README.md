# .factory/skills/

Project-scoped Factory.ai skills (reusable agent capabilities) live here.

- Format: one folder per skill, with a `SKILL.md` describing when/how to invoke it.
- Docs: https://docs.factory.ai/cli/configuration/skills

## Candidate skills for this project

- `arc3-replay-watcher/` — given a `replay.jsonl` from `three.arcprize.org/replay/...`, render the action trace + frame deltas, identify failure modes.
- `kaggle-submit-preflight/` — run all V1–V8 checks from `.factory/verify.md` automatically before any Kaggle submission.
- `paper-distiller/` — given an arXiv URL, produce a 1-page implementation-focused summary suitable for `research/`.

(All empty for now — drop skill folders here when needed.)
