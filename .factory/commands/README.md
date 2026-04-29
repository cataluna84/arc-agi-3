# .factory/commands/

Project-scoped Factory.ai custom slash commands live here.

- Format: one `.md` per command, with YAML frontmatter (`description`, `argument-hint`).
- Docs: https://docs.factory.ai/cli/configuration/custom-slash-commands
- Personal-scope variant: `~/.factory/commands/`.

## Candidate commands for this project

- `submit-baseline.md` — record an LB score, append a row to `.factory/memories.md`, and bump the score-anchor in `.factory/rules/leaderboard-anchors.md`.
- `new-exp.md` — scaffold `experiments/expNNN_<slug>/{README.md,score.txt,notes.md}` and add a row to `EXPERIMENTS.md`.
- `daily-recap.md` — read today's `.factory/memories.md` entry + per-exp `score.txt` and produce a 1-paragraph status summary.

(All empty for now — drop `.md` files in this folder when needed.)
