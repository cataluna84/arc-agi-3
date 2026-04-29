# .factory/droids/

Project-scoped Factory.ai custom droids (subagents) live here.

- Format: one `.yaml` per droid, defining `name`, `description`, `prompt`, `tools`, `model`.
- Docs: https://docs.factory.ai/cli/configuration/custom-droids
- Discovery hierarchy: `~/.factory/droids/` (personal) ← `.factory/droids/` (project) ← `<cwd>/.factory/droids/` (deepest wins).

## Candidate droids for this project

- `arc-game-explainer.yaml` — read a per-game `environment_files/<id>/` and produce a 1-page summary of the mechanics it likely encodes.
- `experiment-postmortem.yaml` — given an `expNNN_*` folder + LB score, draft the next-day experiment hypothesis.
- `notebook-shrinker.yaml` — minify and dead-code-strip a Kaggle notebook before submission to save runtime.

(All empty for now — drop YAMLs in this folder when needed.)
