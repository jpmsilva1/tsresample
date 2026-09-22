# tsresample: entry point for AI coding agents

This repo is built step by step by agents following a written plan. Before doing
anything, read, in order:

1. `Blueprint/PROGRESS.md`: what is done, and what is next. Take your step from here.
2. `CONTRIBUTING.md`: how to claim a step, and the branch, commit and PR rules.
3. `Blueprint/CLAUDE.md`: the standing engineering rules. Despite the file name, they
   apply to every agent and tool, not only Claude.
4. `Blueprint/docs/SPEC.md` §0 and `Blueprint/docs/adr/0011-r-algorithm-fidelity.md`.
5. Your step's brief in `Blueprint/ops/TASK_GRAPH.md`.

Role instructions (implementer, researcher, reviewer, replicator) are in
`.claude/agents/*.md`. If your tool doesn't load them automatically, read the one for
your role and follow it.

Never open a path listed in `Blueprint/docs/PROVENANCE.md` §1 (GPL quarantine).
