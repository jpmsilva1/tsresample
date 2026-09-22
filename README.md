# tsresample

A Python library (planned MIT licence) implementing the resampling strategies of Moniz,
Branco & Torgo (2017), *Resampling strategies for imbalanced time series forecasting*,
plus the relevance-based metrics used to evaluate them. It matches the **original R
algorithms**, quirks included.

**Status: pre-implementation.** The specification, design decisions, verification data
and build plan are complete and live in [`Blueprint/`](Blueprint/). The library is built
step by step, by the maintainer and collaborators working with AI agents, following that
plan. See [`Blueprint/PROGRESS.md`](Blueprint/PROGRESS.md) for what is done and what is
next.

## Start here (humans and agents)

Every session, in this order:

1. **[`Blueprint/PROGRESS.md`](Blueprint/PROGRESS.md):** what is done, when, by whom, and
   what is next. Take your step from here, and claim it before starting (see
   `CONTRIBUTING.md`). Don't rely on memory or old notes.
2. **[`CONTRIBUTING.md`](CONTRIBUTING.md):** how to claim a step, and the branch, commit
   and PR rules.
3. **[`Blueprint/CLAUDE.md`](Blueprint/CLAUDE.md):** the standing engineering rules: the
   spec is the contract, ambiguity is resolved by ADR, cleanroom, numbers come from
   oracles, TDD, one step per session. They apply whichever agent you use.
4. **[`Blueprint/docs/SPEC.md`](Blueprint/docs/SPEC.md) §0 and
   [ADR-0011](Blueprint/docs/adr/0011-r-algorithm-fidelity.md):** which natural-looking
   formulas are wrong, and why R's behaviour is the default.
5. **Your step's brief** in [`Blueprint/ops/TASK_GRAPH.md`](Blueprint/ops/TASK_GRAPH.md),
   plus the SPEC sections and ADRs it names.
   [`Blueprint/ops/KICKOFF.md`](Blueprint/ops/KICKOFF.md) has ready-made prompts for the
   first steps.
6. **Before opening any external code,** check
   [`Blueprint/docs/PROVENANCE.md`](Blueprint/docs/PROVENANCE.md) §1. Several reference
   implementations are GPL and quarantined.
7. **Finish** by meeting the definition of done in
   [`Blueprint/ops/QUALITY_GATES.md`](Blueprint/ops/QUALITY_GATES.md) §4 and updating
   `Blueprint/PROGRESS.md` in the same PR.

If the repo contradicts itself (a path that doesn't resolve, two different tolerances),
stop and open an issue rather than working around it.

## Layout

| Path | What it is |
|---|---|
| `Blueprint/` | The plan: spec, ADRs, gates, task graph, progress, and verification data. Index: [`Blueprint/README.md`](Blueprint/README.md). |
| `CONTRIBUTING.md` | Collaboration workflow: claiming steps, branches, commits, PRs, reviews. |
| `CLAUDE.md`, `AGENTS.md` | Entry points that agent tools load automatically (Claude Code; Codex, Cursor and others). Both point into `Blueprint/`. |
| `.claude/agents/` | The four agent roles (`implementer`, `researcher`, `reviewer`, `replicator`), auto-discovered by Claude Code. With other tools, use them as role prompts. |
| `.github/` | PR checklist template and code owners. |

Library code (`src/`, `tests/`, `pyproject.toml`) arrives at the repo root in step 1.
