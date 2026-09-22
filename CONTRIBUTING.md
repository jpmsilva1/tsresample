# Contributing

This project is built by a maintainer (@jpmsilva1) and collaborators, each usually
working with an AI agent. The rules below keep parallel work from colliding and keep the
repo in a state any newcomer, human or agent, can pick up cold.

## The workflow for one step

1. **Pick.** Open `Blueprint/PROGRESS.md`. Your step is the first row whose status is
   `Not started` and whose dependencies are all `Done`, unless the maintainer assigned you
   a different one. Before claiming, check the open pull requests: a draft PR titled
   `Step <id>: …` means someone already has it.
2. **Claim.** Create a branch `step/<id>-<short-slug>` (for example `step/2a-phi`). Your
   first commit sets the row to `In progress (YYYY-MM-DD, @your-handle)`. Open a **draft
   PR** titled `Step <id>: <node name>` right away, so the claim is visible to everyone.
   A claim with no activity for 14 days may be released by the maintainer.
3. **Plan (GATED steps only).** The tier is in `Blueprint/ops/TASK_GRAPH.md`. For a GATED
   step, post your plan as a PR comment and wait for the maintainer's approval before
   implementing.
4. **Build.** One fresh agent session per step. Follow `Blueprint/CLAUDE.md`: TDD in
   vertical slices, spec as contract, ambiguity resolved by ADR, cleanroom. Keep the PR to
   that one step; if you notice something unrelated, add it to the Open items table in
   `Blueprint/PROGRESS.md` instead of fixing it.
5. **Review.** Run an independent `reviewer` pass (`.claude/agents/reviewer.md`) in a
   **fresh** session that never saw the implementation session. Paste its verdict and
   findings into the PR, and address them or explain why they are declined.
6. **Finish.** In the same PR, set the row to `Done (YYYY-MM-DD)` and fill **Evidence**
   (this PR, gate output, ADR). Mark the PR ready. The maintainer merges; delete the
   branch after the merge.

## Branches and commits

- Never commit to `main` directly. Everything lands through a PR.
- Branch prefixes: `step/` for a PROGRESS step, `fix/` for a bug outside a step,
  `blueprint/` for plan changes outside a step, `docs/` for documentation only.
- [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `test:`,
  `docs:`, `refactor:`, `chore:`.
- No AI co-author trailers (`Co-Authored-By: Claude` and similar) on commits.
- PRs are squash-merged, so the PR title becomes the commit on `main`. Write it as a
  Conventional Commit too.

## Changing the Blueprint

`Blueprint/` is the contract. Changes to it need the maintainer's approval, always:

- **SPEC or ADR changes.** A new ambiguity gets a new ADR (the next free number; check
  open PRs for one already in flight) before any code depends on it. If a step reveals a
  spec error, fix the spec in the same PR and say so in the PR description.
- **Tolerances, coverage floor, versions.** These are stated once, in
  `Blueprint/ops/QUALITY_GATES.md` §1. Change them only with an ADR.
- **Runtime dependencies.** Adding one is always GATED.
- **Verification data** (`Blueprint/tests/fixtures/`, `Blueprint/replication/`). Oracles
  hold recorded R numbers. Never regenerate a fixture from our own code.

## Agents

Any agent tool can be used. `CLAUDE.md` and `AGENTS.md` at the root are the entry
points that tools load automatically; both lead to the same rules.

- The model routing in `Blueprint/ops/TASK_GRAPH.md` names Claude models. With another
  tool, use its strongest model for research, GATED steps and review, and a fast model
  for AUTO steps.
- One step per session. The reviewer never sees the implementer's transcript; the
  independence is the point.
- Before opening any external code, check `Blueprint/docs/PROVENANCE.md` §1. Opening a
  quarantined (GPL) path taints the work, and the fix is a rewrite. Every gate asks
  about it.

## Keeping the repo tidy

- No absolute or machine-local paths (`/Users/...`, `C:\...`). External data comes from
  the pinned public replication repo (`Blueprint/docs/REPLICATION.md` §1).
- Don't commit virtualenvs, caches, local notes, stray worktrees, or files over 10 MB.
  Raise large data in an issue first.
- Delete your branch after merge.
- If you find the repo contradicting itself, open an issue instead of working around it.

## Questions

Open a GitHub issue, or ask in your step's PR.
