# Blueprint

The complete plan for `tsresample`: what to build, why each decision was made, how
"correct" is proven, and the order of work. If you are starting a session, go back to the
[root README](../README.md) for the reading order. This page is the index.

All paths in these documents are written from the **repository root** (`Blueprint/docs/SPEC.md`,
not `docs/SPEC.md`). Paths without the `Blueprint/` prefix (`src/`, `tests/`,
`pyproject.toml`, `docs/MIGRATION.md`) refer to library files at the repo root.

| Path | What it is |
|---|---|
| `PROGRESS.md` | Status of every step, with dates and who did it. **Read first.** |
| `CLAUDE.md` | Standing engineering rules for every session and every agent tool. |
| `docs/SPEC.md` | What to build, exactly. Normative. §0 lists what changed and why. |
| `docs/adr/` | ADR-0001–0014: why each formula or convention is what it is. |
| `docs/REPLICATION.md` | The oracle index, and how each gate proves agreement with R. |
| `docs/PROVENANCE.md` | What you may not read (GPL quarantine), and the audit log. |
| `ops/TASK_GRAPH.md` | The steps (nodes), dependencies, tiers, model routing and briefs. |
| `ops/QUALITY_GATES.md` | Every tolerance, stated once; the test seams; definition of done. |
| `ops/KICKOFF.md` | Setup checklist and the literal prompts for the first sessions. |
| `ops/RESEARCH_VERIFICATION.md` | How to verify a research (non-code) claim. |
| `tests/fixtures/` | Committed oracles: `phi_oracle.json` (gate G0), `metric_oracle.json` (gate G0b). The library's tests read them from here. |
| `replication/` | Vendored source series and a subset of recorded R outputs. `MANIFEST.csv` lists each file's source and validity (one folder is flagged CONTAMINATED). |

**External data.** Gate G4 (step 11) and re-running research probes need the full
recorded R experiment. It is in the public repo
[jpmsilva1/ts-resampling-replication](https://github.com/jpmsilva1/ts-resampling-replication),
pinned to one commit; `docs/REPLICATION.md` §1 has the clone commands. Every other step
runs from this repo alone.
