# Kickoff

## Setup checklist

Before starting node 1 (each contributor, on their own machine):

- [ ] Clone https://github.com/jpmsilva1/tsresample and read `Blueprint/PROGRESS.md` and
      `CONTRIBUTING.md`.
- [ ] Python 3.10+ available; a virtualenv created.
- [ ] Only needed for gate G4 (node 11) and for re-running the M0 probe: clone
      https://github.com/jpmsilva1/ts-resampling-replication at the pinned commit and export
      `TSRESAMPLE_REPLICATION_ROOT` (exact commands in `Blueprint/docs/REPLICATION.md` §1).
      Everything else runs without it.
- [ ] Confirm `Blueprint/tests/fixtures/phi_oracle.json` is present (~29 KB, rebuilt 2026-09-22 with
      R's per-split control points in `r_splits`). Gate G0 depends on it and it is committed,
      not generated.
- [ ] (Maintainer, PROGRESS step 0) PyPI Trusted Publishing configured for the project name
      `tsresample`, and branch protection on `main`. Do this early; node 13 is otherwise a
      scramble at the worst moment.
- [ ] Read `Blueprint/docs/SPEC.md` §0 (v0.8.0) and ADR-0011. Every session's first act.

## Day 1

Paste this to start:

```
Read CLAUDE.md, then Blueprint/docs/SPEC.md §0 and §3 and §6, then Blueprint/ops/TASK_GRAPH.md.

Execute node 1 (Scaffold), tier AUTO, as the `implementer` agent.

Build the package skeleton only — every module is a stub with a docstring and no
logic. Wire the toolchain and CI so that every later node inherits working gates:
hatchling + src/ layout, version single-sourced from __init__.py, ruff, mypy --strict,
pytest with the `replication` and `slow` markers, coverage floor read from
pyproject.toml, and the Layer-1/pipeline import check from SPEC §3 as a real test.
Move Blueprint/pyproject.toml to the repo root (git mv) and build on it; code goes in
src/ and tests/ at the repo root, and tests read oracles from Blueprint/tests/fixtures/.

Do not implement embed(), phi, or anything else. Report the verbatim output of every
gate command you run, and answer the provenance question from Blueprint/ops/QUALITY_GATES.md §5.
```

Then wave 2 — two parallel sessions:

```
Read CLAUDE.md, Blueprint/docs/SPEC.md §0 and §4.1, Blueprint/docs/adr/0001 (with its v0.8.0 amendment),
Blueprint/docs/adr/0002, Blueprint/docs/adr/0008, Blueprint/docs/adr/0011, Blueprint/docs/REPLICATION.md §2, Blueprint/ops/QUALITY_GATES.md
§1 and §2.

Execute node 2a (φ), tier AUTO, as the `implementer` agent.

Note before you start: four natural choices for this function are WRONG and would each
pass a carelessly written test -- medcouple, PchipInterpolator, np.percentile quartiles,
and control points at the fences. R uses Tukey hinges, whisker ends, phi=0 on a side with
no outliers, and constant extension. ADR-0001's amendment has the numbers. phi must also
return the exact endpoint value at and beyond the knots and be clipped to [0, 1] (SPEC
§4.1 Step 4, ADR-0014) -- node 8's SERA gate fails by up to 0.5% without it.

Finish by demonstrating that gate G0 Test B fails when PchipInterpolator is swapped in,
and separately when the v0.7.0 fence formula is. A gate that cannot fail is not a gate.
```

```
Read CLAUDE.md, Blueprint/docs/SPEC.md §0 and §2.1, Blueprint/docs/adr/0005.

Execute node 2b (Embedding), tier AUTO, as the `implementer` agent.
```

Wave 3 starts node 3 **and** node M0. Start M0 now rather than when its dependents are
ready — it is research with an unknown completion time and it gates the whole metrics half
of the library. Everything else on the critical path is implementation against a written
spec.

```
Read CLAUDE.md, Blueprint/docs/SPEC.md §4.7, Blueprint/docs/adr/0009 (with its v0.8.0 amendment),
Blueprint/docs/REPLICATION.md §1 and §3, Blueprint/ops/RESEARCH_VERIFICATION.md.

Execute node M0 (Metric oracle probe), tier GATED, as the `researcher` agent.

This is research. Do not write metrics.py.

The metric conventions are already pinned (>= t_E, 1e-5 on empty selection, |1+u|,
p = 0.5), and the utility surface is known to be exactly recoverable. Your job is a CLEAN
derivation of U from Ribeiro (2011) section 4, verified against
raw_iterations_by_dataset_v2/ to the G0b tolerance. Start with the recorded rows where
prec = 1e-05; they are the easiest cases.

SERA is already closed (REPLICATION.md §3.2); do not reopen it.

Check Blueprint/replication/MANIFEST.csv's provenance column before trusting any oracle file.
CLAUDE.md rule 3 applies with full force: uba's source, and the colleague's Python port
listed in PROVENANCE.md §1, would answer this in thirty seconds and you may not open them.
```

## After each node

Run the `reviewer` agent in a **fresh session**:

```
Read Blueprint/docs/SPEC.md, the ADRs for node <N>, and Blueprint/ops/QUALITY_GATES.md.

Review node <N> adversarially as the `reviewer` agent. You did not write this and you
must not trust the implementer's summary of it -- read the diff, run the gates yourself,
and try to make each gate fail.

Answer gate P explicitly.
```

Do not give the reviewer the implementer's transcript. The independence is the mechanism;
sharing context defeats it more thoroughly than using a weaker model would.

Once the reviewer's verdict is **merge**, the node's PR sets its row in `Blueprint/PROGRESS.md` to
`Done (date)` with the evidence. The next session starts by reading `Blueprint/PROGRESS.md`.

## What you will actually be doing

Two check-ins per GATED node — plan approval, then merge approval. AUTO nodes report at the
end. Dispatch order, model routing, tiering and what happens when a node stalls are all
written into `CLAUDE.md` and `Blueprint/ops/TASK_GRAPH.md`, so you should not need to re-explain any
of it.

## A note on guarantees

This kit removes every ambiguity we could find, and the numbers in it were verified against
recorded data rather than asserted. It cannot guarantee how any agent behaves on a given
run. If something does not match what is written here — a file path that does not resolve,
a tolerance that contradicts `Blueprint/ops/QUALITY_GATES.md` §1 — that is the signal to pause and
fix the kit, not to push through.
