# ADR-0003 — IRonPy is rejected as the φ oracle; a committed fixture replaces it

**Status:** Accepted · **Date:** 2026-09-03 · **Supersedes:** spec v0.6.0 §2.2; closes the
v0.6.0 kit's blocking `fact-check-request.md` Item 1

## Context

Spec v0.6.0 removed the R reference environment on the grounds that `IRonPy`
(`github.com/berreergun/IRonPy`, labelled MIT) could serve as the sole numerical oracle
for φ. Node 2a (`_relevance.py`) and its entire test group rested on that claim, and the
previous kit correctly flagged it as a blocker rather than cleanup.

The claim does not survive inspection of the actual repository.

## Decision

`IRonPy` is not used, not depended on, and not vendored. The φ oracle is
`Blueprint/tests/fixtures/phi_oracle.json`, a 7.6 KB committed file holding, per dataset: the R
control points for iteration 1 and the per-iteration median, the full-series control
points, the paper's Table 1 `%Rare`, and a truncated SHA-256 of the target column.

## Evidence

1. **It cannot run.** `iron/phi.py` dispatches on `sys.platform` and `cdll.LoadLibrary`s
   `phi.dll` (win32), `phi_mac.so` (darwin), or `phi_linux.so` (linux). Neither `.so`
   exists anywhere in the repository. The load is wrapped in a bare `except:` that prints
   and falls through, so the function returns `None` **silently** — a comparison test
   would receive `None`, not an exception, and could be written to pass against nothing.
2. **The shipped binaries are the wrong architecture.** `file` reports `iron/mc.dll` and
   `iron/phi.dll` as `PE32 … Intel 80386, for MS Windows`. The README confirms: *"For
   windows python32 is needed."* 32-bit Windows only.
3. **It will not install.** `setup.py` lists `install_requires=[…, 'sklearn']` — the
   deprecated shim package, which modern pip refuses — and `python_requires='>=2'`.
   `phi.py` calls `y.values`, so it accepts only a pandas Series.
4. **The MIT label does not cover the tree.** `src/` vendors R headers (`R.h`, R Core
   Team, LGPL-2.1+) and robustbase C sources (`mc.c`, `qn_sn.c`, `wgt_himed.c`, GPL ≥ 2).
   The DLLs export `py2phi`, i.e. IRonPy is a thin Windows wrapper around `uba`'s GPL-2
   `phi.c`/`pchip.c`. Depending on it would have imported exactly the GPL exposure the
   cleanroom rule exists to prevent.

## Consequences

- The v0.6.0 blocker on node 2a is **closed**, not deferred; `_relevance.py` can be
  written immediately.
- The oracle is now offline, deterministic, versioned, and reviewable in a diff — strictly
  better than a network- or platform-dependent one.
- Provenance is *improved*: the fixture is derived from output data of our own experiment
  runs, so no GPL source was read to obtain it. See `Blueprint/docs/PROVENANCE.md`.
- Ironically, adopting IRonPy would also have propagated its bugs. Both formula errors
  corrected in ADR-0001 and ADR-0002 were found only because the oracle was rebuilt from
  first principles instead of borrowed.
