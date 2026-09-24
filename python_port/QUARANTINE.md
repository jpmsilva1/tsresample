# python_port/: GPL-derived code, quarantined

This folder is the complete Python replication of Moniz, Branco & Torgo (2017) made for a
final-year project (TCC, CIn/UFPE) by @danielreinaux. It is a translation of the authors' R
experiment, **not** part of the `tsresample` library.

**Licence.** `Py_Code/tsresamp/` is a line-by-line port of GPL-licensed R/C code (`uba`'s
`phi.c`/`pchip.c`/`util.c`, the resampler bodies of `Exps.R`, `UBL::neighbours`), and its
MARS is a port of `earth` (GPL-3). It is therefore GPL-derived and is **not** covered by the
MIT licence of `tsresample`. Nothing in `src/` may import it, copy from it or be written
after reading it.

**Cleanroom.** `Blueprint/docs/PROVENANCE.md` §1 lists what an implementation agent must not
open. In this folder that is all the code: `Py_Code/`, `tests/`, `tools/`, `R_replication/*.R`
and `R_replication/audit/`. Numbers and prose are permitted, as they already
were: the `*.md` reports (`README.md`, `COMPARISON.md`, `REVIEW.md`, `RELATORIO_REPLICACAO.md`)
and `results_cluster/`.

**Layout differences from the standalone repository.**
- `Data/data_NM_PB_LT_DSAA2016.pkl` (15.6 MB) is not committed; the loader parses the
  `.Rdata` when the pickle is missing, or `python Py_Code/convert_rdata.py` recreates it.
- `R_replication/run_exps.R` and `run_table6.R` source the original `Exps.R`; they look for
  `R_Code/` next to this folder's parent or in a sibling clone `../TSResampStrat_JDSA2017`
  (github.com/nunompmoniz/TSResampStrat_JDSA2017).
- `tools/compare_fig7_all.py` reads the article PDF, which
  is not redistributed; pass its path or place it one level above this folder.
- Not copied here, kept in the standalone repository `TSResampStrat_Python`: the per-iteration
  scores of the original R code run locally (`R_replication/results/`, 188 CSVs; their summaries
  are the tables of `COMPARISON.md` §6-7 and `REVIEW.md` §7-8) and the article's figures
  (`Figures/`, unchanged in the original repository), the report PDF (generated from
  `RELATORIO_REPLICACAO.md`), the Apuana SLURM job (`cluster/apuana/`) with its logs and
  durations, the scripts that need the raw R CSVs (`R_replication/compare.py`,
  `tools/compare_mars_v2.py`) and the older Fig. 7 reader (`Py_Code/tools/read_fig7.py`).
  The reports below still mention those paths; they refer to the standalone repository.
  The R-report part of `tools/compare_full_run.py` is skipped here (it needs
  `R_replication/results/joao_report_F1.json`).
- Machine-specific details (cluster login, internal IPs, local paths) were replaced by
  placeholders.

Run from this folder: `pip install -r requirements.txt && python -m pytest tests -q`
(127 tests).
