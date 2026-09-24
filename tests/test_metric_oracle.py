"""Gate G0b: metrics against R's recorded values (Blueprint/docs/REPLICATION.md §3.4).

Expected values come from ``metric_oracle.json`` (recorded canonical R output),
never from our code. phi is R's recorded per-split control points.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tsresample.metrics import f1_phi, precision_phi, recall_phi, sera

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "Blueprint"
    / "tests"
    / "fixtures"
    / "metric_oracle.json"
)
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def _case(c: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array(c["y_true"]),
        np.array(c["y_pred"]),
        np.array(c["ctrl"]),
    )


IDS = [f"{c['dataset']}-{c['workflow']}-it{c['iteration']}" for c in CASES]


@pytest.mark.parametrize("c", CASES, ids=IDS)
def test_precision_recall_f1_match_r_within_1e_6(c: dict) -> None:
    y, yh, cp = _case(c)
    got = [f(y, yh, relevance=cp) for f in (precision_phi, recall_phi, f1_phi)]
    want = [float(c["prec"]), float(c["rec"]), float(c["F1"])]
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)


SERA_CASES = [c for c in CASES if c.get("sera_step_0_01") not in (None, "", "NA")]


@pytest.mark.parametrize(
    "c", SERA_CASES, ids=[f"{c['dataset']}-{c['workflow']}" for c in SERA_CASES]
)
def test_sera_at_step_0_01_matches_r_within_1e_6_relative(c: dict) -> None:
    y, yh, cp = _case(c)
    got = sera(y, yh, relevance=cp, step=0.01)
    assert got == pytest.approx(float(c["sera_step_0_01"]), rel=1e-6, abs=0)


def test_gate_spans_the_rare_regimes() -> None:
    # %Rare from 4.8 (DS10) to 21.1 (DS09); DS05 (3.5 %) is in the probe only,
    # too large for the fixture (ADR-0014 "Acceptance").
    pct = {float(c["paper_pct_rare"]) for c in CASES}
    assert min(pct) <= 4.8 and max(pct) >= 21.1 and len(SERA_CASES) >= 3
