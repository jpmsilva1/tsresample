"""Gate G0: phi against R's recorded output (Blueprint/docs/REPLICATION.md §2).

Every expected value here is read from ``phi_oracle.json`` (R's recorded control
points) or from the paper's Table 1 -- never computed by our own code.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tsresample import _relevance

BLUEPRINT = Path(__file__).resolve().parents[1] / "Blueprint"
ORACLE = json.loads(
    (BLUEPRINT / "tests" / "fixtures" / "phi_oracle.json").read_text(encoding="utf-8")
)["datasets"]
# DS05 sits exactly on its +-0.08 fences and the CSV export lost the last bit that
# decides them (open item, REPLICATION.md §2). Excluded, never tolerated away.
EXCLUDED_FROM_A = {"DS05"}


def _target(ds: str) -> np.ndarray:
    path = next((BLUEPRINT / "replication" / "datasets").glob(f"{ds}_*.csv"))
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r["target"]) if r["target"] else np.nan for r in rows])


SPLITS = [
    pytest.param(ds, s, id=f"{ds}-it{s['iteration']}")
    for ds, d in ORACLE.items()
    if ds not in EXCLUDED_FROM_A
    for s in d.get("r_splits", [])
]


@pytest.mark.parametrize(("ds", "split"), SPLITS)
def test_a_control_points_match_r_per_split(ds: str, split: dict) -> None:
    a, b = split["train_slice"]
    cp = _relevance.control_points(_target(ds)[a:b])
    np.testing.assert_allclose(cp[:, 0], split["ctrl_x"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(cp[:, 1], split["ctrl_phi"], rtol=0, atol=1e-6)


def test_a_covers_every_non_excluded_recorded_split() -> None:
    # 18 NA-free datasets x 5 iterations, minus DS05's 5 (REPLICATION.md §2).
    assert len(SPLITS) == 85


NA_FREE = [ds for ds, d in ORACLE.items() if d.get("r_splits")]


def _pct_rare(ds: str) -> float:
    # create.data(ts, 10) target = series[9:] (ADR-0005); rare = phi >= 0.9.
    y = _target(ds)[9:]
    return 100.0 * float(
        np.mean(_relevance.phi(y, _relevance.control_points(y)) >= 0.9)
    )


def test_b_pct_rare_matches_paper_table_1() -> None:
    err = {ds: abs(_pct_rare(ds) - ORACLE[ds]["paper_pct_rare"]) for ds in NA_FREE}
    assert len(err) == 18
    mae = sum(err.values()) / len(err)
    worst = max(err, key=err.__getitem__)
    # Tolerances: Blueprint/ops/QUALITY_GATES.md §1.
    assert mae <= 0.25, f"MAE {mae:.3f} pp > 0.25; per dataset: {err}"
    assert err[worst] <= 2.5, f"{worst} off by {err[worst]:.2f} pp > 2.5"


@pytest.mark.parametrize(
    "split", ORACLE["DS10"]["r_splits"], ids=lambda s: f"it{s['iteration']}"
)
def test_c_one_sided_phi_is_zero_at_and_beyond_its_endpoint(split: dict) -> None:
    # R recorded phi = 0 at one endpoint in every DS10 split.
    zero_side = [i for i in (0, 2) if split["ctrl_phi"][i] == 0.0]
    assert zero_side, "fixture: DS10 split expected to be one-sided"
    a, b = split["train_slice"]
    cp = _relevance.control_points(_target("DS10")[a:b])
    for i in zero_side:
        x_end = split["ctrl_x"][i]
        assert cp[i, 1] == 0.0
        beyond = x_end + np.array([0.0, 1.0, 1e3]) * (-1 if i == 0 else 1)
        np.testing.assert_array_equal(_relevance.phi(beyond, cp), 0.0)
