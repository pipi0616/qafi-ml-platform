import numpy as np
import pandas as pd
import pytest

from qafi_platform.metrics import auroc, median_per_protein_pearson, per_protein_pearson


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for protein, slope in [("A", 1.0), ("B", -1.0), ("C", 0.0)]:
        x = rng.normal(size=30)
        noise = rng.normal(scale=0.1, size=30)
        rows.append(pd.DataFrame({"uniprot": protein, "target": x, "pred": slope * x + noise}))
    return pd.concat(rows, ignore_index=True)


def test_per_protein_pearson_signs():
    r = per_protein_pearson(_frame(), "target", "pred")
    assert r["A"] > 0.95
    assert r["B"] < -0.95
    assert abs(r["C"]) < 0.5


def test_small_proteins_are_skipped():
    df = _frame()
    df.loc[df.uniprot == "C", "pred"] = np.nan
    df.loc[df[df.uniprot == "C"].index[:3], "pred"] = 1.0  # only 3 complete rows for C
    r = per_protein_pearson(df, "target", "pred", min_variants=5)
    assert set(r.index) == {"A", "B"}


def test_median_is_middle_protein():
    assert median_per_protein_pearson(_frame(), "target", "pred") == pytest.approx(
        per_protein_pearson(_frame(), "target", "pred").median()
    )


def test_auroc_is_orientation_free():
    y = pd.Series([0, 0, 1, 1])
    assert auroc(y, pd.Series([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(y, pd.Series([0.9, 0.8, 0.2, 0.1])) == 1.0  # inverted scores give the same AUC


def test_auroc_ignores_missing_scores():
    y = pd.Series([0, 0, 1, 1, 1])
    s = pd.Series([0.1, 0.4, 0.6, np.nan, 0.9])
    assert auroc(y, s) == 1.0
