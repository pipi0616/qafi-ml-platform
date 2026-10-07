"""Evaluation metrics used by the benchmarks, the golden tests and the training gate.

Both functions follow the QAFI paper's conventions so that the published numbers
reproduce exactly:
- per-protein Pearson is computed on rows where target and prediction are both present,
  proteins with fewer than ``min_variants`` such rows are skipped, and the summary is the
  median across proteins;
- AUROC is orientation-free: a predictor whose score decreases with pathogenicity is
  reported as ``1 - auc``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def per_protein_pearson(
    df: pd.DataFrame,
    target: str,
    prediction: str,
    protein_col: str = "uniprot",
    min_variants: int = 5,
) -> pd.Series:
    """Pearson r between ``target`` and ``prediction`` for every protein.

    Returns a Series indexed by protein. Proteins with fewer than ``min_variants``
    complete rows are left out.
    """
    valid = df[[protein_col, target, prediction]].dropna()
    counts = valid.groupby(protein_col).size()
    keep = counts[counts >= min_variants].index
    valid = valid[valid[protein_col].isin(keep)]
    r = valid.groupby(protein_col).apply(
        lambda g: float(np.corrcoef(g[target], g[prediction])[0, 1]), include_groups=False
    )
    return r.rename("pearson_r")


def median_per_protein_pearson(df: pd.DataFrame, target: str, prediction: str, **kwargs) -> float:
    """Median of :func:`per_protein_pearson`: the paper's headline benchmark statistic."""
    return float(per_protein_pearson(df, target, prediction, **kwargs).median())


def auroc(y_true: pd.Series, score: pd.Series) -> float:
    """AUROC on rows where both label and score are present, orientation-free."""
    valid = pd.DataFrame({"y": y_true, "s": score}).dropna()
    auc = float(roc_auc_score(valid["y"].astype(int), valid["s"]))
    return max(auc, 1.0 - auc)
