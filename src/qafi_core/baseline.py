"""The baseline model (chapter 07f): gradient-boosted trees on the feature table.

It sets the bar the deep-learning models of chapter 08 have to clear, and it is tested the way chapter 07a
fixed: proteins held out whole, related proteins kept together. A dataset's score is the Spearman
correlation of its labels with the predictions; the summary is the median over datasets, overall and for
each kind of data, as in ProteinGym.

Nothing here knows about MLflow or Databricks: the job notebook ``train_baseline`` calls these functions
and records what they return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from qafi_core import validation
from qafi_core.features import FEATURES

KEY = ["dataset_id", "uniprot_id", "uniprot_position", "alt_aa"]
PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 20,  # a leaf needs this much weight: no rules built on a handful of variants
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "random_state": 0,
}


# The runs of chapter 07f: the baseline, and the same model with something taken away, to see what each
# family of features and each kind of data contributes. ``features`` names feature groups, ``train_on``
# kinds of data; left out = everything.
RUNS = {
    "baseline": {},
    "only_language_model": {"features": "language_model"},
    "only_structure": {"features": "structure"},
    "only_substitution": {"features": "substitution"},
    "without_neighbourhood": {"features": "substitution,language_model,structure"},
    "without_domainome": {"train_on": "mavedb,proteingym"},
    "only_domainome": {"train_on": "domainome"},
}


def weights(df: pd.DataFrame) -> np.ndarray:
    """A weight per variant so that every dataset counts the same, however many variants it measured.

    Without it the few assays with tens of thousands of variants would decide what the model learns.
    """
    size = df.groupby("dataset_id")["label"].transform("size").to_numpy()
    return len(df) / df["dataset_id"].nunique() / size  # the mean weight is 1


def fit(df: pd.DataFrame, features: list[str] = FEATURES, params: dict | None = None):
    """A gradient-boosted tree model trained on every row of ``df``."""
    import xgboost

    model = xgboost.XGBRegressor(**{**PARAMS, **(params or {})})
    return model.fit(df[features], df["label"], sample_weight=weights(df))


def cross_validate(
    df: pd.DataFrame,
    features: list[str] = FEATURES,
    params: dict | None = None,
    n_folds: int = 5,
    train_on: list[str] | None = None,
) -> pd.DataFrame:
    """``df`` with two columns added: ``fold``, and ``predicted`` by a model that never saw that fold.

    Folds follow ``similarity_group``, so a protein and its relatives are predicted by a model trained
    without any of them. ``train_on`` limits TRAINING to those kinds of data (``data_kind``); every variant
    is still predicted, which shows what a kind of data adds: train with and without it, score the rest.
    """
    df = df.reset_index(drop=True)
    fold = validation.group_folds(df["similarity_group"], n_folds)
    usable = df["data_kind"].isin(train_on) if train_on else pd.Series(True, index=df.index)
    predicted = np.full(len(df), np.nan)
    for k in range(n_folds):
        model = fit(df[(fold != k) & usable], features, params)
        predicted[fold == k] = model.predict(df.loc[fold == k, features])
    return df.assign(fold=fold, predicted=predicted)


def dataset_scores(df: pd.DataFrame, prediction: str = "predicted") -> pd.DataFrame:
    """Spearman correlation of label and ``prediction`` within each dataset (at least 10 variants)."""
    rows = []
    for (dataset, kind), g in df.groupby(["dataset_id", "data_kind"]):
        if len(g) < 10 or g["label"].nunique() < 2 or g[prediction].nunique() < 2:
            continue
        rows.append(
            {
                "dataset_id": dataset,
                "data_kind": kind,
                "uniprot_id": g["uniprot_id"].iloc[0],
                "variants": len(g),
                "spearman": stats.spearmanr(g["label"], g[prediction])[0],
            }
        )
    return pd.DataFrame(rows)


def summarise(scores: pd.DataFrame) -> dict[str, float]:
    """The numbers to compare models by: median Spearman over datasets, overall and per kind of data."""
    summary = {"datasets": float(len(scores)), "spearman_median": scores["spearman"].median()}
    for kind, g in scores.groupby("data_kind"):
        summary[f"spearman_median_{kind}"] = g["spearman"].median()
    return {name: round(float(value), 4) for name, value in summary.items()}


def compare(scores: pd.DataFrame, reference: pd.DataFrame) -> dict[str, float]:
    """Is one model better than another? Paired, dataset by dataset, for each kind of data.

    Two medians can differ by chance. So each dataset is scored by both models and the differences are
    tested (Wilcoxon signed-rank): ``better`` is the share of datasets where ``scores`` beats ``reference``,
    ``difference`` the mean gain in Spearman, ``p_value`` how likely such gains are if the two were equal.
    """
    both = scores.merge(reference, on="dataset_id", suffixes=("", "_reference"))
    result = {}
    for kind, g in both.groupby("data_kind"):
        gain = g["spearman"] - g["spearman_reference"]
        result[f"better_{kind}"] = float((gain > 0).mean())
        result[f"difference_{kind}"] = float(gain.mean())
        result[f"p_value_{kind}"] = float(stats.wilcoxon(gain).pvalue) if (gain != 0).any() else 1.0
    return {name: round(value, 4) for name, value in result.items()}


def rank_average(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Several predictions combined into one: the average of their ranks within each dataset.

    Predictions live on different scales (a log-probability, a label, a standardised label), so they are
    first turned into ranks, 0 to 1, per dataset. Models that err in different ways partly cancel out.
    """
    return sum(df.groupby("dataset_id")[column].rank(pct=True) for column in columns) / len(columns)


def compare_with_paper(predictions: pd.DataFrame, paper: pd.DataFrame, minimum: int = 50) -> pd.DataFrame:
    """Our held-out predictions next to the QAFI paper's, on the variants both have, per protein.

    ``paper`` is the paper's leave-one-protein-out output (uniprot, pos, first, second,
    score_log_normalized, qafi2). Both predictions were made without the protein in training, and both
    are scored against the paper's own label, so only the model differs.
    """
    ours = predictions.groupby(["uniprot_id", "uniprot_position", "ref_aa", "alt_aa"], as_index=False).agg(
        predicted=("predicted", "first"), esm_delta=("esm_delta", "first")
    )
    shared = paper.merge(
        ours,
        left_on=["uniprot", "pos", "first", "second"],
        right_on=["uniprot_id", "uniprot_position", "ref_aa", "alt_aa"],
    )
    rows = []
    for protein, g in shared.groupby("protein"):
        if len(g) < minimum:
            continue
        label = g["score_log_normalized"]
        rows.append(
            {
                "protein": protein,
                "variants": len(g),
                "paper_qafi2": stats.spearmanr(g["qafi2"], label)[0],
                "new_baseline": stats.spearmanr(g["predicted"], label)[0],
                "esm2_zero_shot": stats.spearmanr(g["esm_delta"], label)[0],
            }
        )
    return pd.DataFrame(rows).sort_values("paper_qafi2", ascending=False, ignore_index=True).round(3)


def importance(model, features: list[str] = FEATURES) -> pd.DataFrame:
    """How much each feature improved the trees' fit (gain), as a share of the total, largest first."""
    gain = model.get_booster().get_score(importance_type="total_gain")
    table = pd.DataFrame({"feature": features, "gain": [gain.get(name, 0.0) for name in features]})
    table["share"] = (table["gain"] / table["gain"].sum()).round(4)
    return table.sort_values("share", ascending=False, ignore_index=True)[["feature", "share"]]
