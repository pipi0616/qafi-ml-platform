import numpy as np
import pandas as pd
import pytest

from qafi_core import baseline
from qafi_core.features import FEATURES

FAST = {"n_estimators": 30, "min_child_weight": 1}


def _variants(proteins: int = 12, per_dataset: int = 60, seed: int = 0) -> pd.DataFrame:
    """A made-up feature table: the label follows esm_delta, plus noise. Two proteins share each group."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(proteins):
        kind = ["domainome", "mavedb", "proteingym"][p % 3]
        for d in range(1 + p % 2):  # some proteins have two datasets
            size = per_dataset * (1 + 4 * d)  # of very different sizes
            block = pd.DataFrame(rng.normal(size=(size, len(FEATURES))), columns=FEATURES)
            block["label"] = block["esm_delta"] + 0.3 * rng.normal(size=size)
            block["dataset_id"] = f"d{p}_{d}"
            block["data_kind"] = kind
            block["uniprot_id"] = f"P{p}"
            block["similarity_group"] = f"G{p // 2}"
            block["uniprot_position"] = np.arange(size) + 1
            block["ref_aa"], block["alt_aa"] = "A", "C"
            rows.append(block)
    return pd.concat(rows, ignore_index=True)


def test_every_dataset_weighs_the_same():
    df = _variants()
    weight = pd.Series(baseline.weights(df))
    per_dataset = weight.groupby(df["dataset_id"]).sum()
    assert per_dataset.max() == pytest.approx(per_dataset.min())
    assert weight.mean() == pytest.approx(1)


def test_cross_validation_predicts_every_variant_without_its_relatives():
    df = _variants()
    cv = baseline.cross_validate(df, params=FAST, n_folds=3)
    assert cv["predicted"].notna().all() and len(cv) == len(df)
    assert (cv.groupby("similarity_group")["fold"].nunique() == 1).all()  # relatives share a fold
    scores = baseline.dataset_scores(cv)
    assert len(scores) == df["dataset_id"].nunique()
    assert scores["spearman"].median() > 0.8  # the signal is learnable


def test_training_can_be_limited_to_kinds_of_data_while_all_are_predicted():
    df = _variants()
    df.loc[df["data_kind"] == "mavedb", "label"] *= -1  # this kind follows the opposite rule
    without = baseline.cross_validate(df, params=FAST, n_folds=3, train_on=["domainome", "proteingym"])
    assert without["predicted"].notna().all()
    scores = baseline.dataset_scores(without).groupby("data_kind")["spearman"].median()
    assert scores["mavedb"] < -0.7 < 0.7 < scores["domainome"]  # it never saw that the rule differs


def test_the_summary_gives_the_median_overall_and_per_kind():
    scores = pd.DataFrame(
        {
            "dataset_id": list("abcde"),
            "data_kind": ["domainome"] * 3 + ["mavedb"] * 2,
            "spearman": [0.1, 0.5, 0.6, 0.2, 0.4],
        }
    )
    assert baseline.summarise(scores) == {
        "datasets": 5.0,
        "spearman_median": 0.4,
        "spearman_median_domainome": 0.5,
        "spearman_median_mavedb": 0.3,
    }


def test_models_are_compared_dataset_by_dataset():
    reference = pd.DataFrame(
        {
            "dataset_id": [f"d{i}" for i in range(12)],
            "data_kind": "mavedb",
            "spearman": np.linspace(0.2, 0.6, 12),
        }
    )
    better = reference.assign(spearman=reference["spearman"] + np.linspace(0.05, 0.1, 12))
    result = baseline.compare(better, reference)
    assert result["better_mavedb"] == 1.0 and result["difference_mavedb"] == pytest.approx(0.075)
    assert result["p_value_mavedb"] < 0.01
    same = baseline.compare(reference, reference)
    assert same["better_mavedb"] == 0.0 and same["p_value_mavedb"] == 1.0


def test_the_paper_comparison_scores_both_models_on_the_shared_variants():
    rng = np.random.default_rng(1)
    n = 80
    paper = pd.DataFrame(
        {
            "protein": "TP53", "uniprot": "P04637", "pos": np.arange(n) + 1, "first": "A", "second": "C",
            "score_log_normalized": rng.normal(size=n),
        }
    )  # fmt: skip
    paper["qafi2"] = paper["score_log_normalized"] + rng.normal(size=n)
    ours = pd.DataFrame(
        {
            "uniprot_id": "P04637", "uniprot_position": np.arange(n + 20) + 1, "ref_aa": "A", "alt_aa": "C",
            "predicted": np.r_[paper["score_log_normalized"].to_numpy(), np.zeros(20)], "esm_delta": 0.0,
        }
    )  # fmt: skip
    ours.loc[0, "esm_delta"] = 1.0
    table = baseline.compare_with_paper(
        pd.concat([ours, ours]), paper
    )  # a variant in two datasets counts once
    assert table.loc[0, "variants"] == n and table.loc[0, "new_baseline"] == 1.0
    assert 0 < table.loc[0, "paper_qafi2"] < 1


def test_importance_ranks_the_feature_that_carries_the_signal_first():
    table = baseline.importance(baseline.fit(_variants(), params=FAST))
    assert table.loc[0, "feature"] == "esm_delta" and table["share"].sum() == pytest.approx(1, abs=0.01)
