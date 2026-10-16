import numpy as np
import pandas as pd
import pytest

from qafi_core import lown


def landscape(n_positions=20, seed=0):
    """A protein whose label is explained by the residue vector, so a fitted model can beat zero-shot."""
    rng = np.random.default_rng(seed)
    rows, vectors = [], []
    for position in range(1, n_positions + 1):
        vectors.append(rng.normal(size=8))
        for alt in lown.AMINO_ACIDS:
            rows.append(
                {
                    "position": position,
                    "mut_aa": alt,
                    "row": position - 1,
                    "weight": rng.normal(),
                    "noise": rng.normal(),
                }
            )
    df = pd.DataFrame(rows)
    embeddings = np.array(vectors, dtype=np.float32)
    signal = embeddings[df.row.values, 0] * 2 + df.weight
    df["label"] = signal + 0.2 * df.noise
    df["predicted"] = signal + 2.0 * df.noise  # a weak zero-shot prediction of the same thing
    return df.reset_index(drop=True), embeddings


def test_one_hot_marks_the_position_and_the_new_amino_acid():
    df = pd.DataFrame({"position": [3, 7], "mut_aa": ["A", "W"]})

    out = lown.one_hot(df, positions=[3, 7])

    assert out.shape == (2, 2 + 20)
    assert out[0, 0] == 1 and out[0, 2 + lown.AMINO_ACIDS.index("A")] == 1
    assert out[1, 1] == 1 and out[1, 2 + lown.AMINO_ACIDS.index("W")] == 1
    assert out.sum() == 4


def test_ridge_recovers_a_linear_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5))
    y = x @ np.array([2.0, -1.0, 0.0, 0.5, 0.0]) + 3

    predict = lown.fit_ridge(x, y, alpha=1e-6)

    assert np.corrcoef(predict(x), y)[0, 1] > 0.99


def test_more_measurements_rank_the_rest_better():
    df, embeddings = landscape()

    out = lown.curve(df, embeddings, budgets=(24, 96), draws=3)

    median = out.groupby(["model", "budget"]).spearman.median().unstack()
    # zero-shot fits nothing, so the budget cannot change it
    assert median.loc["zero-shot", 24] == pytest.approx(median.loc["zero-shot", 96], abs=0.05)
    for model in ("one-hot + ridge", "ESM-2 + ridge", "ESM-2 + zero-shot + ridge"):
        assert median.loc[model, 96] > median.loc[model, 24]
    assert median.loc["ESM-2 + zero-shot + ridge", 96] > median.loc["zero-shot", 96]


def test_a_campaign_finds_more_of_the_best_variants_than_picking_at_random():
    df, embeddings = landscape(n_positions=150)  # 3,000 variants: too many to stumble on the best ones

    guided = lown.campaign(df, embeddings, batch=30, rounds=4, strategy="greedy")
    random = lown.campaign(df, embeddings, batch=30, rounds=4, strategy="random")

    assert list(guided["round"]) == [1, 2, 3, 4]
    assert guided.measured.tolist() == [30, 60, 90, 120]
    assert guided["best found"].is_monotonic_increasing
    assert guided["top 1% found"].iloc[-1] > random["top 1% found"].iloc[-1]
