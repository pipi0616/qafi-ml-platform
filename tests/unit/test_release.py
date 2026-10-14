import numpy as np
import pandas as pd
import pytest

from qafi_core import release


def test_a_percentile_says_where_a_value_falls_in_the_training_distribution():
    table = release.quantiles(np.arange(1001))
    assert len(table) == release.N_QUANTILES
    assert release.percentile([0, 500, 1000, -5, 2000], table).tolist() == [0.0, 0.5, 1.0, 0.0, 1.0]
    assert release.quantiles([1.0, np.nan, 3.0])[[0, -1]].tolist() == [1.0, 3.0]  # missing values ignored


def test_the_function_score_averages_the_members_percentiles():
    train = pd.DataFrame(
        {"esm_delta": np.arange(100.0), "trees": np.arange(100.0) / 100, "stability": -np.arange(100.0)}
    )
    tables = release.distributions(train)
    new = pd.DataFrame({"esm_delta": [99.0, 0.0], "trees": [0.99, 0.0], "stability": [0.0, 0.0]})
    scored = release.scores(new, tables)
    assert scored["stability_score"].tolist() == [1.0, 1.0]  # stability 0 is the highest seen
    assert scored["function_score"].tolist() == pytest.approx([1.0, 1 / 3])


def _datasets(kind: str, n: int, quality: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n):
        label = rng.normal(size=60)
        good, bad = label + rng.normal(scale=quality, size=60), rng.normal(size=60)
        names = {"dataset_id": f"{kind}{d}", "data_kind": kind, "uniprot_id": f"P{d}"}
        rows.append(pd.DataFrame({**names, "label": label, "good": good, "bad": bad}))
    return pd.concat(rows, ignore_index=True)


def test_the_gate_passes_only_scores_that_clearly_beat_the_simpler_one():
    df = pd.concat([_datasets("domainome", 15, 0.3, 0), _datasets("mavedb", 15, 0.3, 1)], ignore_index=True)
    columns = {"stability_score": "good", "function_score": "good", "esm_delta": "bad", "trees": "bad"}
    gate = release.gate(release.test_scores(df, columns))
    assert gate["passed"].all() and len(gate) == 3
    columns["trees"] = "good"  # now the network ties the trees
    gate = release.gate(release.test_scores(df, columns))
    assert not gate.loc["stability beats the trees on stability assays", "passed"]


def test_published_predictors_are_oriented_and_compared_on_their_own_variants():
    df = _datasets("mavedb", 8, 0.5, 2).assign(
        uniprot_position=lambda d: d.groupby("dataset_id").cumcount() + 1
    )
    df["ref_aa"], df["alt_aa"] = "A", "C"
    published = df[["uniprot_id", "uniprot_position", "ref_aa", "alt_aa"]].rename(
        columns={"uniprot_position": "position"}
    )
    published["am_pathogenicity"] = -df["label"].to_numpy()  # higher = more pathogenic, as AlphaMissense
    published["qafi2"] = df["bad"].to_numpy()
    published["eve"] = np.nan
    for column in ("popeve", "cpt1"):
        published[column] = np.nan
    table = release.against_published(df, {"function": "good"}, published).set_index("published")
    assert set(table.index) == {
        "AlphaMissense",
        "QAFI2 (the paper)",
    }  # no shared variants with EVE and the rest
    assert table.loc["AlphaMissense", "published_median"] == pytest.approx(1.0)  # flipped: high = pathogenic
    assert table.loc["QAFI2 (the paper)", "ours_better"] == 1.0
