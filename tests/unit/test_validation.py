import numpy as np
import pandas as pd
import pytest

from qafi_core import validation as v

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def _random(length: int, seed: int) -> str:
    return "".join(np.random.default_rng(seed).choice(AMINO_ACIDS, length))


def _mutate(sequence: str, share: float, seed: int) -> str:
    """The sequence with ``share`` of its residues replaced: a made-up relative."""
    rng = np.random.default_rng(seed)
    residues = list(sequence)
    for i in rng.choice(len(residues), int(share * len(residues)), replace=False):
        residues[i] = rng.choice(AMINO_ACIDS)
    return "".join(residues)


def test_relatives_are_recognised_and_strangers_are_not():
    a = _random(200, 1)
    assert v.related(a, _mutate(a, 0.3, 2))
    assert not v.related(a, _random(200, 3))
    assert not v.related(a[:60], _random(5000, 4))  # a short protein finds chance matches in a very long one


def test_groups_follow_chains_of_relatives():
    a = _random(200, 1)
    b = _mutate(a, 0.3, 2)
    c = _mutate(b, 0.3, 3)
    groups = v.similar_groups({"A": a, "B": b, "C": c, "D": _random(200, 4)})
    assert groups == {"A": "A", "B": "A", "C": "A", "D": "D"}


def test_the_split_keeps_groups_together_and_never_changes():
    groups = {f"P{i}": f"G{i // 2}" for i in range(2000)}  # 1000 groups of two relatives
    split = v.split_proteins(groups)
    assert split == v.split_proteins(groups)
    assert all(split[f"P{i}"] == split[f"P{i + 1}"] for i in range(0, 2000, 2))
    assert 0.17 < sum(side == "test" for side in split.values()) / 2000 < 0.23


def test_the_split_is_written_once_then_only_read(tmp_path, monkeypatch):
    path = tmp_path / "protein_split.csv"
    first = v.lock_split(path, {"A": _random(100, 1), "B": _random(100, 2)})
    monkeypatch.setattr(v, "similar_groups", lambda sequences: pytest.fail("the locked split was recomputed"))
    pd.testing.assert_frame_equal(v.lock_split(path, {"A": "changed since"}), first)


def test_representative_groups_do_not_follow_chains():
    """A and C are both half B, so a chain would merge them; the representative keeps them apart."""
    a, c = _random(100, 1), _random(100, 2)
    b = a[:50] + c[:50]
    groups = v.representative_groups({"A": a, "B": b, "C": c})
    assert groups["A"] != groups["C"]
    assert v.similar_groups({"A": a, "B": b, "C": c})["A"] == v.similar_groups({"A": a, "B": b, "C": c})["C"]


def test_extending_the_split_keeps_old_rows_and_relatives_together(tmp_path):
    path = tmp_path / "split.csv"
    base = {"A": _random(100, 1), "B": _random(100, 2)}
    locked = v.lock_split(path, base)
    a_side = locked.set_index("uniprot_id").loc["A", "split"]
    more = {**base, "A2": base["A"], "C": _random(100, 3)}
    table = v.extend_split(path, more).set_index("uniprot_id")
    assert (
        table.loc[["A", "B"], "split"].tolist()
        == locked.set_index("uniprot_id").loc[["A", "B"], "split"].tolist()
    )
    assert table.loc["A2", "split"] == a_side and table.loc["A2", "similarity_group"] == "A"
    assert set(table.index) == {"A", "A2", "B", "C"}
    again = v.extend_split(path, more)
    pd.testing.assert_frame_equal(again.set_index("uniprot_id"), table)


def test_folds_hold_out_whole_groups():
    groups = pd.Series(list("aabbccddee"))
    fold = v.group_folds(groups, n_folds=5)
    assert fold.groupby(groups).nunique().eq(1).all()
    assert fold.nunique() == 5


def test_scores_are_computed_within_each_protein():
    df = pd.DataFrame(
        {
            "protein": ["X"] * 20 + ["Y"] * 20,
            "label": list(range(20)) * 2,
            "predicted": list(range(20)) + list(range(20))[::-1],
        }
    )
    scores = v.per_protein(df, "label", "predicted").set_index("protein")
    assert scores.loc["X", "spearman"] == pytest.approx(1)
    assert scores.loc["Y", "spearman"] == pytest.approx(-1)


def test_a_random_split_flatters_a_model_that_memorises_positions():
    # Each position has its own effect, and one feature identifies the position. Seeing other variants of
    # the same position (random split) lets the model look the effect up; a new protein gives it nothing.
    rng = np.random.default_rng(0)
    rows = []
    for protein in range(30):
        for _position in range(20):
            effect, code = rng.normal(), rng.uniform()
            rows += [
                {
                    "protein": f"P{protein}",
                    "position_code": code,
                    "noise": rng.normal(),
                    "label": effect + 0.3 * rng.normal(),
                }
                for _ in range(5)
            ]
    scores = v.compare_splits(pd.DataFrame(rows), ["position_code", "noise"], "label", "protein")
    random, held_out = scores["median Spearman"]
    assert random > held_out + 0.3
