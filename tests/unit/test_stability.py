import pickle

import numpy as np
import pandas as pd
import pytest

from qafi_core import stability


def write_splits(path, splits):
    """A split file shaped like the published one: lists for train, numpy arrays for val and test."""
    with open(path, "wb") as f:
        pickle.dump(
            {
                "train": list(splits["train"]),
                "val": np.array(splits["val"]),
                "test": np.array(splits["test"]),
                "cv_train_0": list(splits["train"]),  # extra keys the real file has and we ignore
            },
            f,
            protocol=4,
        )


def test_splits_are_read_without_running_what_the_file_says(tmp_path):
    path = tmp_path / "splits.pkl"
    write_splits(path, {"train": ["a.pdb"], "val": ["b.pdb"], "test": ["c.pdb", "d.pdb"]})

    splits = stability.load_splits(path)

    assert splits == {"train": ["a.pdb"], "val": ["b.pdb"], "test": ["c.pdb", "d.pdb"]}


def test_a_split_file_that_asks_for_anything_else_is_refused(tmp_path):
    path = tmp_path / "evil.pkl"
    with open(path, "wb") as f:
        pickle.dump({"train": pd.Series(["a.pdb"])}, f, protocol=4)  # pandas is not on the allow-list

    with pytest.raises(pickle.UnpicklingError, match="refused to import pandas"):
        stability.load_splits(path)


SEQ = "MKVLA"


def megascale_csv(tmp_path, rows):
    path = tmp_path / "mega.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def row(name, mut_type, ddg, seq=SEQ):
    return {"WT_name": name, "mut_type": mut_type, "ddG_ML": ddg, "dG_ML": "2.0", "aa_seq": seq}


def test_the_three_published_filters_are_applied(tmp_path):
    path = megascale_csv(
        tmp_path,
        [
            row("a.pdb", "wt", "0.0"),
            row("a.pdb", "K2A", "0.5"),
            row("a.pdb", "V3G", "-"),  # no reliable free energy
            row("a.pdb", "V3ins", "0.1"),  # insertion
            row("a.pdb", "L4del", "0.1"),  # deletion
            row("a.pdb", "K2A:V3G", "0.1"),  # two mutations at once
            row("b.pdb", "wt", "0.0"),
            row("b.pdb", "L4A", "-1.5"),
            row("other.pdb", "wt", "0.0"),  # a perturbed background: not in the split
            row("other.pdb", "K2A", "0.9"),
        ],
    )

    df = stability.megascale(path, {"train": ["a.pdb"], "val": [], "test": ["b.pdb"]})

    assert list(df.mut_type) == ["K2A", "L4A"]
    assert list(df.split) == ["train", "test"]
    assert list(df.position) == [2, 4]
    assert list(df.wt_aa) == ["K", "L"]
    assert list(df.mut_aa) == ["A", "A"]
    assert df.ddg.tolist() == [0.5, -1.5]
    assert set(df.seq) == {SEQ}


def test_a_mutation_the_sequence_does_not_have_is_an_error(tmp_path):
    path = megascale_csv(tmp_path, [row("a.pdb", "wt", "0.0"), row("a.pdb", "W2A", "0.5")])

    with pytest.raises(ValueError, match="amino acid the sequence does not have"):
        stability.megascale(path, {"train": ["a.pdb"], "val": [], "test": []})


def test_the_sign_convention_of_the_published_comparison_is_recorded():
    # ThermoMPNN trains on the negated column, so its "stabilising" is our negative. Correlations keep
    # their magnitude, errors in kcal/mol do not: the chapter has to say which sign it reports.
    assert stability.THERMOMPNN_SIGN == -1
    assert stability.PUBLISHED_COUNTS["test_mutations"] == 28_172


def small_dataset(n_domains=4, length=12, seed=0):
    """A tiny made-up dataset shaped exactly like the real one."""
    rng = np.random.default_rng(seed)
    aa = list(stability.AMINO_ACIDS)
    domains = [f"d{i}.pdb" for i in range(n_domains)]
    seqs = {d: "".join(rng.choice(aa, length)) for d in domains}
    rows = []
    for d in domains:
        for pos in range(1, length + 1):
            for alt in rng.choice(aa, 6, replace=False):
                rows.append(
                    {
                        "domain": d,
                        "split": "test" if d == domains[-1] else "train",
                        "position": pos,
                        "wt_aa": seqs[d][pos - 1],
                        "mut_aa": str(alt),
                        "ddg": float(rng.normal()),
                        "seq": seqs[d],
                    }
                )
    mutations = pd.DataFrame(rows)
    index = pd.DataFrame([{"domain": d, "position": p} for d in domains for p in range(1, length + 1)])
    structure = index.assign(
        aa=[seqs[r.domain][r.position - 1] for r in index.itertuples()],
        rsa=rng.random(len(index)),
        plddt=rng.uniform(50, 100, len(index)),
        contacts_8=rng.integers(0, 20, len(index)),
        contacts_12=rng.integers(0, 40, len(index)),
        hse_up=rng.integers(0, 30, len(index)),
        hse_down=rng.integers(0, 30, len(index)),
    )
    logprobs = rng.normal(size=(len(index), 20)).astype(np.float32)
    embeddings = rng.normal(size=(len(index), 16)).astype(np.float16)
    return mutations, structure, index, logprobs, embeddings


def test_features_point_at_the_right_residue():
    mutations, structure, index, logprobs, _ = small_dataset()

    df = stability.add_features(mutations, structure, index, logprobs)

    first = df.iloc[0]
    row = index.index[(index.domain == first.domain) & (index.position == first.position)][0]
    assert first.row == row
    assert first.esm_wt == pytest.approx(logprobs[row, stability.AA_INDEX[first.wt_aa]])
    expected = logprobs[row, stability.AA_INDEX[first.mut_aa]] - first.esm_wt
    assert first.esm_delta == pytest.approx(expected)
    assert first.rsalor == pytest.approx((1 - first.rsa) * first.esm_delta)
    assert not df[stability.FEATURES].isna().any(axis=None)


def test_a_structure_that_disagrees_with_the_mutation_is_caught():
    mutations, structure, index, logprobs, _ = small_dataset()
    structure.loc[0, "aa"] = "W" if structure.loc[0, "aa"] != "W" else "A"

    with pytest.raises(ValueError, match="differs from the mutation's wild type"):
        stability.add_features(mutations, structure, index, logprobs)


def test_scoring_is_per_domain_as_well_as_pooled():
    mutations, structure, index, logprobs, _ = small_dataset()
    df = stability.add_features(mutations, structure, index, logprobs)
    df["perfect"] = df.ddg

    result = stability.score(df, "perfect")

    assert result["spearman"] == pytest.approx(1.0)
    assert result["median_spearman"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["domains"] == df.domain.nunique()


def test_the_network_learns_a_signal_it_can_see():
    mutations, structure, index, logprobs, embeddings = small_dataset(n_domains=6, length=20)
    df = stability.add_features(mutations, structure, index, logprobs)
    # a target the features really explain: burial plus the language model's opinion
    df["ddg"] = 3 * df.buried - df.esm_delta
    train, test = df[df.split == "train"], df[df.split == "test"].copy()

    settings = {"epochs": 150, "batch": 128, "lr": 1e-3, "dropout": 0.0}
    model, scaling = stability.fit(train, embeddings, settings=settings)
    test["predicted"] = stability.predict(model, scaling, test, embeddings)

    assert stability.score(test, "predicted")["pearson"] > 0.8


def test_doubles_splits_each_pair_and_flags_synonymous_components(tmp_path):
    """A double is read into its two components; T9Q:N23N is a single wearing a double's name."""
    splits_path = tmp_path / "splits.pkl"
    write_splits(splits_path, {"train": ["train.pdb"], "val": [], "test": ["test.pdb"]})
    csv = tmp_path / "mega.csv"
    pd.DataFrame(
        {
            "WT_name": ["test.pdb"] * 4 + ["train.pdb"],
            "mut_type": ["T9Q:N23E", "T9Q:N23N", "A1_insG:K2L", "K5A", "T9Q:N23E"],
            "ddG_ML": ["-1.5", "-0.4", "0.2", "-2.0", "-1.1"],
        }
    ).to_csv(csv, index=False)

    got = stability.doubles(csv, stability.load_splits(splits_path))

    # the single, the insertion and the training domain's row are all gone
    assert list(got.mut_type) == ["T9Q:N23E", "T9Q:N23N"]
    assert list(got.wt0) == ["T", "T"] and list(got.pos0) == [9, 9] and list(got.mut0) == ["Q", "Q"]
    assert list(got.mut1) == ["E", "N"]
    assert list(got.synonymous) == [False, True]
    assert got.ddg.tolist() == [-1.5, -0.4]


def test_components_lays_out_two_rows_per_double_that_regroup_by_pair():
    pairs = pd.DataFrame(
        {
            "domain": ["d.pdb", "d.pdb"],
            "wt0": ["T", "A"],
            "pos0": [9, 3],
            "mut0": ["Q", "G"],
            "wt1": ["N", "L"],
            "pos1": [23, 40],
            "mut1": ["E", "M"],
        }
    )

    got = stability.components(pairs)

    assert len(got) == 4
    assert sorted(got.pair) == [0, 0, 1, 1]
    assert "ddg" in got, "the tensor builder reads this column even when predicting"
    first = got[got.pair == 0].sort_values("position")
    assert list(first.position) == [9, 23] and list(first.mut_aa) == ["Q", "E"]
