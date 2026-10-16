import pandas as pd
import pytest

from qafi_core import dms

WT = "MKV*"


def write_dimsum_tables(folder):
    """DiMSum's layout: a tab-separated table per kind of variant, NA for missing, doubles with two fitness
    columns."""
    pd.DataFrame({"aa_seq": [WT], "Nham_aa": [0], "WT": [True], "fitness": [0.0], "sigma": [0.01]}).to_csv(
        folder / "fitness_wildtype.txt", sep="\t", index=False, na_rep="NA"
    )
    pd.DataFrame(
        {
            "aa_seq": ["AKV*", "MAV*", "MKA*"],
            "Nham_aa": [1, 1, 1],
            "WT": [None, None, None],
            "fitness": [-0.5, -1.0, 0.1],
            "sigma": [0.1, 0.2, 0.1],
        }
    ).to_csv(folder / "fitness_singles.txt", sep="\t", index=False, na_rep="NA")
    pd.DataFrame(
        {
            "aa_seq": ["AAV*"],
            "Nham_aa": [2],
            "WT": [None],
            "fitness_uncorr": [-1.4],
            "sigma_uncorr": [0.3],
            "fitness_cond": [-1.2],
            "sigma_cond": [0.2],
        }
    ).to_csv(folder / "fitness_doubles.txt", sep="\t", index=False, na_rep="NA")


def test_every_kind_of_variant_is_read_into_one_table(tmp_path):
    write_dimsum_tables(tmp_path)

    ours = dms.read_fitness(tmp_path)

    assert len(ours) == 5
    assert ours.loc[ours.WT, "aa_seq"].tolist() == [WT]
    double = ours.set_index("aa_seq").loc["AAV*"]
    assert double.fitness == -1.4  # the uncorrected value, as in the deposited table
    assert double.Nham_aa == 2


def test_an_empty_folder_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        dms.read_fitness(tmp_path)


def test_ours_and_the_published_table_are_lined_up_by_sequence(tmp_path):
    write_dimsum_tables(tmp_path)
    ours = dms.read_fitness(tmp_path)
    published = pd.DataFrame(
        {
            "aa_seq": [WT, "AKV*", "MAV*", "MKA*", "WWW*"],
            "Nham_aa": [0, 1, 1, 1, 3],
            "WT": [True, False, False, False, False],
            "fitness": [0.0, -0.5, -1.0, 0.1, -2.0],
            "sigma": [0.01, 0.1, 0.2, 0.1, 0.5],
        }
    )

    joined, summary = dms.compare(ours, published)

    assert summary["shared"] == 4
    assert summary["only_ours"] == 1  # the double the published table does not have
    assert summary["only_published"] == 1
    assert summary["pearson"] == pytest.approx(1.0)
    assert summary["max_abs_difference"] == pytest.approx(0.0)
    assert summary["shared_1_mutations"] == 3
    assert set(joined.columns) >= {"fitness_ours", "fitness_published"}
