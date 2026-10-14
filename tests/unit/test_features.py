from pathlib import Path

import pandas as pd
import yaml
from lakehouse import language_model

from qafi_core import features

DBT = Path(__file__).parents[2] / "dbt"


def test_the_seed_files_are_exactly_what_the_code_produces():
    """The lookup tables in git must not drift from their source: regenerate and compare."""
    pd.testing.assert_frame_equal(pd.read_csv(DBT / "seeds" / "amino_acid.csv"), features.amino_acid_table())
    pd.testing.assert_frame_equal(pd.read_csv(DBT / "seeds" / "blosum62.csv"), features.blosum62_table())


def test_the_lookup_tables_hold_the_published_values():
    amino_acid = features.amino_acid_table().set_index("aa")
    assert amino_acid.loc["I"].tolist() == ["Ile", 4.5, 166.7, 0]  # most hydrophobic
    assert amino_acid.loc["R"].tolist() == ["Arg", -4.5, 173.4, 1]
    assert amino_acid.loc["D", "charge"] == -1
    blosum = features.blosum62_table().set_index(["ref_aa", "alt_aa"])["score"]
    assert len(blosum) == 400
    assert blosum["W", "W"] == 11 and blosum["L", "I"] == 2 and blosum["W", "D"] == -4
    assert all(blosum[a, b] == blosum[b, a] for a, b in blosum.index)


def test_sql_and_python_read_the_log_probabilities_in_the_same_order():
    """dbt picks an amino acid's log-probability by its place in a string; it must be ESM-2's order."""
    project = yaml.safe_load((DBT / "dbt_project.yml").read_text())
    assert project["vars"]["amino_acids"] == features.AMINO_ACIDS == language_model.AMINO_ACIDS


def test_every_feature_is_a_documented_column_of_the_feature_table():
    models = yaml.safe_load((DBT / "models" / "marts" / "features" / "_features.yml").read_text())["models"]
    table = next(m for m in models if m["name"] == "fct_variant_features")
    documented = {column["name"] for column in table["columns"]}
    assert set(features.FEATURES) <= documented
    assert len(features.FEATURES) == len(set(features.FEATURES)) == 16


def test_features_are_selected_by_group():
    assert features.select("all") == features.FEATURES
    assert features.select("language_model, structure") == [
        "esm_delta", "esm_wt_logprob", "esm_entropy",
        "plddt", "rsa", "contacts_8", "contacts_12", "hse_up", "hse_down",
    ]  # fmt: skip
