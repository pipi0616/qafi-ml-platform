import numpy as np
import pandas as pd
import pytest

from qafi_core import sites


def test_uniprot_feature_columns_become_positions():
    table = pd.DataFrame(
        {
            "Entry": ["P1", "P2", "P3"],
            "Active site": ['ACT_SITE 124; /note="Phosphocysteine intermediate"', None, ""],
            "Binding site": [
                'BINDING 12; /ligand="ATP"; BINDING 340; /ligand="Mg(2+)"',
                'BINDING 7; /ligand="heme"',
                None,
            ],
            "Site": [None, 'SITE 7; /note="Cleavage"', None],
        }
    )

    out = sites.known_sites(table)

    assert sorted(map(tuple, out[["uniprot_id", "position"]].to_numpy())) == [
        ("P1", 12),
        ("P1", 124),
        ("P1", 340),
        ("P2", 7),
    ]
    assert len(out) == 4  # P2's position 7 is annotated twice and counted once


def mapped(seed=0):
    """Two proteins where a few positions are made costly in one energy or the other."""
    rng = np.random.default_rng(seed)
    rows = []
    for protein, special in (("P1", {5: "fold", 9: "func"}), ("P2", {3: "func"})):
        for position in range(1, 21):
            for alt in "ACDEFG":
                kind = special.get(position)
                rows.append(
                    {
                        "uniprot_id": protein,
                        "position": position,
                        "mut_aa": alt,
                        "ddg_fold": (-3.0 if kind == "fold" else 0.0) + 0.05 * rng.normal(),
                        "ddg_func": (-3.0 if kind == "func" else 0.0) + 0.05 * rng.normal(),
                    }
                )
    return pd.DataFrame(rows)


def test_positions_are_labelled_by_which_energy_they_cost():
    out = sites.classify(mapped(), quantile=0.1)

    label = out.set_index(["uniprot_id", "position"]).position_is
    assert label[("P1", 5)] == sites.STRUCTURAL
    assert label[("P1", 9)] == sites.FUNCTIONAL
    assert label[("P2", 3)] == sites.FUNCTIONAL
    assert label[("P1", 1)] in (sites.TOLERANT, sites.MIXED)
    assert len(out) == 40  # one row per position, not per variant


def test_functional_positions_are_enriched_for_annotated_sites():
    positions = sites.classify(mapped(), quantile=0.1)
    annotated = pd.DataFrame({"uniprot_id": ["P1", "P2"], "position": [9, 3], "annotation": "active"})

    out = sites.enrichment(positions, annotated).set_index("position_is")

    # both annotated sites land in the functional class, and no other class holds any
    assert out.loc[sites.FUNCTIONAL, "annotated"] == 2
    assert out.drop(index=sites.FUNCTIONAL).annotated.sum() == 0
    assert out.loc[sites.FUNCTIONAL, "p"] < 0.01


def test_a_pocket_the_model_was_not_told_about_can_be_ranked():
    positions = sites.classify(mapped(), quantile=0.1)

    out = sites.pocket_rank(positions, "P1", pocket=[9])

    assert out["pocket residues"] == 1
    assert out["median percentile"] < out["median percentile elsewhere"]
    assert out["p"] < 0.1
    with pytest.raises(ValueError, match="none of the pocket positions"):
        sites.pocket_rank(positions, "P1", pocket=[999])
