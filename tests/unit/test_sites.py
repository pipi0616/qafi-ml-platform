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


def test_measured_and_predicted_maps_use_the_same_rule():
    fold = np.array([-3.0, -2.5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    func = np.array([-3.0, 0.0, -2.5, 0.1, 0.2, 0.3, 0.4, 0.5])
    labels = sites.position_classes(fold, func)
    assert list(labels[:3]) == [sites.MIXED, sites.STRUCTURAL, sites.FUNCTIONAL]
    assert set(labels[3:]) == {sites.TOLERANT}


def test_agreement_is_one_for_a_perfect_map_and_near_chance_for_a_shuffled_one():
    rng = np.random.default_rng(0)
    measured = rng.choice(
        [sites.STRUCTURAL, sites.FUNCTIONAL, sites.MIXED, sites.TOLERANT], 400, p=[0.15, 0.15, 0.1, 0.6]
    )
    perfect = sites.agreement(measured, measured, permutations=200)
    assert perfect["kappa"] == pytest.approx(1.0)
    assert perfect["p"] < 0.01
    assert all(found == pytest.approx(1.0) for found, _ in perfect["recall"].values())
    shuffled = sites.agreement(measured, rng.permutation(measured), permutations=200)
    assert abs(shuffled["kappa"]) < 0.1
    assert shuffled["p"] > 0.05


def test_what_abundance_explains_is_taken_out_of_activity():
    rng = np.random.default_rng(0)
    abundance = rng.uniform(0, 1, 500)
    function = rng.normal(0, 0.1, 500)
    activity = abundance**2 + function  # activity follows abundance, bent, plus a part of its own
    left = sites.beyond_abundance(abundance, activity)
    assert abs(np.corrcoef(left, abundance)[0, 1]) < 0.1
    assert np.corrcoef(left, function)[0, 1] > 0.8


def test_a_chain_is_numbered_along_the_full_protein():
    protein = "MKTAYIAKQRQISFVKSHFSRQ"
    assert sites.receptor_numbering("AYIAK", protein) == {1: 4, 2: 5, 3: 6, 4: 7, 5: 8}
    # an engineered mutation (I -> V) is aligned through, but the changed position is not mapped
    shifted = sites.receptor_numbering("TAYVAKQRQISF", protein)
    assert shifted[1] == 3 and shifted[12] == 14
    assert 4 not in shifted


def test_biolip_lines_become_protein_positions_and_the_best_structure_is_the_pocket():
    sequences = {"P1": "MKTAYIAKQRQISFVKSHFSRQ"}
    table = pd.DataFrame(
        {
            1: ["1abc", "2xyz", "3zzz"],
            3: [2.5, 1.8, 2.0],
            5: ["DRG", "DRG", "ZN"],
            9: ["Y2 I3", "A1 Y2", "K5"],
            18: ["P1", "P1", "P1"],
            21: ["AYIAK", "AYIAK", "AYIAK"],
        }
    )
    found = sites.biolip_sites(table, sequences)
    assert sorted(found[found.pdb == "1abc"].position) == [5, 6]
    assert found[found.ligand == "ZN"].metal.all() and not found[found.ligand == "DRG"].metal.any()
    pdb, pocket = sites.drug_pocket(found, "P1", {"DRG"})
    assert pdb == "2xyz" and pocket == [4, 5]
    with pytest.raises(ValueError):
        sites.drug_pocket(found, "P1", {"NOPE"})
