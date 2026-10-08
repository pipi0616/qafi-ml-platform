import pandas as pd
import pytest
from lakehouse import mavedb_variants as mv

UNIPROT = "MKVLAGGTTTTTTTTTTRWQ"  # the made-up protein of notebook 05's unit test


@pytest.mark.parametrize(
    ("hgvs", "expected"),
    [
        ("p.Leu12Pro", ("missense", 12, "L", "P")),
        ("p.Lys440Ter", ("nonsense", 440, "K", "*")),  # "Ter" looks like an amino acid code
        ("p.Lys440*", ("nonsense", 440, "K", "*")),
        ("p.Leu12=", ("synonymous", 12, "L", "L")),
        ("p.=", ("synonymous", None, None, None)),  # all synonymous variants together
        ("_sy", ("synonymous", None, None, None)),
        ("ENSP00000494040.1:p.Gly439Asp", ("missense", 439, "G", "D")),
        ("p.[Leu12Pro;Ala13Gly]", ("other", None, None, None)),
        ("p.Leu12del", ("other", None, None, None)),
        ("NA", ("other", None, None, None)),
        (None, ("other", None, None, None)),
    ],
)
def test_protein_hgvs_is_classified(hgvs, expected):
    assert mv.classify(hgvs) == expected


def test_dna_targets_are_translated_up_to_the_stop_codon():
    assert mv.target_protein("ATGGCTTAAGGG", "dna") == "MA"
    assert mv.target_protein("mkv", "protein") == "MKV"


def test_positions_follow_identical_sequences_fragments_and_windows():
    assert mv.position_map(UNIPROT, UNIPROT) == ({i: i for i in range(1, 21)}, "identical")
    mapping, method = mv.position_map("VLAGG", UNIPROT)
    assert (method, mapping[2]) == ("fragment", 4)
    mapping, method = mv.position_map("MKVLAGGTTTTTTTTTTRWE", UNIPROT)  # differs in its last residue
    assert (method, mapping[2]) == ("window", 2)
    assert 20 not in mapping  # its window contains the difference: left unmapped rather than guessed
    assert mv.position_map("WWWWWWWWWWWWWWW", UNIPROT) == ({}, "unmapped")


def test_one_row_per_protein_variant_with_the_median_score():
    rows = pd.DataFrame(
        {
            "hgvs_pro": [
                "p.Leu2Pro",
                "p.Leu2Pro",
                "p.Leu2Pro",
                "p.Gly4Ter",
                "p.Val1=",
                "p.=",
                "p.Ala3Gly",
                "p.[Leu2Pro;Ala3Gly]",
            ],
            "score": ["-1", "-2", "-9", "-3.0", "0.1", "0.0", "NA", "-4"],
        }
    )
    out = mv.process_score_set(rows, "VLAGG", UNIPROT).set_index(["kind", "assay_position"])
    missense = out.loc[("missense", 2)]
    assert (missense.score, missense.measurements, missense.uniprot_position) == (-2.0, 3, 4)
    assert missense.ref_matches_uniprot and missense.position_mapping == "fragment"
    assert out.loc[("nonsense", 4)].alt_aa == "*"
    assert out.loc[("synonymous", pd.NA)].position_mapping == "all synonymous"
    assert ("missense", 3) not in out.index  # its score is missing
    assert len(out) == 4


def test_a_reference_residue_that_disagrees_with_uniprot_is_flagged():
    rows = pd.DataFrame({"hgvs_pro": ["p.Trp2Pro"], "score": ["-1"]})  # position 2 of VLAGG is L, not W
    [flag] = mv.process_score_set(rows, "VLAGG", UNIPROT).ref_matches_uniprot
    assert flag is False


def test_positions_already_on_uniprot_numbering_are_recognised():
    # the target is the fragment VLAGG (residues 3-7), but the submitter numbered along the full protein
    rows = pd.DataFrame({"hgvs_pro": ["p.Leu4Pro", "p.Ala5Gly", "p.Gly6Ter"], "score": ["-1", "-2", "-3"]})
    out = mv.process_score_set(rows, "VLAGG", UNIPROT)
    assert set(out.position_mapping) == {"UniProt numbering"}
    assert out.uniprot_position.tolist() == [4, 5, 6] and out.ref_matches_uniprot.all()


def test_a_mavedb_offset_is_used_when_it_explains_the_positions():
    # positions count from the target's start, but the target sequence was stored with an extra leading codon
    rows = pd.DataFrame({"hgvs_pro": ["p.Lys1Pro", "p.Val2Gly", "p.Leu3Ter"], "score": ["-1", "-2", "-3"]})
    out = mv.process_score_set(rows, "XXXXXXXX", UNIPROT, offset=3, sequence_type="dna")  # 3 nt = 1 residue
    assert set(out.position_mapping) == {"MaveDB offset"}
    assert out.uniprot_position.tolist() == [2, 3, 4] and out.ref_matches_uniprot.all()
