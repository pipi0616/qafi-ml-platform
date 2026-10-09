import gzip
import io
import tarfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from lakehouse import alphafold, structure

MODEL = Path(__file__).parent / "data" / "AF-A0A024R1R8-F1-model_v6.pdb.gz"


@pytest.fixture(scope="module")
def pdb_text():
    return gzip.decompress(MODEL.read_bytes()).decode()


@pytest.fixture(scope="module")
def sequence(pdb_text):
    return "".join(structure.fragment_features(pdb_text)["aa"])


def _renumber(pdb_text: str, first: int, last: int, start_at: int) -> str:
    """Residues first..last of the model, numbered from start_at: a made-up AlphaFold fragment."""
    lines = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and first <= int(line[22:26]) <= last:
            lines.append(f"{line[:22]}{int(line[22:26]) - first + start_at:>4}{line[26:]}")
    return "\n".join(lines) + "\nEND\n"


def test_one_row_per_residue_with_plausible_values(pdb_text):
    f = structure.fragment_features(pdb_text)
    assert len(f) == 64 and f["position"].tolist() == list(range(1, 65))
    assert f["plddt"].iloc[0] == pytest.approx(57.69)  # the B-factor column of residue 1
    assert f["rsa"].between(0, 1.6).all()
    assert (f["contacts_12"] >= f["contacts_8"]).all()


def test_neighbours_are_mutual(pdb_text):
    f = structure.fragment_features(pdb_text).set_index("position")
    for position, row in f.iterrows():
        for other, distance in zip(row["neighbours"], row["distances"], strict=True):
            assert position in f.loc[other, "neighbours"] and distance < structure.NEIGHBOUR_DISTANCE


def test_a_single_model_covering_the_sequence_is_ok(pdb_text, sequence):
    features, status = structure.protein_features([(1, pdb_text)], sequence)
    assert status == "ok" and len(features) == 64


def test_a_sequence_that_differs_from_uniprot_gives_no_features(pdb_text, sequence):
    features, status = structure.protein_features([(1, pdb_text)], "M" + sequence[1:-1] + "W")
    assert status == "sequence differs from UniProt" and features.empty


def test_overlapping_fragments_are_stitched_by_the_most_central_residue(pdb_text, sequence):
    # two made-up fragments, each numbered from 1: residues 1-40 and 25-64
    fragments = [(1, _renumber(pdb_text, 1, 40, 1)), (2, _renumber(pdb_text, 25, 64, 1))]
    features, status = structure.protein_features(fragments, sequence)
    assert status == "ok" and features["position"].tolist() == list(range(1, 65))
    by_position = features.set_index("position")["fragment"]
    assert by_position[30] == 1  # 10 from fragment 1's end, 5 from fragment 2's start
    assert by_position[36] == 2  # 4 from fragment 1's end, 11 from fragment 2's start
    assert all(1 <= n <= 64 for row in features["neighbours"] for n in row)  # on UniProt numbering


def test_the_archive_keeps_each_pdb_model_as_delivered(tmp_path):
    data = MODEL.read_bytes()
    with tarfile.open(tmp_path / "proteome.tar", "w") as archive:
        for name in [
            "AF-A0A024R1R8-F1-model_v6.pdb.gz",
            "AF-A0A024R1R8-F1-model_v6.cif.gz",
            "AF-Q8WZ42-F2-model_v6.pdb.gz",
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    assert alphafold.write_bronze(tmp_path / "proteome.tar", tmp_path / "out") == 2  # the .cif.gz is skipped
    table = pq.read_table(tmp_path / "out").to_pandas().sort_values("uniprot_id")
    assert table[["uniprot_id", "fragment", "model_version"]].values.tolist() == [
        ["A0A024R1R8", 1, "v6"],
        ["Q8WZ42", 2, "v6"],
    ]
    assert table["pdb_gz"].iloc[0] == data


def test_half_sphere_exposure_matches_biopython(pdb_text):
    from Bio.PDB import HSExposureCB, PDBParser

    model = PDBParser(QUIET=True).get_structure("m", io.StringIO(pdb_text))[0]
    HSExposureCB(model, radius=13.0)
    expected = [(r.xtra["EXP_HSE_B_U"], r.xtra["EXP_HSE_B_D"]) for r in model.get_residues()]
    f = structure.fragment_features(pdb_text)
    assert list(zip(f["hse_up"], f["hse_down"], strict=True)) == expected


def test_accessible_surface_area_matches_biopython(pdb_text):
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley

    model = PDBParser(QUIET=True).get_structure("m", io.StringIO(pdb_text))[0]
    ShrakeRupley().compute(model, level="R")
    expected = [r.sasa for r in model.get_residues()]
    assert structure.fragment_features(pdb_text)["sasa"].tolist() == pytest.approx(expected, abs=0.01)
