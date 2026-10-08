import gzip
import io
import tarfile

import pyarrow.parquet as pq
import pytest
from lakehouse.proteome import COLUMNS, write_bronze_parts
from lakehouse.uniprot import read_fasta


def _row(acc: str, pos: int, ref: str, alt: str, am: str = "0.5") -> list[str]:
    values = dict.fromkeys(COLUMNS, "")  # empty cells, as in the real files
    values |= {"UniProtID": acc, "GeneName": "GENE", "GeneID": f"{acc}_HUMAN", "PositionAA": str(pos)}
    values |= {"RefAA": ref, "AltAA": alt, "VariantID": f"{ref}{pos}{alt}", "AM_pathogenicity": am}
    return [values[c] for c in COLUMNS]


def _archive(tmp_path, proteins: dict[str, list[list[str]]], columns: list[str] = COLUMNS):
    path = tmp_path / "proteome.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for acc, rows in proteins.items():
            data = "\n".join("\t".join(r) for r in [columns, *rows]).encode() + b"\n"
            info = tarfile.TarInfo(f"exhaustive_mutagenesis_data/{acc}/{acc}_variants.tsv")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_archive_becomes_parquet_exactly_as_written(tmp_path):
    tar = _archive(tmp_path, {"P1": [_row("P1", 1, "M", "A")] * 2, "P2": [_row("P2", 2, "K", "R")]})
    assert write_bronze_parts(tar, tmp_path / "out", version="v1.0") == {"proteins": 2, "rows": 3, "files": 1}
    table = pq.read_table(tmp_path / "out")
    assert table.num_rows == 3  # the duplicate row is kept: bronze is as-is
    assert table.column_names == [*COLUMNS, "_missing_columns", "_source_file", "_dataset_version"]
    assert set(table["_missing_columns"].to_pylist()) == {""}  # these files have every column
    assert table["REVEL"].to_pylist() == ["", "", ""]  # empty cells stay empty strings
    assert set(table["_source_file"].to_pylist()) == {
        "exhaustive_mutagenesis_data/P1/P1_variants.tsv",
        "exhaustive_mutagenesis_data/P2/P2_variants.tsv",
    }
    assert set(table["_dataset_version"].to_pylist()) == {"v1.0"}


def test_limit_and_new_file_after_enough_rows(tmp_path):
    tar = _archive(tmp_path, {f"P{i}": [_row(f"P{i}", 1, "M", "A")] * 3 for i in range(5)})
    stats = write_bronze_parts(tar, tmp_path / "out", "v1.0", limit=4, rows_per_part=5)
    assert stats == {"proteins": 4, "rows": 12, "files": 2}
    assert pq.read_table(tmp_path / "out").num_rows == 12


def test_missing_columns_are_null_and_recorded(tmp_path):
    keep = [c for c in COLUMNS if c not in ("qafi1", "QAFImt")]
    row = dict(zip(COLUMNS, _row("Q9", 1, "M", "A"), strict=True))
    tar = _archive(tmp_path, {"Q9": [[row[c] for c in keep]]}, columns=keep)
    write_bronze_parts(tar, tmp_path / "out", "v1.0")
    table = pq.read_table(tmp_path / "out")
    assert table["qafi1"].to_pylist() == [None]  # the file has no qafi1 column at all
    assert table["REVEL"].to_pylist() == [""]  # the file has REVEL, but the cell is empty
    assert table["_missing_columns"].to_pylist() == ["qafi1,QAFImt"]


def test_unknown_column_stops_the_load(tmp_path):
    columns = [*COLUMNS, "NewScore"]
    tar = _archive(tmp_path, {"P1": [[*_row("P1", 1, "M", "A"), "0.1"]]}, columns=columns)
    with pytest.raises(ValueError, match="unexpected columns"):
        write_bronze_parts(tar, tmp_path / "out", "v1.0")


def test_fasta_headers_become_columns(tmp_path):
    path = tmp_path / "proteome.fasta.gz"
    path.write_bytes(
        gzip.compress(
            b">sp|P04637|P53_HUMAN Cellular tumor antigen p53 OS=Homo sapiens OX=9606 GN=TP53 PE=1 SV=4\n"
            b"MEEPQSDPSV\nEPPLSQETFS\n"
            b">tr|A0A075B6K4|A0A075B6K4_HUMAN Uncharacterized protein OS=Homo sapiens OX=9606 PE=4 SV=1\n"
            b"MKK\n"
        )
    )
    p53, unnamed = read_fasta(path)
    assert p53 == {
        "accession": "P04637",
        "entry_name": "P53_HUMAN",
        "reviewed": True,
        "protein_name": "Cellular tumor antigen p53",
        "gene_name": "TP53",
        "length": 20,
        "sequence": "MEEPQSDPSVEPPLSQETFS",
    }
    assert (unnamed["reviewed"], unnamed["gene_name"], unnamed["length"]) == (False, None, 3)
