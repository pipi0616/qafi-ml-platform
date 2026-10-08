"""Turn the QAFI proteome tarball into Parquet files without unpacking it (bronze, notebook 04).

The archive holds one TSV per protein (``exhaustive_mutagenesis_data/<ACC>/<ACC>_variants.tsv``,
86 GB unpacked). It is read as a stream, one member at a time, so it never has to be extracted
to disk. Every value is kept exactly as written, as a string: an empty cell stays "". Typing and
cleaning happen in silver. Each row records which archive member it came from and the dataset
version, so any bronze row can be traced back to its source file.

Not every file has every column (some lack qafi1, QAFImt, qafisplit1 and qafisplit2). A column
missing from a file is NULL in bronze, unlike an empty cell (""), and ``_missing_columns`` lists
what that file lacked. A column we don't know stops the load: the format has changed.
"""

from __future__ import annotations

import tarfile
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

COLUMNS = [
    "UniProtID", "GeneName", "GeneID", "chr", "hg19_pos", "PositionAA", "RefAA", "AltAA", "VariantID",
    "AM_pathogenicity", "AM_class", "popEVE", "EVE", "CPT1", "REVEL", "BayesDel",
    "qafi1", "qafi2", "QAFImt", "qafisplit1", "qafisplit2", "qafisplit3", "QAFIClass1", "QAFIClass2",
    "QAFIMeta_v1", "QAFIMeta_v2", "QAFIMeta_v3", "QAFIMeta_v4",
]  # fmt: skip

_PARSE = pacsv.ParseOptions(delimiter="\t", quote_char=False)
_CONVERT = pacsv.ConvertOptions(column_types={c: pa.string() for c in COLUMNS}, strings_can_be_null=False)


def protein_tables(tar_path: str | Path, limit: int = 0) -> Iterator[tuple[str, pa.Table]]:
    """Yield ``(member name, table of strings)`` for each protein TSV in the archive, in archive order.

    Every table has ``COLUMNS`` in that order plus ``_missing_columns``. ``limit`` stops after
    that many proteins (0 means all). A file with a column outside ``COLUMNS`` raises.
    """
    count = 0
    with tarfile.open(tar_path, mode="r|gz") as tar:
        for member in tar:
            if not (member.isfile() and member.name.endswith("_variants.tsv")):
                continue
            data = tar.extractfile(member).read()
            raw = pacsv.read_csv(pa.BufferReader(data), parse_options=_PARSE, convert_options=_CONVERT)
            unknown = [c for c in raw.column_names if c not in COLUMNS]
            if unknown:
                raise ValueError(f"{member.name}: unexpected columns {unknown}")
            missing = [c for c in COLUMNS if c not in raw.column_names]
            n = raw.num_rows
            arrays = [raw[c] if c in raw.column_names else pa.nulls(n, pa.string()) for c in COLUMNS]
            arrays.append(pa.array([",".join(missing)] * n, pa.string()))
            yield member.name, pa.table(arrays, names=[*COLUMNS, "_missing_columns"])
            count += 1
            if limit and count >= limit:
                return


def write_bronze_parts(
    tar_path: str | Path, out_dir: str | Path, version: str, limit: int = 0, rows_per_part: int = 20_000_000
) -> dict:
    """Write the archive as zstd Parquet files in ``out_dir``, adding the lineage columns.

    A protein never spans two files; a new file starts once the current one holds
    ``rows_per_part`` rows. Returns how many proteins, rows and files were written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proteins = rows = part = part_rows = 0
    writer = None
    try:
        for name, table in protein_tables(tar_path, limit):
            n = table.num_rows
            table = table.append_column("_source_file", pa.array([name] * n, pa.string()))
            table = table.append_column("_dataset_version", pa.array([version] * n, pa.string()))
            if writer is None or part_rows >= rows_per_part:
                if writer is not None:
                    writer.close()
                part += 1
                writer = pq.ParquetWriter(
                    out_dir / f"part-{part:05d}.parquet", table.schema, compression="zstd"
                )
                part_rows = 0
            writer.write_table(table)
            proteins, rows, part_rows = proteins + 1, rows + n, part_rows + n
    finally:
        if writer is not None:
            writer.close()
    return {"proteins": proteins, "rows": rows, "files": part}
