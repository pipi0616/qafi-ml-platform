"""AlphaFold DB human proteome archive -> bronze (notebook 07d).

The archive (UP000005640_9606_HUMAN_v<N>.tar) holds, per model, AF-<UniProt>-F<fragment>-model_v<N>.cif.gz
and the same model as .pdb.gz. The PDB files are kept, compressed, exactly as delivered: one row per model
with the protein, the fragment number and the file's bytes. The archive is read once as a stream.
"""

from __future__ import annotations

import re
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

NAME = re.compile(r"AF-(?P<uniprot>[A-Z0-9-]+)-F(?P<fragment>\d+)-model_v(?P<version>\d+)\.pdb\.gz$")
SCHEMA = pa.schema(
    [
        ("uniprot_id", pa.string()),
        ("fragment", pa.int32()),
        ("model_version", pa.string()),
        ("pdb_gz", pa.binary()),
    ]
)


def write_bronze(tar_path: str | Path, out_dir: str | Path, rows_per_part: int = 2000) -> int:
    """Write every PDB model of the archive as parquet parts under ``out_dir``; returns how many there are."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    batch: dict[str, list] = {name: [] for name in SCHEMA.names}
    parts = total = 0

    def flush() -> None:
        nonlocal parts, batch
        if batch["uniprot_id"]:
            pq.write_table(pa.table(batch, schema=SCHEMA), out / f"part-{parts:05d}.parquet")
            parts += 1
            batch = {name: [] for name in SCHEMA.names}

    with tarfile.open(tar_path, "r|") as archive:
        for member in archive:
            match = NAME.search(member.name)
            if not member.isfile() or not match:
                continue
            batch["uniprot_id"].append(match["uniprot"])
            batch["fragment"].append(int(match["fragment"]))
            batch["model_version"].append(f"v{match['version']}")
            batch["pdb_gz"].append(archive.extractfile(member).read())
            total += 1
            if len(batch["uniprot_id"]) >= rows_per_part:
                flush()
    flush()
    return total
