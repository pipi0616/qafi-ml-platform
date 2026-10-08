"""Read UniProt's reference proteome FASTA into one record per protein (bronze, notebook 04)."""

from __future__ import annotations

import gzip
import re
from pathlib import Path

# >sp|P04637|P53_HUMAN Cellular tumor antigen p53 OS=Homo sapiens OX=9606 GN=TP53 PE=1 SV=4
_HEADER = re.compile(r">(?P<db>sp|tr)\|(?P<accession>[^|]+)\|(?P<entry_name>\S+) (?P<rest>.*)")
_GENE = re.compile(r"\bGN=(\S+)")
_NAME = re.compile(r"^(.*?) OS=")


def read_fasta(path: str | Path) -> list[dict]:
    """One dict per protein: accession, entry name, reviewed, protein name, gene name, length, sequence."""
    records: list[dict] = []
    header, chunks = None, []

    def flush() -> None:
        if header is None:
            return
        m = _HEADER.match(header)
        if m is None:
            raise ValueError(f"unexpected FASTA header: {header[:80]}")
        gene, name = _GENE.search(m["rest"]), _NAME.match(m["rest"])
        sequence = "".join(chunks)
        records.append(
            {
                "accession": m["accession"],
                "entry_name": m["entry_name"],
                "reviewed": m["db"] == "sp",
                "protein_name": name[1] if name else m["rest"],
                "gene_name": gene[1] if gene else None,
                "length": len(sequence),
                "sequence": sequence,
            }
        )

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush()
                header, chunks = line, []
            elif line:
                chunks.append(line)
    flush()
    return records
