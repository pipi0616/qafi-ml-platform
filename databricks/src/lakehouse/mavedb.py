"""MaveDB bulk download -> bronze tables (notebook 07b).

The archive (mavedb-dump.<timestamp>.tar.gz) holds:
  main.json                    every experiment set > experiment > score set, with all metadata
  csv/{urn}.scores.csv         one row per variant: accession, hgvs_nt, hgvs_pro, hgvs_splice, scores.*
  csv/{urn}.annotations.csv    MaveDB's mapping to human reference sequences, VEP consequence, gnomAD,
                               ClinGen (only score sets that went through MaveDB's mapping)
  csv/{urn}.counts.csv, mapped/, va/   not needed here
The archive was packed on a Mac, so next to every file there is a "._" file of macOS metadata; skipped.

The archive is read once as a stream, without unpacking it to disk, and written as parquet parts that
Spark loads into Delta. Values stay as delivered, as strings. Each score set defines its own extra score
columns, so those are kept together as one JSON string; the main score is always ``scores.score``.
"""

from __future__ import annotations

import csv
import io
import json
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCORE_COLUMNS = ["urn", "accession", "hgvs_nt", "hgvs_pro", "hgvs_splice", "score", "other_scores"]
# The annotation file's documented columns; dots become underscores, because Spark reads a dot as a field
ANNOTATION_COLUMNS = [
    "accession",
    "hgvs_nt",
    "hgvs_pro",
    "hgvs_splice",
    "mavedb.post_mapped_hgvs_g",
    "mavedb.post_mapped_hgvs_c",
    "mavedb.post_mapped_hgvs_p",
    "mavedb.post_mapped_hgvs_at_assay_level",
    "mavedb.post_mapped_vrs_digest",
    "vep.vep_functional_consequence",
    "gnomad.gnomad_af",
    "clingen.clingen_allele_id",
]


def urn_from_file(name: str) -> str:
    """'.../csv/urn-mavedb-00000001-a-1.scores.csv' -> 'urn:mavedb:00000001-a-1'."""
    stem = Path(name).name.split(".")[0]
    return stem.replace("urn-mavedb-", "urn:mavedb:", 1)


def _first(items: list | None) -> dict:
    return (items or [{}])[0] or {}


def _text(row: dict) -> dict:
    """Bronze keeps values as delivered, as text: JSON numbers become their text, missing stays missing."""
    return {k: None if v is None else str(v) for k, v in row.items()}


def score_set_rows(main: dict) -> list[dict]:
    """One row per score set, with the fields needed downstream and the full record as JSON."""
    rows = []
    for experiment_set in main["experimentSets"]:
        for experiment in experiment_set.get("experiments") or []:
            for s in experiment.get("scoreSets") or []:
                target = _first(s.get("targetGenes"))
                sequence = target.get("targetSequence") or {}
                publication = _first(s.get("primaryPublicationIdentifiers"))
                first_author = next(
                    (a.get("name") for a in publication.get("authors") or [] if a.get("primary")), None
                )
                superseding = s.get("supersedingScoreSet") or {}
                if isinstance(superseding, dict):
                    superseding = superseding.get("urn")
                rows.append(
                    _text(
                        {
                            "urn": s["urn"],
                            "title": s.get("title"),
                            "short_description": s.get("shortDescription"),
                            "experiment_urn": experiment.get("urn"),
                            "experiment_set_urn": experiment_set.get("urn"),
                            "num_variants": s.get("numVariants"),
                            "published_date": s.get("publishedDate"),
                            "superseded_by": superseding,
                            "target_count": len(s.get("targetGenes") or []),
                            "target_name": target.get("name"),
                            "target_category": target.get("category"),
                            "target_hgnc": target.get("mappedHgncName"),
                            "target_uniprot": target.get("uniprotIdFromMappedMetadata"),
                            "target_organism": (sequence.get("taxonomy") or {}).get("organismName"),
                            "target_sequence_type": sequence.get("sequenceType"),
                            "target_sequence": sequence.get("sequence"),
                            "primary_pubmed": publication.get("identifier"),
                            "primary_doi": publication.get("doi"),
                            "first_author": first_author,
                            "publication_year": publication.get("publicationYear"),
                            "score_columns": json.dumps((s.get("datasetColumns") or {}).get("scoreColumns")),
                            "calibrations": json.dumps(s.get("scoreCalibrations")),
                            "record": json.dumps(s),
                        }
                    )
                )
    return rows


def score_rows(urn: str, text: str) -> dict[str, list]:
    """The rows of one scores.csv, column by column."""
    out: dict[str, list] = {c: [] for c in SCORE_COLUMNS}
    for row in csv.DictReader(io.StringIO(text)):
        extra = {k: v for k, v in row.items() if k.startswith("scores.") and k != "scores.score"}
        out["urn"].append(urn)
        out["accession"].append(row.get("accession"))
        out["hgvs_nt"].append(row.get("hgvs_nt"))
        out["hgvs_pro"].append(row.get("hgvs_pro"))
        out["hgvs_splice"].append(row.get("hgvs_splice"))
        out["score"].append(row.get("scores.score"))
        out["other_scores"].append(json.dumps(extra) if extra else None)
    return out


def annotation_rows(urn: str, text: str) -> dict[str, list]:
    """The rows of one annotations.csv, column by column."""
    out: dict[str, list] = {"urn": []} | {c.replace(".", "_"): [] for c in ANNOTATION_COLUMNS}
    for row in csv.DictReader(io.StringIO(text)):
        out["urn"].append(urn)
        for c in ANNOTATION_COLUMNS:
            out[c.replace(".", "_")].append(row.get(c))
    return out


class _PartWriter:
    """Collects columns and writes them as numbered parquet parts of at most ``rows`` rows."""

    def __init__(self, directory: Path, rows: int):
        self.directory, self.rows, self.parts, self.total = directory, rows, 0, 0
        self.buffer: dict[str, list] = {}
        directory.mkdir(parents=True, exist_ok=True)

    def add(self, columns: dict[str, list]) -> None:
        for name, values in columns.items():
            self.buffer.setdefault(name, []).extend(values)
        if self.buffer and len(next(iter(self.buffer.values()))) >= self.rows:
            self.flush()

    def flush(self) -> None:
        n = len(next(iter(self.buffer.values()), []))
        if n == 0:
            return
        table = pa.table({k: pa.array(v, type=pa.string()) for k, v in self.buffer.items()})
        pq.write_table(table, self.directory / f"part-{self.parts:05d}.parquet")
        self.parts, self.total, self.buffer = self.parts + 1, self.total + n, {}


def write_bronze(tar_path: str | Path, out_dir: str | Path, rows_per_part: int = 500_000) -> dict[str, int]:
    """Stream the archive once into parquet parts under ``out_dir``/{score_sets,scores,annotations}.

    Returns the number of rows written per table.
    """
    out = Path(out_dir)
    scores = _PartWriter(out / "scores", rows_per_part)
    annotations = _PartWriter(out / "annotations", rows_per_part)
    sets: list[dict] = []
    with tarfile.open(tar_path, "r|gz") as archive:
        for member in archive:
            name = member.name
            if not member.isfile() or Path(name).name.startswith("._"):
                continue  # "._" files are macOS metadata the archive was packed with, not data
            if name.endswith("/main.json"):
                sets = score_set_rows(json.load(archive.extractfile(member)))
            elif "/csv/" in name and name.endswith(".scores.csv"):
                scores.add(score_rows(urn_from_file(name), archive.extractfile(member).read().decode()))
            elif "/csv/" in name and name.endswith(".annotations.csv"):
                annotations.add(
                    annotation_rows(urn_from_file(name), archive.extractfile(member).read().decode())
                )
    scores.flush()
    annotations.flush()
    (out / "score_sets").mkdir(parents=True, exist_ok=True)
    columns = list(sets[0]) if sets else []
    pq.write_table(
        pa.table({c: pa.array([r[c] for r in sets], type=pa.string()) for c in columns}),
        out / "score_sets" / "part-00000.parquet",
    )
    return {"score_sets": len(sets), "scores": scores.total, "annotations": annotations.total}
