"""The prediction API (notebook 10b): the released scores of any missense variant of the human proteome.

  GET /health                                       is the service up (no data read)
  GET /v1/proteins/{protein}                        a protein, by gene symbol (TP53) or accession (P04637)
  GET /v1/proteins/{protein}/variants/{variant}     one variant, e.g. /v1/proteins/TP53/variants/R175H
  GET /v1/proteins/{protein}/positions/{position}   the 19 substitutions at one position
  GET /v1/proteins/{protein}/sites                  what each position is for (notebook 14e)

Both scores are percentiles between 0 and 1, low = likely damaging (notebook 09a). Every answer comes from the
Athena views ``variant_effect_current``, ``proteins_current`` and ``sites_current`` (notebooks 10a and 14e):
whatever release they point at is what the API serves. The site map covers the proteins notebook 14e mapped,
not the whole proteome, so a protein outside it answers 404.

Run it here with ``uvicorn qafi_platform.api:app``; on AWS the same container runs in Lambda (notebook
10b). The interactive documentation is at /docs.
"""

from __future__ import annotations

import json
import os
import re
import time
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from qafi_platform import athena

ACCESSION = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$")  # UniProt
PROTEIN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,29}$"  # a gene symbol or an accession
VARIANT = r"^[ACDEFGHIKLMNPQRSTVWY][1-9][0-9]{0,4}[ACDEFGHIKLMNPQRSTVWY]$"

VARIANT_COLUMNS = (
    "uniprot_id, gene_name, variant, position, ref_aa, alt_aa, stability_score, function_score, zero_shot, "
    "release, stability_head_version, function_trees_version"
)
PROTEIN_COLUMNS = "uniprot_id, gene_name, entry_name, protein_name, length, has_dms, release"
SITE_COLUMNS = "position, fold_cost, func_cost, position_is"
POSITION_KINDS = ("structural", "functional", "mixed", "tolerant")


class Protein(BaseModel):
    uniprot_id: str = Field(examples=["P04637"])
    gene_name: str | None = Field(examples=["TP53"])
    entry_name: str | None = Field(examples=["P53_HUMAN"])
    protein_name: str | None = Field(examples=["Cellular tumor antigen p53"])
    length: int = Field(description="Residues in the UniProt sequence", examples=[393])
    has_dms: bool = Field(description="Has deep mutational scanning data in the platform")
    release: str = Field(description="Release of the scores this answer comes from")


class Variant(BaseModel):
    uniprot_id: str = Field(examples=["P04637"])
    gene_name: str | None = Field(examples=["TP53"])
    variant: str = Field(examples=["R175H"])
    position: int = Field(examples=[175])
    ref_aa: str = Field(examples=["R"])
    alt_aa: str = Field(examples=["H"])
    stability_score: float = Field(description="0 to 1, low = likely destabilising (a percentile)")
    function_score: float = Field(description="0 to 1, low = likely loss of function (a percentile)")
    zero_shot: float = Field(description="ESM-2 log-probability of the new amino acid minus the old one")
    release: str
    stability_head_version: str = Field(description="Version of qafi_dev.ml.stability_head in the registry")
    function_trees_version: str = Field(description="Version of qafi_dev.ml.function_trees in the registry")


class Site(BaseModel):
    position: int = Field(examples=[12])
    fold_cost: float = Field(description="Mean ddG of folding over the 19 substitutions, kcal/mol")
    func_cost: float = Field(description="Mean ddG of function over the 19 substitutions, kcal/mol")
    position_is: str = Field(
        description="structural (holds the fold together), functional (the protein still folds but stops "
        "working), mixed (both) or tolerant (neither)",
        examples=["functional"],
    )


class Sites(BaseModel):
    protein: Protein
    mapped: int = Field(description="Positions of this protein in the map")
    sites: list[Site] = Field(description="Ordered by functional cost, most sensitive first")


class Position(BaseModel):
    protein: Protein
    position: int
    ref_aa: str
    variants: list[Variant] = Field(description="The 19 substitutions, alphabetical by new amino acid")


class Lake:
    """The two views, read through Athena. Answers are also kept in memory for ``ttl`` seconds: while
    Lambda keeps the function warm, a repeated question costs nothing and takes milliseconds."""

    def __init__(self, client, workgroup: str, database: str, ttl: float = 300, size: int = 2048):
        self.client, self.workgroup, self.database = client, workgroup, database
        self.ttl, self.size, self.cache = ttl, size, {}

    def query(self, sql: str, parameters: list) -> list[dict]:
        key = (sql, tuple(parameters))
        hit = self.cache.get(key)
        if hit and time.monotonic() - hit[0] < self.ttl:
            return hit[1]
        rows = athena.run(self.client, sql, self.workgroup, self.database, parameters, reuse_minutes=60).rows
        if len(self.cache) >= self.size:
            self.cache.pop(next(iter(self.cache)))  # the oldest entry
        self.cache[key] = (time.monotonic(), rows)
        return rows

    def proteins(self, name: str) -> list[dict]:
        """Proteins whose accession or gene symbol is ``name`` (upper case)."""
        return self.query(
            f"SELECT {PROTEIN_COLUMNS} FROM proteins_current WHERE uniprot_id = ? OR upper(gene_name) = ?",
            [athena.literal(name), athena.literal(name)],
        )

    def sites(self, uniprot_id: str, position_is: str | None) -> list[dict]:
        """What each position of a protein is for, most functionally sensitive first (notebook 14e)."""
        where = "uniprot_id = ?" + (" AND position_is = ?" if position_is else "")
        parameters = [athena.literal(uniprot_id)] + ([athena.literal(position_is)] if position_is else [])
        return self.query(
            f"SELECT {SITE_COLUMNS} FROM sites_current WHERE {where} ORDER BY func_cost", parameters
        )

    def position(self, uniprot_id: str, position: int) -> list[dict]:
        """The scored substitutions at one position. The files are sorted by protein, so Athena reads only
        the few megabytes holding this protein, not the whole table."""
        return self.query(
            f"SELECT {VARIANT_COLUMNS} FROM variant_effect_current WHERE uniprot_id = ? AND position = ? "
            "ORDER BY alt_aa",
            [athena.literal(uniprot_id), int(position)],
        )


@lru_cache
def lake() -> Lake:
    import boto3

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "eu-central-1"
    return Lake(
        boto3.client("athena", region_name=region),  # Lambda sets AWS_REGION
        workgroup=os.environ.get("ATHENA_WORKGROUP", "qafi-dev"),
        database=os.environ.get("ATHENA_DATABASE", "qafi_dev_lake"),
    )


app = FastAPI(
    title="QAFI variant effect API",
    version="1.0",
    description="Stability and function scores of every possible missense variant of the human proteome. "
    "Both are between 0 and 1; low means likely damaging.",
)
LakeDependency = Annotated[Lake, Depends(lake)]
ProteinName = Annotated[str, Path(pattern=PROTEIN, description="Gene symbol or UniProt accession")]


@app.middleware("http")
async def log_request(request: Request, call_next):
    """One JSON line per request in the logs (CloudWatch on AWS): what was asked, the answer, how long."""
    started = time.monotonic()
    response = await call_next(request)
    line = {"path": request.url.path, "status": response.status_code}
    print(json.dumps({**line, "ms": round((time.monotonic() - started) * 1000)}), flush=True)
    return response


def _protein(lake: Lake, name: str) -> dict:
    found = lake.proteins(name.upper())
    if not found:
        raise HTTPException(404, f"no protein named {name}")
    if len(found) > 1:
        accessions = ", ".join(sorted(p["uniprot_id"] for p in found))
        raise HTTPException(409, f"{name} names several proteins ({accessions}): ask for one accession")
    return found[0]


def _position(lake: Lake, protein: dict, position: int) -> list[dict]:
    label = protein["gene_name"] or protein["uniprot_id"]
    if not 1 <= position <= protein["length"]:
        raise HTTPException(404, f"{label} has {protein['length']} residues")
    rows = lake.position(protein["uniprot_id"], position)
    if not rows:  # a residue that is not one of the 20 amino acids, e.g. selenocysteine
        raise HTTPException(404, f"position {position} of {label} has no scores")
    return rows


def _variant(row: dict) -> Variant:
    scores = {k: round(row[k], 4) for k in ("stability_score", "function_score", "zero_shot")}
    return Variant(**{**row, **scores})


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/proteins/{protein}")
def get_protein(protein: ProteinName, lake: LakeDependency) -> Protein:
    return Protein(**_protein(lake, protein))


@app.get("/v1/proteins/{protein}/variants/{variant}")
def get_variant(
    protein: ProteinName,
    variant: Annotated[str, Path(pattern=VARIANT, description="One-letter code, e.g. R175H")],
    lake: LakeDependency,
) -> Variant:
    found = _protein(lake, protein)
    ref, position, alt = variant[0], int(variant[1:-1]), variant[-1]
    rows = _position(lake, found, position)
    if rows[0]["ref_aa"] != ref:
        label = found["gene_name"] or found["uniprot_id"]
        raise HTTPException(422, f"position {position} of {label} is {rows[0]['ref_aa']}, not {ref}")
    if alt == ref:
        raise HTTPException(422, f"{variant} changes nothing")
    return _variant(next(row for row in rows if row["alt_aa"] == alt))


@app.get("/v1/proteins/{protein}/sites")
def get_sites(
    protein: ProteinName,
    lake: LakeDependency,
    position_is: Annotated[str | None, Query(pattern="|".join(POSITION_KINDS))] = None,
) -> Sites:
    """What each position of a protein is for: structural, functional, mixed or tolerant (notebook 14e)."""
    found = _protein(lake, protein)
    rows = lake.sites(found["uniprot_id"], position_is)
    if not rows:
        label = found["gene_name"] or found["uniprot_id"]
        kind = f" classed {position_is}" if position_is else ""
        raise HTTPException(404, f"{label} has no mapped positions{kind}")
    sites = [
        Site(**{**row, "fold_cost": round(row["fold_cost"], 3), "func_cost": round(row["func_cost"], 3)})
        for row in rows
    ]
    return Sites(protein=Protein(**found), mapped=len(sites), sites=sites)


@app.get("/v1/proteins/{protein}/positions/{position}")
def get_position(
    protein: ProteinName, position: Annotated[int, Path(ge=1, le=99999)], lake: LakeDependency
) -> Position:
    found = _protein(lake, protein)
    rows = _position(lake, found, position)
    return Position(
        protein=Protein(**found),
        position=position,
        ref_aa=rows[0]["ref_aa"],
        variants=[_variant(r) for r in rows],
    )
