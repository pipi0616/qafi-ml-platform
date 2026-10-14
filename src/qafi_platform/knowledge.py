"""The knowledge base behind question answering (notebook 11): documents in, passages and cited answers out.

  documents(path)                          one Markdown document per reviewed human protein (Swiss-Prot)
  upload(s3, bucket, documents)            the documents and their metadata into the knowledge bucket
  ingest(client, knowledge_base, source)   cut, embed and index what is new or changed in a source
  retrieve(client, knowledge_base, query)  the passages closest in meaning to a question
  answer(session, question, passages)      Claude answers from those passages only, citing them

The knowledge base itself (chunking, embedding with Titan, the S3 Vectors index) lives in Bedrock, built by
Terraform (infra/dev/knowledge.tf). Retrieval and generation are two separate steps here, so every step can be
inspected: which passages were found, what Claude was given, what it answered.
"""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

from qafi_platform import claude

COMMENTS = {  # UniProt comment sections kept, in this order, with their headings
    "FUNCTION": "Function",
    "SUBUNIT": "Subunit",
    "SUBCELLULAR LOCATION": "Subcellular location",
    "TISSUE SPECIFICITY": "Tissue specificity",
    "DOMAIN": "Domains",
    "PTM": "Post-translational modifications",
    "DISEASE": "Disease",
    "POLYMORPHISM": "Polymorphism",
    "SIMILARITY": "Similarity",
}
REGIONS = ("SIGNAL", "TRANSIT", "TRANSMEM", "DOMAIN", "REGION", "DNA_BIND", "ZN_FING", "MOTIF", "ACT_SITE",
           "BINDING", "SITE")  # fmt: skip
EVIDENCE = re.compile(r"\s*\{ECO:[^}]*\}")  # evidence codes: noise for a reader and for the embeddings
REFERENCES = re.compile(r"\s*\((?:PubMed:\d+(?:, )?)+\)")  # lists of PubMed IDs, e.g. (PubMed:1, PubMed:2)
# VARIANT notes read "R -> H (in LFS; ...)", MUTAGEN notes "R->A,H: Loss of DNA binding."
CHANGE = re.compile(r"^([A-Z]+) ?-> ?([A-Z]+(?:,[A-Z]+)*)(?: \((.*)\)|: (.*))?$")


def _clean(text: str) -> str:
    text = REFERENCES.sub("", EVIDENCE.sub("", text))
    return " ".join(re.sub(r"(?<!\.)\.\.(?!\.)", ".", text).split())  # "tissues. {ECO:..}." left "tissues.."


def _where(feature) -> str:
    """Residues of a feature, numbered from 1 as in UniProt: ``175`` or ``102-292`` (``?`` where unknown)."""

    def number(position, shift):
        try:
            return str(int(position) + shift)
        except TypeError:  # an unknown or fuzzy end, written ? in UniProt
            return "?"

    start, end = number(feature.location.start, 1), number(feature.location.end, 0)
    return start if start == end else f"{start}-{end}"


def _change(feature) -> str:
    """A VARIANT or MUTAGEN feature as one line, e.g. ``R175H: in LFS; somatic mutation``."""
    where = _where(feature)
    note = _clean(feature.qualifiers.get("note", ""))
    found = CHANGE.match(note)
    if found and where.isdigit() and len(found.group(1)) == 1:
        ref, alts, in_brackets, after_colon = found.groups()
        name = "/".join(f"{ref}{where}{alt}" for alt in alts.split(","))  # an experiment may test several
        return f"- {name}: {in_brackets or after_colon or 'no description'}"
    return f"- {where}: {note}"


def document(record) -> tuple[str, dict]:
    """One Swiss-Prot record as a Markdown document, and its metadata for filtering."""
    accession = record.accessions[0]
    gene = _clean(record.gene_name[0].get("Name") or "") if record.gene_name else ""  # may carry evidence too
    name = re.search(r"RecName: Full=([^;{]+)", record.description)
    name = name.group(1).strip() if name else record.entry_name
    lines = [
        f"# {gene or accession}: {name}",
        "",
        f"UniProt accession {accession}, entry {record.entry_name}, human, "
        f"{record.sequence_length} amino acids." + (f" Gene: {gene}." if gene else ""),
    ]
    for kind, heading in COMMENTS.items():
        texts = [_clean(c.split(":", 1)[1]) for c in record.comments if c.startswith(kind + ":")]
        if texts:
            lines += ["", f"## {heading}", *texts]
    regions = [f for f in record.features if f.type in REGIONS]
    if regions:
        lines += ["", "## Regions and sites"]
        for f in regions:
            note = _clean(f.qualifiers.get("note", ""))
            lines.append(f"- {f.type.replace('_', ' ').lower()} {_where(f)}" + (f": {note}" if note else ""))
    for kind, heading in (("VARIANT", "Natural variants"), ("MUTAGEN", "Mutagenesis experiments")):
        changes = [_change(f) for f in record.features if f.type == kind]
        if changes:
            lines += ["", f"## {heading} ({gene or accession})", *changes]
    if record.keywords:
        lines += ["", "## Keywords", ", ".join(record.keywords)]
    metadata = {"uniprot_id": accession, "collection": "uniprot"} | (
        {"gene_name": gene} if gene else {}
    )  # no empty values
    return "\n".join(lines) + "\n", metadata


def documents(path) -> Iterator[tuple[str, str, dict]]:
    """``(accession, document, metadata)`` for every entry of a Swiss-Prot flat file (``.dat.gz``)."""
    from Bio import SwissProt

    with gzip.open(path, "rt") as handle:
        for record in SwissProt.parse(handle):
            text, metadata = document(record)
            yield record.accessions[0], text, metadata


def upload(s3, bucket: str, items: list[tuple[str, bytes, dict]], workers: int = 32) -> int:
    """Put ``(key, body, metadata)`` items into the bucket, each with its ``<key>.metadata.json``: the
    metadata Bedrock attaches to every passage of that document, so searches can be filtered by it.

    A file already there with the same content is skipped (its ETag is the MD5 of the content), so running
    this again uploads only what changed, and the next ingestion re-embeds only that. Returns files written.
    """
    import hashlib

    files = []
    for key, body, metadata in items:
        files.append((key, body))
        files.append((f"{key}.metadata.json", json.dumps({"metadataAttributes": metadata}).encode()))
    stored = {}
    for prefix in sorted({key.split("/", 1)[0] + "/" for key, _ in files}):
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            stored.update({o["Key"]: o["ETag"].strip('"') for o in page.get("Contents", [])})
    changed = [(k, b) for k, b in files if stored.get(k) != hashlib.md5(b).hexdigest()]

    def put(item):
        s3.put_object(Bucket=bucket, Key=item[0], Body=item[1])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(put, changed))
    return len(changed)


def ingest(client, knowledge_base: str, source: str, timeout: float = 7200, every: float = 30) -> dict:
    """Run an ingestion job and wait for it: Bedrock reads the source's documents, cuts them into passages,
    embeds each passage and writes it to the index. Unchanged documents are skipped on a later run.
    Returns the job's statistics (documents scanned, indexed, failed ...) and its minutes."""
    import time

    job = client.start_ingestion_job(knowledgeBaseId=knowledge_base, dataSourceId=source)["ingestionJob"]
    started = time.monotonic()
    while job["status"] in ("STARTING", "IN_PROGRESS"):
        if time.monotonic() - started > timeout:
            raise TimeoutError(f"ingestion job {job['ingestionJobId']} still running after {timeout} s")
        time.sleep(every)
        job = client.get_ingestion_job(
            knowledgeBaseId=knowledge_base, dataSourceId=source, ingestionJobId=job["ingestionJobId"]
        )["ingestionJob"]
    if job["status"] != "COMPLETE":
        raise RuntimeError(f"ingestion job {job['status']}: {job.get('failureReasons')}")
    return {**job.get("statistics", {}), "minutes": round((time.monotonic() - started) / 60, 1)}


def retrieve(client, knowledge_base: str, query: str, k: int = 5, where: dict | None = None) -> list[dict]:
    """The ``k`` passages closest in meaning to ``query``: text, score, source document and metadata.

    ``where`` filters on metadata first, e.g. ``{"equals": {"key": "gene_name", "value": "TP53"}}``.
    """
    search = {"numberOfResults": k}
    if where:
        search["filter"] = where
    response = client.retrieve(
        knowledgeBaseId=knowledge_base,
        retrievalQuery={"text": query},
        retrievalConfiguration={"vectorSearchConfiguration": search},
    )
    return [
        {
            "score": round(r.get("score", 0.0), 3),
            "source": r["location"]["s3Location"]["uri"].rsplit("/", 1)[-1],
            "text": r["content"]["text"],
            "metadata": {k: v for k, v in r.get("metadata", {}).items() if not k.startswith("x-amz-bedrock")},
        }
        for r in response["retrievalResults"]
    ]


SYSTEM = (
    "You answer questions about human proteins and the QAFI variant effect predictor. Use only the "
    "numbered passages you are given. After every statement, cite the passages it rests on, like [1] or "
    "[2][3]. If the passages do not contain the answer, say that you cannot find it in the knowledge base; "
    "do not use outside knowledge. Answer in at most 150 words."
)


def answer(session, question: str, passages: list[dict], model: str = claude.HAIKU) -> dict:
    """Claude's answer from the passages only: the text, the passages it cited, tokens and cost."""
    numbered = "\n\n".join(f"[{i}] (source: {p['source']})\n{p['text']}" for i, p in enumerate(passages, 1))
    reply = claude.messages(
        session,
        {
            "model": model,
            "max_tokens": 600,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": f"Passages:\n\n{numbered}\n\nQuestion: {question}"}],
        },
    )
    text = "".join(block.get("text", "") for block in reply["content"])
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(passages)})
    return {
        "answer": text,
        "cited": [passages[i - 1]["source"] for i in cited],
        "input_tokens": reply["usage"]["input_tokens"],
        "output_tokens": reply["usage"]["output_tokens"],
        "usd": claude.cost(model, reply["usage"]),
    }
