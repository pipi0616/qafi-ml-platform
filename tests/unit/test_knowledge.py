import io
import json
from types import SimpleNamespace as NS

from Bio.SeqFeature import FeatureLocation, SeqFeature, UnknownPosition

from qafi_platform import claude, knowledge

from .conftest import BUCKET


def feature(kind, start, end, note=None):
    qualifiers = {"note": note} if note else {}
    return SeqFeature(FeatureLocation(start, end), type=kind, qualifiers=qualifiers)


TP53 = NS(
    accessions=["P04637", "Q15086"],
    entry_name="P53_HUMAN",
    gene_name=[{"Name": "TP53 {ECO:0000312|HGNC:HGNC:11998}", "Synonyms": ["P53"]}],
    description="RecName: Full=Cellular tumor antigen p53 {ECO:0000305}; AltName: Full=Antigen NY-CO-13;",
    sequence_length=393,
    comments=[
        "FUNCTION: Multifunctional transcription factor (PubMed:11025664, PubMed:12524540). Acts as a "
        "tumor suppressor. {ECO:0000269|PubMed:9840937}.",
        "DISEASE: Li-Fraumeni syndrome (LFS) [MIM:151623]: Autosomal dominant familial cancer.",
        "INTERACTION: Q00987; MDM2; NbExp=24;",  # not kept
    ],
    features=[
        feature("DNA_BIND", 101, 292),
        feature("VARIANT", 174, 175, "R -> H (in LFS; germline mutation; dbSNP:rs28934578)"),
        feature("VARIANT", 238, 240, "Missing (in a sporadic cancer)"),
        feature("MUTAGEN", 174, 175, "R->A,H: Loss of DNA binding. {ECO:0000269|PubMed:1}"),
        feature("MUTAGEN", 21, 23, "LW->QS: Loss of interaction with MDM2."),
        feature("REGION", UnknownPosition(), 50, "Disordered"),
        feature("HELIX", 10, 20),  # not kept
    ],
    keywords=["Apoptosis", "Tumor suppressor"],
)


def test_a_swiss_prot_record_becomes_a_readable_document():
    text, metadata = knowledge.document(TP53)
    assert text.startswith("# TP53: Cellular tumor antigen p53\n")
    assert "393 amino acids. Gene: TP53." in text
    assert (
        "## Function\nMultifunctional transcription factor. Acts as a tumor suppressor." in text
    )  # no noise
    assert "## Disease\nLi-Fraumeni syndrome (LFS) [MIM:151623]" in text
    assert "Q00987" not in text and "helix" not in text  # interactions and secondary structure left out
    assert "- dna bind 102-292" in text and "- region ?-50: Disordered" in text
    assert "- R175H: in LFS; germline mutation; dbSNP:rs28934578" in text
    assert "- 239-240: Missing (in a sporadic cancer)" in text
    assert "- R175A/R175H: Loss of DNA binding." in text  # an experiment testing two substitutions
    assert "- 22-23: LW->QS: Loss of interaction with MDM2." in text
    assert text.rstrip().endswith("Apoptosis, Tumor suppressor")
    assert metadata == {"uniprot_id": "P04637", "gene_name": "TP53", "collection": "uniprot"}
    _, unnamed = knowledge.document(NS(**{**vars(TP53), "gene_name": []}))
    assert unnamed == {"uniprot_id": "P04637", "collection": "uniprot"}  # Bedrock refuses empty values


def test_documents_go_up_with_their_metadata_and_only_when_changed(s3):
    items = [("uniprot/2026_03/P04637.md", b"# TP53", {"uniprot_id": "P04637"})]
    assert knowledge.upload(s3, BUCKET, items) == 2  # the document and its metadata
    assert knowledge.upload(s3, BUCKET, items) == 0  # nothing changed: nothing written
    assert (
        knowledge.upload(s3, BUCKET, [("uniprot/2026_03/P04637.md", b"# TP53 v2", {"uniprot_id": "P04637"})])
        == 1
    )
    sidecar = json.loads(
        s3.get_object(Bucket=BUCKET, Key="uniprot/2026_03/P04637.md.metadata.json")["Body"].read()
    )
    assert sidecar == {"metadataAttributes": {"uniprot_id": "P04637"}}


def test_retrieval_returns_passages_with_their_source_and_filters_on_metadata():
    sent = {}

    def retrieve(**request):
        sent.update(request)
        return {
            "retrievalResults": [
                {
                    "content": {"text": "TP53 acts as a tumor suppressor."},
                    "location": {"s3Location": {"uri": "s3://kb/uniprot/2026_03/P04637.md"}},
                    "score": 0.81234,
                    "metadata": {"gene_name": "TP53", "x-amz-bedrock-kb-chunk-id": "c1"},
                }
            ]
        }

    where = {"equals": {"key": "gene_name", "value": "TP53"}}
    passages = knowledge.retrieve(NS(retrieve=retrieve), "KB1", "what does p53 do", k=3, where=where)
    assert passages == [
        {
            "score": 0.812,
            "source": "P04637.md",
            "text": "TP53 acts as a tumor suppressor.",
            "metadata": {"gene_name": "TP53"},
        }
    ]
    assert sent["retrievalConfiguration"]["vectorSearchConfiguration"] == {
        "numberOfResults": 3,
        "filter": where,
    }


def test_the_answer_rests_on_the_numbered_passages_and_says_which(monkeypatch):
    seen = {}

    def messages(session, body):
        seen.update(body)
        return {"content": [{"type": "text", "text": "A tumor suppressor [2]. See also [7]."}],
                "usage": {"input_tokens": 1000, "output_tokens": 100}}  # fmt: skip

    monkeypatch.setattr(claude, "messages", messages)
    passages = [
        {"source": "A.md", "text": "a"},
        {"source": "P04637.md", "text": "TP53 is a tumor suppressor."},
    ]
    result = knowledge.answer(None, "What does TP53 do?", passages)
    assert "[2] (source: P04637.md)\nTP53 is a tumor suppressor." in seen["messages"][0]["content"]
    assert "Use only the" in seen["system"] and seen["model"] == claude.HAIKU
    assert result["cited"] == ["P04637.md"]  # [7] does not exist and is ignored
    assert result["usd"] == round((1000 * 1.0 + 100 * 5.0) / 1e6, 6)


def test_claude_requests_are_signed_and_retried_when_busy(monkeypatch):
    import urllib.error

    calls = []

    def urlopen(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 529, "overloaded", {}, io.BytesIO(b"busy"))
        return io.BytesIO(json.dumps({"content": [], "usage": {}}).encode())

    monkeypatch.setattr(claude.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(claude.time, "sleep", lambda s: None)
    session = NS(get_credentials=lambda: NS(get_frozen_credentials=lambda: NS(
        access_key="AKIDEXAMPLE", secret_key="secret", token=None)))  # fmt: skip
    claude.messages(session, {"model": claude.HAIKU})
    assert (
        len(calls) == 2
        and calls[1].full_url == "https://bedrock-mantle.eu-west-1.api.aws/anthropic/v1/messages"
    )
    authorization = calls[1].get_header("Authorization")
    assert "/eu-west-1/bedrock-mantle/aws4_request" in authorization
    assert calls[1].get_header("Anthropic-version") == "2023-06-01"


def test_a_refused_request_says_why():
    import urllib.error

    import pytest

    def urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, 403, "no", {}, io.BytesIO(b'{"error": "not available"}')
        )

    session = NS(get_credentials=lambda: NS(get_frozen_credentials=lambda: NS(
        access_key="AKIDEXAMPLE", secret_key="secret", token=None)))  # fmt: skip
    original = claude.urllib.request.urlopen
    claude.urllib.request.urlopen = urlopen
    try:
        with pytest.raises(RuntimeError, match="403.*not available"):
            claude.messages(session, {"model": claude.HAIKU})
    finally:
        claude.urllib.request.urlopen = original


def test_ingestion_is_started_and_followed_until_it_completes(monkeypatch):
    import time

    monkeypatch.setattr(time, "sleep", lambda s: None)
    states = iter(["IN_PROGRESS", "COMPLETE"])
    client = NS(
        start_ingestion_job=lambda **kw: {"ingestionJob": {"ingestionJobId": "J1", "status": "STARTING"}},
        get_ingestion_job=lambda **kw: {
            "ingestionJob": {
                "ingestionJobId": "J1",
                "status": next(states),
                "statistics": {"numberOfDocumentsScanned": 3},
            }
        },
    )
    assert knowledge.ingest(client, "KB1", "DS1")["numberOfDocumentsScanned"] == 3


def test_a_failed_ingestion_says_why():
    import pytest

    client = NS(
        start_ingestion_job=lambda **kw: {"ingestionJob": {"ingestionJobId": "J1", "status": "STARTING"}},
        get_ingestion_job=lambda **kw: {
            "ingestionJob": {"ingestionJobId": "J1", "status": "FAILED", "failureReasons": ["access denied"]}
        },
    )
    with pytest.raises(RuntimeError, match="access denied"):
        knowledge.ingest(client, "KB1", "DS1", every=0)
