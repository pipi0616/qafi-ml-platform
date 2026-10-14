import pytest
from fastapi.testclient import TestClient

from qafi_platform import api, athena

from .athena_fake import FakeAthena

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PROTEINS = [  # uniprot_id, gene_name, entry_name, protein_name, length, has_dms, release
    ("P04637", "TP53", "P53_HUMAN", "Cellular tumor antigen p53", 393, True, "r1"),
    ("Q00001", "TWIN", "TWIN1_HUMAN", "Twin one", 100, False, "r1"),
    ("Q00002", "TWIN", "TWIN2_HUMAN", "Twin two", 120, False, "r1"),
]
PROTEIN_TYPES = ["varchar", "varchar", "varchar", "varchar", "integer", "boolean", "varchar"]
VARIANT_COLUMNS = [
    ("uniprot_id", "varchar"), ("gene_name", "varchar"), ("variant", "varchar"), ("position", "integer"),
    ("ref_aa", "varchar"), ("alt_aa", "varchar"), ("stability_score", "float"), ("function_score", "float"),
    ("zero_shot", "float"), ("release", "varchar"), ("stability_head_version", "varchar"),
    ("function_trees_version", "varchar"),
]  # fmt: skip


def answer(sql, parameters):
    """The two views: TP53 has scores at position 175 (R) only; position 200 has none (not scored)."""
    if "FROM proteins_current" in sql:
        accession, gene = parameters
        columns = list(zip([c.strip() for c in api.PROTEIN_COLUMNS.split(",")], PROTEIN_TYPES, strict=True))
        return columns, [p for p in PROTEINS if p[0] == accession or p[1].upper() == gene]
    if "FROM variant_effect_current" in sql:
        uniprot_id, position = parameters[0], int(parameters[1])
        if (uniprot_id, position) != ("P04637", 175):
            return VARIANT_COLUMNS, []
        rows = [
            ("P04637", "TP53", f"R175{alt}", 175, "R", alt, 0.123456, 0.3, -5.0, "r1", "1", "1")
            for alt in AMINO_ACIDS
            if alt != "R"
        ]
        return VARIANT_COLUMNS, rows
    raise AssertionError(sql)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(athena.time, "sleep", lambda s: None)
    fake = FakeAthena(answer)
    lake = api.Lake(fake, "wg", "db")
    api.app.dependency_overrides[api.lake] = lambda: lake
    with TestClient(api.app) as test_client:
        test_client.athena = fake
        yield test_client
    api.app.dependency_overrides.clear()


def test_health_reads_no_data(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.athena.requests == []


def test_a_protein_by_gene_symbol_or_accession(client):
    by_gene = client.get("/v1/proteins/tp53").json()
    assert by_gene["uniprot_id"] == "P04637" and by_gene["length"] == 393 and by_gene["release"] == "r1"
    assert client.get("/v1/proteins/P04637").json() == by_gene


def test_one_variant(client):
    body = client.get("/v1/proteins/TP53/variants/R175H").json()
    assert body["variant"] == "R175H" and body["alt_aa"] == "H"
    assert body["stability_score"] == 0.1235  # rounded for reading
    assert body["stability_head_version"] == "1"


def test_the_19_substitutions_at_a_position(client):
    body = client.get("/v1/proteins/TP53/positions/175").json()
    assert body["ref_aa"] == "R" and body["protein"]["gene_name"] == "TP53"
    assert [v["alt_aa"] for v in body["variants"]] == [a for a in AMINO_ACIDS if a != "R"]


@pytest.mark.parametrize(
    "path, status, detail",
    [
        ("/v1/proteins/NOPE1/variants/R175H", 404, "no protein named NOPE1"),
        ("/v1/proteins/TWIN/variants/A5G", 409, "Q00001, Q00002"),
        ("/v1/proteins/TP53/variants/R400H", 404, "TP53 has 393 residues"),
        ("/v1/proteins/TP53/variants/G175H", 422, "position 175 of TP53 is R, not G"),
        ("/v1/proteins/TP53/variants/R175R", 422, "R175R changes nothing"),
        ("/v1/proteins/TP53/positions/200", 404, "position 200 of TP53 has no scores"),
    ],
)
def test_questions_without_an_answer_say_why(client, path, status, detail):
    response = client.get(path)
    assert response.status_code == status and detail in response.json()["detail"]


@pytest.mark.parametrize(
    "path",
    ["/v1/proteins/TP53/variants/R175X", "/v1/proteins/TP53/variants/175H", "/v1/proteins/x'--/variants/R1H"],
)
def test_malformed_input_never_reaches_athena(client, path):
    assert client.get(path).status_code in (404, 422)
    assert client.athena.requests == []


def test_a_repeated_question_is_answered_from_memory(client):
    for _ in range(3):
        client.get("/v1/proteins/TP53/variants/R175H")
    assert len(client.athena.requests) == 2  # the protein once, the position once
    request = client.athena.requests[1]
    assert request["ExecutionParameters"] == ["'P04637'", "175"]  # values as parameters, not in the SQL
    assert request["ResultReuseConfiguration"]["ResultReuseByAgeConfiguration"]["MaxAgeInMinutes"] == 60


def test_the_documentation_lists_the_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {
        "/health",
        "/v1/proteins/{protein}",
        "/v1/proteins/{protein}/variants/{variant}",
        "/v1/proteins/{protein}/positions/{position}",
    }


def test_without_a_region_the_service_still_answers_bad_input_properly(monkeypatch):
    for var in ("AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    api.lake.cache_clear()
    try:
        with TestClient(api.app) as test_client:
            assert test_client.get("/v1/proteins/TP53/variants/R175X").status_code == 422
        assert api.lake().client.meta.region_name == "eu-central-1"
    finally:
        api.lake.cache_clear()
