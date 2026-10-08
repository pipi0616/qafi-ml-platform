import io
import json
import tarfile

import pyarrow.parquet as pq
from lakehouse import mavedb

MAIN = {
    "title": "MaveDB public data",
    "asOf": "2026-06-24T18:13:15+00:00",
    "experimentSets": [
        {
            "urn": "urn:mavedb:00000001",
            "experiments": [
                {
                    "urn": "urn:mavedb:00000001-a",
                    "scoreSets": [
                        {
                            "urn": "urn:mavedb:00000001-a-1",
                            "title": "BRCA1 RING domain",
                            "numVariants": 3,
                            "supersedingScoreSet": {"urn": "urn:mavedb:00000001-a-2"},
                            "targetGenes": [
                                {
                                    "name": "BRCA1",
                                    "category": "protein_coding",
                                    "mappedHgncName": "BRCA1",
                                    "uniprotIdFromMappedMetadata": "P38398",
                                    "targetSequence": {
                                        "sequence": "ATGGATTTA",
                                        "sequenceType": "dna",
                                        "taxonomy": {"organismName": "Homo sapiens"},
                                    },
                                }
                            ],
                            "primaryPublicationIdentifiers": [
                                {
                                    "identifier": "25823446",
                                    "doi": "10.1534/genetics.115.175802",
                                    "publicationYear": 2015,
                                    "authors": [{"name": "Starita, Lea", "primary": True}],
                                }
                            ],
                            "datasetColumns": {"scoreColumns": ["score", "se"]},
                        }
                    ],
                }
            ],
        }
    ],
}
SCORES = "accession,hgvs_nt,hgvs_pro,hgvs_splice,scores.score,scores.se\n" + "".join(
    f"urn:mavedb:00000001-a-1#{i},NA,{p},NA,{s},0.1\n"
    for i, (p, s) in enumerate([("p.Met1Leu", "-1.2"), ("p.=", "0.05"), ("p.Asp2Ter", "-2.9")], 1)
)
ANNOTATIONS = (
    ",".join(mavedb.ANNOTATION_COLUMNS)
    + "\n"
    + (
        "urn:mavedb:00000001-a-1#1,NA,p.Met1Leu,NA,NA,NA,NP_009225.1:p.Met1Leu,NA,ga4gh:VA.x,missense_variant,0.0001,CA1\n"
    )
)


def _archive(path):
    files = {
        "main.json": json.dumps(MAIN),
        "csv/urn-mavedb-00000001-a-1.scores.csv": SCORES,
        "csv/urn-mavedb-00000001-a-1.annotations.csv": ANNOTATIONS,
        "csv/urn-mavedb-00000001-a-1.counts.csv": "accession,counts.c_0\nx,1\n",  # not needed: skipped
        "mapped/urn-mavedb-00000001-a-1.mapped-variants.json": "{}",  # not needed: skipped
        "csv/._urn-mavedb-00000001-a-1.scores.csv": "\x00Mac OS X        \x00\x02",  # macOS metadata: skipped
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(f"mavedb-dump.20260624/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_file_names_give_the_score_set_urn():
    assert mavedb.urn_from_file("dump/csv/urn-mavedb-00000001-a-1.scores.csv") == "urn:mavedb:00000001-a-1"


def test_score_sets_are_flattened_with_their_target_and_publication():
    [row] = mavedb.score_set_rows(MAIN)
    assert (row["urn"], row["experiment_urn"], row["target_uniprot"]) == (
        "urn:mavedb:00000001-a-1",
        "urn:mavedb:00000001-a",
        "P38398",
    )
    assert (row["first_author"], row["publication_year"], row["target_organism"]) == (
        "Starita, Lea",
        "2015",
        "Homo sapiens",
    )
    assert row["superseded_by"] == "urn:mavedb:00000001-a-2"
    assert json.loads(row["record"])["title"] == "BRCA1 RING domain"


def test_the_archive_becomes_three_bronze_tables(tmp_path):
    _archive(tmp_path / "dump.tar.gz")
    counts = mavedb.write_bronze(tmp_path / "dump.tar.gz", tmp_path / "out")
    assert counts == {"score_sets": 1, "scores": 3, "annotations": 1}
    scores = pq.read_table(tmp_path / "out" / "scores").to_pandas()
    assert scores["hgvs_pro"].tolist() == ["p.Met1Leu", "p.=", "p.Asp2Ter"]
    assert scores["score"].tolist() == ["-1.2", "0.05", "-2.9"]  # strings, as delivered
    assert json.loads(scores["other_scores"][0]) == {"scores.se": "0.1"}
    annotations = pq.read_table(tmp_path / "out" / "annotations").to_pandas()
    assert annotations.loc[0, "vep_vep_functional_consequence"] == "missense_variant"
    assert annotations.loc[0, "mavedb_post_mapped_hgvs_p"] == "NP_009225.1:p.Met1Leu"
