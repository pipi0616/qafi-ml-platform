# Databricks notebook source
# MAGIC %md
# MAGIC # Silver: MaveDB variants on UniProt numbering (notebook 07c)
# MAGIC
# MAGIC Score sets kept: human protein-coding targets, current version (not superseded), one target protein.
# MAGIC Each score set is processed on its own (`groupBy("urn").applyInPandas`) by the tested functions in
# MAGIC `lakehouse/mavedb_variants.py`: variants classified as missense, nonsense or synonymous, positions put
# MAGIC on UniProt canonical numbering, one row per protein variant (median of its codon-level scores).
# MAGIC - `silver.mavedb_score_sets`: one row per kept score set (protein, publication, Domainome or not)
# MAGIC - `silver.mavedb_variants`: one row per protein variant per score set

# COMMAND ----------

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(".."))
from lakehouse.mavedb_variants import OUTPUT_COLUMNS, process_score_set, target_protein

dbutils.widgets.text("catalog", "qafi_dev")
catalog = dbutils.widgets.get("catalog")


def uniprot_reference(record: str) -> tuple[str | None, int | None]:
    """The target's UniProt accession and MaveDB's offset to it, from the score set's full metadata."""
    target = (json.loads(record).get("targetGenes") or [{}])[0]
    ids = [x for x in target.get("externalIdentifiers") or [] if (x.get("identifier") or {}).get("dbName") == "UniProt"]
    if ids:
        return ids[0]["identifier"]["identifier"], ids[0].get("offset")
    return target.get("uniprotIdFromMappedMetadata"), None


sets = spark.sql(f"""
    SELECT urn, title, record, target_hgnc, target_sequence, target_sequence_type,
           primary_pubmed, first_author, publication_year, experiment_urn
    FROM {catalog}.bronze.mavedb_score_sets
    WHERE target_organism = 'Homo sapiens' AND target_category = 'protein_coding'
      AND superseded_by IS NULL AND target_count = '1'
""").toPandas()
references = [uniprot_reference(r) for r in sets["record"]]
sets["uniprot_id"] = [accession for accession, _ in references]
sets["uniprot_offset"] = pd.array([offset for _, offset in references], dtype="Int64")
sets["target_protein"] = [target_protein(s, t) for s, t in zip(sets["target_sequence"], sets["target_sequence_type"], strict=True)]
sets["is_domainome"] = sets["title"].fillna("").str.contains("domainome", case=False)
print(f"{len(sets)} score sets kept")

# COMMAND ----------

uniprot = dict(spark.table(f"{catalog}.bronze.uniprot_proteome").select("accession", "sequence").toPandas().values)
info = {
    r.urn: (r.target_protein, uniprot.get(r.uniprot_id, ""), r.uniprot_offset, r.target_sequence_type, r.uniprot_id)
    for r in sets.itertuples()
}
schema = (
    "urn string, uniprot_id string, kind string, assay_position long, uniprot_position long, "
    "position_mapping string, ref_aa string, alt_aa string, score double, measurements long, ref_matches_uniprot boolean"
)


def one_score_set(rows: pd.DataFrame) -> pd.DataFrame:
    urn = rows["urn"].iloc[0]
    target, sequence, offset, sequence_type, accession = info[urn]
    offset = None if pd.isna(offset) else int(offset)
    out = process_score_set(rows, target, sequence, offset, sequence_type)
    return out.assign(urn=urn, uniprot_id=accession)[["urn", "uniprot_id", *OUTPUT_COLUMNS]]


kept = spark.createDataFrame(sets[["urn"]])
variants = (
    spark.table(f"{catalog}.bronze.mavedb_scores")
    .join(kept, "urn")
    .select("urn", "hgvs_pro", "score")
    .groupBy("urn")
    .applyInPandas(one_score_set, schema)
)
table = f"{catalog}.silver.mavedb_variants"
variants.write.mode("overwrite").option("overwriteSchema", "true").clusterBy("uniprot_id").saveAsTable(table)
spark.sql(f"COMMENT ON TABLE {table} IS 'MaveDB human variants, one row per protein variant per score set, on UniProt numbering (notebook 07c)'")
print(f"{table}: {spark.table(table).count():,} rows")

# COMMAND ----------

info_table = f"{catalog}.silver.mavedb_score_sets"
columns = ["urn", "title", "uniprot_id", "uniprot_offset", "target_hgnc", "target_sequence_type", "target_protein",
           "primary_pubmed", "first_author", "publication_year", "experiment_urn", "is_domainome"]  # fmt: skip
(
    spark.createDataFrame(sets[columns])
    .withColumnRenamed("target_hgnc", "gene_name")
    .withColumnRenamed("target_sequence_type", "sequence_type")
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(info_table)
)
spark.sql(f"COMMENT ON TABLE {info_table} IS 'MaveDB human score sets kept for labels, one row per score set (notebook 07c)'")
print(f"{info_table}: {spark.table(info_table).count():,} rows")
