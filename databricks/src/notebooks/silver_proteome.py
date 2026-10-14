# Databricks notebook source
# MAGIC %md
# MAGIC # Silver: one row per protein variant
# MAGIC
# MAGIC In bronze a variant can appear hundreds of times: exact duplicates, plus a join fan-out that paired
# MAGIC every genomic position with every REVEL and BayesDel value (up to 729 rows per variant). Only the
# MAGIC genomic columns (chr, hg19_pos, REVEL, BayesDel) and the QAFIMeta scores built from them vary; the
# MAGIC protein-level scores are the same in every copy. Silver therefore keeps:
# MAGIC
# MAGIC - grain: one row per (uniprot_id, position, ref_aa, alt_aa)
# MAGIC - the protein-level scores, typed as numbers, with empty strings turned into NULL
# MAGIC - `entry_name` (the bronze column wrongly called GeneID) and the real NCBI `gene_id` from UniProt
# MAGIC - `source_rows`: how many bronze rows collapsed into this one; `conflicting_columns`: the scores on
# MAGIC   which those rows disagreed, and `has_conflict` whether there are any. A score whose copies disagree
# MAGIC   is unknown, so it is NULL for that variant. In practice only EVE and popEVE ever disagree (0.4% of
# MAGIC   variants in the first 2,000 proteins, apparently values of different transcripts joined to the
# MAGIC   same position); the quality checks fail the job if any other score does.
# MAGIC
# MAGIC The fan-out columns are left out on purpose: which pairing is real cannot be told from this file.
# MAGIC They stay available in bronze until they are rebuilt from their own sources.

# COMMAND ----------

dbutils.widgets.text("catalog", "qafi_dev")
c = dbutils.widgets.get("catalog")

# bronze column -> silver column, for the columns that must be identical in every copy of a variant
PROTEIN_LEVEL = {
    "GeneName": "gene_name",
    "GeneID": "entry_name",
    "VariantID": "variant",
    "AM_class": "am_class",
    "AM_pathogenicity": "am_pathogenicity",
    "popEVE": "popeve",
    "EVE": "eve",
    "CPT1": "cpt1",
    "qafi1": "qafi1",
    "qafi2": "qafi2",
    "QAFImt": "qafi_mt",
    "qafisplit1": "qafi_split1",
    "qafisplit2": "qafi_split2",
    "qafisplit3": "qafi_split3",
    "QAFIClass1": "qafi_class1",
    "QAFIClass2": "qafi_class2",
}
TEXT = {"GeneName", "GeneID", "VariantID", "AM_class"}

# a score is kept only when every copy of the variant agrees on it; otherwise it is unknown (NULL)
firsts = ",\n        ".join(f"CASE WHEN min({b}) = max({b}) THEN min({b}) END AS {b}" for b in PROTEIN_LEVEL)
conflicting = ", ".join(f"CASE WHEN min({b}) <> max({b}) THEN '{s}' END" for b, s in PROTEIN_LEVEL.items())
columns = ",\n    ".join(
    f"NULLIF(v.{b}, '') AS {s}" if b in TEXT else f"TRY_CAST(NULLIF(v.{b}, '') AS DOUBLE) AS {s}"
    for b, s in PROTEIN_LEVEL.items()
)

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {c}.silver.variant_scores
CLUSTER BY (uniprot_id)
COMMENT 'QAFI proteome predictions v1.0: one row per protein variant with the protein-level scores'
AS
WITH by_variant AS (
    SELECT UniProtID, PositionAA, RefAA, AltAA,
        count(*) AS source_rows,
        {firsts},
        min(_missing_columns) AS missing_columns,
        filter(array({conflicting}), name -> name IS NOT NULL) AS conflicting_columns
    FROM {c}.bronze.proteome_predictions
    GROUP BY UniProtID, PositionAA, RefAA, AltAA
),
gene_ids AS (
    SELECT accession, array_join(array_sort(collect_set(id)), ';') AS gene_id
    FROM {c}.bronze.uniprot_idmapping
    WHERE id_type = 'GeneID'
    GROUP BY accession
),
reference AS (
    SELECT DISTINCT accession FROM {c}.bronze.uniprot_proteome
)
SELECT
    v.UniProtID AS uniprot_id,
    CAST(v.PositionAA AS INT) AS position,
    v.RefAA AS ref_aa,
    v.AltAA AS alt_aa,
    g.gene_id,
    {columns},
    v.source_rows,
    v.conflicting_columns,
    size(v.conflicting_columns) > 0 AS has_conflict,
    NULLIF(v.missing_columns, '') AS missing_columns,
    r.accession IS NOT NULL AS in_reference_proteome
FROM by_variant v
LEFT JOIN gene_ids g ON g.accession = v.UniProtID
LEFT JOIN reference r ON r.accession = v.UniProtID
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {c}.silver.protein_audit
COMMENT 'Per protein: how many bronze rows became how many variants, and what its source file lacked'
AS
SELECT
    uniprot_id,
    sum(source_rows) AS bronze_rows,
    count(*) AS variants,
    max(source_rows) AS max_rows_per_variant,
    round(sum(source_rows) / count(*), 1) AS rows_per_variant,
    max(missing_columns) AS missing_columns,
    bool_or(has_conflict) AS any_conflict,
    bool_and(in_reference_proteome) AS in_reference_proteome
FROM {c}.silver.variant_scores
GROUP BY uniprot_id
""")

for t in ("variant_scores", "protein_audit"):
    print(f"{c}.silver.{t}: {spark.table(f'{c}.silver.{t}').count():,} rows")
