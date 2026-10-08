# Databricks notebook source
# MAGIC %md
# MAGIC # Silver: the DMS training data, one row per single substitution per assay
# MAGIC
# MAGIC - grain: one row per (dms_id, position, ref_aa, alt_aa)
# MAGIC - only single substitutions: QAFI predicts the effect of one amino-acid change. Multi-mutants
# MAGIC   (`A1C:D2E`, about 72% of the rows) stay in bronze.
# MAGIC - the mutant string is split into position, reference and alternative amino acid, and the reference is
# MAGIC   checked against the assay's target sequence (`ref_matches_sequence`, enforced by the quality checks)
# MAGIC - scores as published: ProteinGym already orients DMS_score so that higher means fitter. Scaling per
# MAGIC   assay is a modelling choice and belongs to the training chapters.
# MAGIC - `uniprot_id`: the accession, from the UniProt reference proteome by entry name (human proteins)

# COMMAND ----------

dbutils.widgets.text("catalog", "qafi_dev")
c = dbutils.widgets.get("catalog")
SINGLE = "'^([A-Z])([0-9]+)([A-Z])$'"

spark.sql(f"""
CREATE OR REPLACE TABLE {c}.silver.dms_scores
CLUSTER BY (dms_id)
COMMENT 'ProteinGym DMS assays: one row per single amino-acid substitution per assay (the QAFI training data)'
AS
WITH singles AS (
    SELECT
        DMS_filename,
        mutant,
        regexp_extract(mutant, {SINGLE}, 1) AS ref_aa,
        CAST(regexp_extract(mutant, {SINGLE}, 2) AS INT) AS position,
        regexp_extract(mutant, {SINGLE}, 3) AS alt_aa,
        TRY_CAST(DMS_score AS DOUBLE) AS dms_score,
        TRY_CAST(DMS_score_bin AS INT) AS dms_score_bin
    FROM {c}.bronze.proteingym_dms
    WHERE mutant RLIKE {SINGLE}
)
SELECT
    a.DMS_id AS dms_id,
    a.UniProt_ID AS uniprot_entry_name,
    p.accession AS uniprot_id,
    a.taxon,
    a.coarse_selection_type AS selection_type,
    s.position,
    s.ref_aa,
    s.alt_aa,
    s.mutant,
    s.dms_score,
    s.dms_score_bin,
    substring(a.target_seq, s.position, 1) = s.ref_aa AS ref_matches_sequence
FROM singles s
JOIN {c}.bronze.proteingym_assays a ON a.DMS_filename = s.DMS_filename
LEFT JOIN {c}.bronze.uniprot_proteome p ON p.entry_name = a.UniProt_ID
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {c}.silver.dms_assay_audit
COMMENT 'Per DMS assay: rows published, single substitutions kept, multi-mutants left in bronze'
AS
SELECT
    a.DMS_id AS dms_id,
    a.UniProt_ID AS uniprot_entry_name,
    a.taxon,
    count(*) AS bronze_rows,
    count_if(d.mutant RLIKE {SINGLE}) AS single_substitutions,
    count_if(NOT d.mutant RLIKE {SINGLE}) AS multi_mutants_left_in_bronze
FROM {c}.bronze.proteingym_dms d
JOIN {c}.bronze.proteingym_assays a ON a.DMS_filename = d.DMS_filename
GROUP BY a.DMS_id, a.UniProt_ID, a.taxon
""")

for t in ("dms_scores", "dms_assay_audit"):
    print(f"{c}.silver.{t}: {spark.table(f'{c}.silver.{t}').count():,} rows")
