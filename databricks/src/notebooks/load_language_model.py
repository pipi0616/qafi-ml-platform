# Databricks notebook source
# MAGIC %md
# MAGIC # Silver: load ESM-2 results computed elsewhere (notebook 07e)
# MAGIC
# MAGIC ESM-2 belongs on a GPU. Where the workspace has none (Free Edition's GPU quota is 0), the proteins
# MAGIC are read on a laptop's GPU with `lakehouse.language_model` (same code, model version and precision),
# MAGIC saved as parquet files with the silver tables' columns and uploaded to the volume. This task needs no
# MAGIC GPU: it loads such a folder (`backfill`). A protein is added when the tables don't have it, or have
# MAGIC another sequence or model version for it, or when the folder brings vectors the tables lack. Loading
# MAGIC the same folder twice changes nothing; a load cut off half-way is completed by the next one.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
from lakehouse import language_model as lm

dbutils.widgets.text("catalog", "qafi_dev")
dbutils.widgets.text("backfill", "")  # a volume folder written by lakehouse.language_model.read_to_parquet
catalog = dbutils.widgets.get("catalog")
folder = dbutils.widgets.get("backfill")
residues = f"{catalog}.silver.residue_language_model"
embeddings = f"{catalog}.silver.residue_embedding"
proteins = f"{catalog}.silver.language_model_proteins"
for statement in lm.table_ddl(catalog):
    spark.sql(statement)

if folder:
    new = spark.sql(f"""
        SELECT b.uniprot_id FROM parquet.`{folder}/proteins` b
        LEFT JOIN {proteins} t ON t.uniprot_id = b.uniprot_id
            AND t.sequence_md5 = b.sequence_md5 AND t.model_version = b.model_version
        WHERE t.uniprot_id IS NULL OR (b.embedded AND NOT t.embedded)
    """).toPandas()
    print(f"{len(new):,} proteins to add from {folder}")
    if len(new):
        spark.createDataFrame(new).createOrReplaceTempView("new_proteins")
        listed = "SELECT uniprot_id FROM new_proteins"
        # proteins last: only then do these proteins count as done
        for table, kind in ((residues, "residues"), (embeddings, "embeddings"), (proteins, "proteins")):
            spark.sql(f"DELETE FROM {table} WHERE uniprot_id IN ({listed})")
            if os.path.exists(f"{folder}/{kind}"):
                rows = f"SELECT * FROM parquet.`{folder}/{kind}` WHERE uniprot_id IN ({listed})"
                spark.sql(f"INSERT INTO {table} {rows}")
display(
    spark.sql(f"""
        SELECT model_version, count(*) AS proteins, sum(residues) AS residues,
               sum(CAST(embedded AS INT)) AS with_vectors
        FROM {proteins} GROUP BY model_version
    """)
)
