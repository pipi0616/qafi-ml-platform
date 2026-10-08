# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze: MaveDB, every published CC0 score set (notebook 07b)
# MAGIC
# MAGIC The bulk download is one tar.gz: `main.json` with the metadata of every score set, and per score set
# MAGIC a scores CSV plus, when MaveDB has mapped it to the human reference, an annotations CSV (VEP
# MAGIC consequence, post-mapped protein HGVS, gnomAD, ClinGen). The archive is streamed once into parquet
# MAGIC parts, then loaded as three tables with values as delivered:
# MAGIC - `bronze.mavedb_score_sets`: one row per score set
# MAGIC - `bronze.mavedb_scores`: one row per variant per score set
# MAGIC - `bronze.mavedb_annotations`: one row per mapped variant per score set

# COMMAND ----------

import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(".."))
from lakehouse.mavedb import write_bronze

dbutils.widgets.text("catalog", "qafi_dev")
dbutils.widgets.text("mavedb_version", "2026-06-25")
catalog = dbutils.widgets.get("catalog")
version = dbutils.widgets.get("mavedb_version")
source = f"/Volumes/{catalog}/raw/files/datasets/mavedb/{version}"
archive = f"{source}/{next(f for f in sorted(os.listdir(source)) if f.endswith('.tar.gz'))}"
staging = f"/Volumes/{catalog}/raw/files/_staging/mavedb_{version}"

shutil.rmtree(staging, ignore_errors=True)
print(write_bronze(archive, staging))

# COMMAND ----------

tables = {
    "score_sets": "MaveDB score sets, one row per score set with target, publication and full metadata",
    "scores": "MaveDB scores, one row per variant per score set, as delivered",
    "annotations": "MaveDB annotations, VEP consequence and post-mapped HGVS per mapped variant",
}
for name, comment in tables.items():
    table = f"{catalog}.bronze.mavedb_{name}"
    spark.sql(f"""
        CREATE OR REPLACE TABLE {table}
        CLUSTER BY (urn)
        COMMENT '{comment} ({version})'
        AS SELECT *, '{archive}' AS _source_file, '{version}' AS _dataset_version, current_timestamp() AS _ingested_at
        FROM parquet.`{staging}/{name}`
    """)
    print(f"{table}: {spark.table(table).count():,} rows")
shutil.rmtree(staging)
