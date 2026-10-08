# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze: AlphaFold DB human proteome models (notebook 07d)
# MAGIC
# MAGIC The archive holds every model twice (mmCIF and PDB). The PDB files are kept, compressed, exactly as
# MAGIC delivered: `bronze.alphafold_models` has one row per model with the protein, the fragment number
# MAGIC (long proteins come in several) and the file's bytes. The archive is read once as a stream.

# COMMAND ----------

import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(".."))
from lakehouse.alphafold import write_bronze

dbutils.widgets.text("catalog", "qafi_dev")
dbutils.widgets.text("alphafold_version", "v6")
catalog = dbutils.widgets.get("catalog")
version = dbutils.widgets.get("alphafold_version")
source = f"/Volumes/{catalog}/raw/files/datasets/alphafold-human/{version}"
archive = f"{source}/{next(f for f in sorted(os.listdir(source)) if f.endswith('.tar'))}"
staging = f"/Volumes/{catalog}/raw/files/_staging/alphafold_{version}"

shutil.rmtree(staging, ignore_errors=True)
print(write_bronze(archive, staging), "models")

# COMMAND ----------

table = f"{catalog}.bronze.alphafold_models"
spark.sql(f"""
    CREATE OR REPLACE TABLE {table}
    CLUSTER BY (uniprot_id)
    COMMENT 'AlphaFold DB {version} human proteome: one row per model, PDB file compressed as delivered'
    AS SELECT *, '{archive}' AS _source_file, '{version}' AS _dataset_version, current_timestamp() AS _ingested_at
    FROM parquet.`{staging}`
""")
print(f"{table}: {spark.table(table).count():,} rows")
shutil.rmtree(staging)
