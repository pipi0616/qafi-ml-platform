# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze: the QAFI proteome predictions, exactly as delivered
# MAGIC
# MAGIC The 7.4 GB archive holds one TSV per protein (86 GB unpacked). It is streamed one protein at a time
# MAGIC into Parquet files in the volume (see `lakehouse/proteome.py`), then loaded into a Delta table.
# MAGIC Nothing is cleaned here: duplicates, empty strings and mislabelled columns are kept, so the table
# MAGIC matches the source file for file. Parameter `limit` loads only the first N proteins (0 = all).

# COMMAND ----------

import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(".."))  # the bundle deploys src/ next to this notebook's folder
from lakehouse.proteome import write_bronze_parts

dbutils.widgets.text("catalog", "qafi_dev")
dbutils.widgets.text("limit", "0")
catalog = dbutils.widgets.get("catalog")
limit = int(dbutils.widgets.get("limit"))

volume = f"/Volumes/{catalog}/raw/files"
archive = f"{volume}/datasets/qafi-predictions/v1.0/exhaustive_mutagenesis_data_v1.0.tar.gz"
staging = f"{volume}/_staging/bronze_proteome"

# COMMAND ----------

shutil.rmtree(staging, ignore_errors=True)
stats = write_bronze_parts(archive, staging, version="v1.0", limit=limit)
print(stats)

# COMMAND ----------

table = f"{catalog}.bronze.proteome_predictions"
spark.sql(f"""
    CREATE OR REPLACE TABLE {table}
    CLUSTER BY (UniProtID)
    COMMENT 'QAFI proteome-wide predictions v1.0, one row per line of the source TSVs, exactly as delivered'
    AS SELECT *, current_timestamp() AS _ingested_at FROM parquet.`{staging}`
""")
spark.sql(f"""ALTER TABLE {table} ALTER COLUMN GeneID COMMENT
    'Holds UniProt entry names such as P53_HUMAN, not gene IDs: renamed entry_name in silver'""")
spark.sql(f"""ALTER TABLE {table} ALTER COLUMN _missing_columns COMMENT
    'Columns absent from the source file of this protein: NULL there, unlike empty cells (empty strings)'""")
shutil.rmtree(staging)

loaded = spark.table(table).count()
assert loaded == stats["rows"], f"{table} has {loaded} rows, the archive {stats['rows']}"
print(f"{table}: {loaded:,} rows from {stats['proteins']:,} proteins")
