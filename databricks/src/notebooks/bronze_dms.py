# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze: ProteinGym DMS assays (the training data), exactly as published
# MAGIC
# MAGIC ProteinGym v1.3 has 217 deep mutational scanning assays: one CSV per assay inside a zip (columns
# MAGIC mutant, mutated_sequence, DMS_score, DMS_score_bin), plus a reference table describing each assay
# MAGIC (protein, taxon, target sequence, selection type, ...). The zip is unpacked into the volume's staging
# MAGIC area and both are loaded as delivered; each DMS row also records which assay file it came from.

# COMMAND ----------

import shutil
import zipfile

from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "qafi_dev")
dbutils.widgets.text("proteingym_version", "v1.3")
catalog = dbutils.widgets.get("catalog")
version = dbutils.widgets.get("proteingym_version")
source = f"/Volumes/{catalog}/raw/files/datasets/proteingym/{version}"
staging = f"/Volumes/{catalog}/raw/files/_staging/proteingym_{version}"


def save(df, name, path, comment):
    table = f"{catalog}.bronze.{name}"
    df = (
        df.withColumn("_source_file", F.lit(path))
        .withColumn("_dataset_version", F.lit(version))
        .withColumn("_ingested_at", F.current_timestamp())
    )
    df.createOrReplaceTempView("incoming")
    spark.sql(f"""
        CREATE OR REPLACE TABLE {table}
        TBLPROPERTIES ('delta.columnMapping.mode' = 'name')
        COMMENT '{comment}'
        AS SELECT * FROM incoming
    """)
    print(f"{table}: {spark.table(table).count():,} rows")


# COMMAND ----------

path = f"{source}/DMS_substitutions.csv"
assays = spark.read.option("header", True).option("quote", '"').option("escape", '"').csv(path)
save(assays, "proteingym_assays", path, f"ProteinGym {version} reference table: one row per DMS assay")

# COMMAND ----------

path = f"{source}/DMS_ProteinGym_substitutions.zip"
shutil.rmtree(staging, ignore_errors=True)
with zipfile.ZipFile(path) as z:
    z.extractall(staging)

dms = (
    spark.read.option("header", True)
    .csv(f"{staging}/DMS_ProteinGym_substitutions/*.csv")
    .withColumn("DMS_filename", F.element_at(F.split(F.col("_metadata.file_path"), "/"), -1))
)
save(dms, "proteingym_dms", path, f"ProteinGym {version} DMS substitutions: one row per measured variant per assay")
shutil.rmtree(staging)
