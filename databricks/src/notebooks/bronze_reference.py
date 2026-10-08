# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze: reference data used to check and fix the predictions
# MAGIC
# MAGIC - ClinVar variant summary (clinical labels; used for evaluation in later chapters)
# MAGIC - UniProt human reference proteome: one record per protein (accession, entry name, gene name, sequence)
# MAGIC - UniProt ID mapping: accession to other identifiers, such as the NCBI GeneID
# MAGIC
# MAGIC Values and column names are kept as delivered. ClinVar's column names contain spaces and brackets
# MAGIC (`RS# (dbSNP)`), which Delta only accepts with column mapping, so these tables turn it on.

# COMMAND ----------

import os
import sys

from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, IntegerType, StringType, StructField, StructType

sys.path.insert(0, os.path.abspath(".."))
from lakehouse.uniprot import read_fasta

dbutils.widgets.text("catalog", "qafi_dev")
dbutils.widgets.text("clinvar_version", "2026-10")
dbutils.widgets.text("uniprot_version", "2026_03")
catalog = dbutils.widgets.get("catalog")
clinvar_version = dbutils.widgets.get("clinvar_version")
uniprot_version = dbutils.widgets.get("uniprot_version")
volume = f"/Volumes/{catalog}/raw/files/datasets"


def lineage(df, path, version):
    return (
        df.withColumn("_source_file", F.lit(path))
        .withColumn("_dataset_version", F.lit(version))
        .withColumn("_ingested_at", F.current_timestamp())
    )


def save(df, name, comment):
    table = f"{catalog}.bronze.{name}"
    df.createOrReplaceTempView("incoming")
    spark.sql(f"""
        CREATE OR REPLACE TABLE {table}
        TBLPROPERTIES ('delta.columnMapping.mode' = 'name')
        COMMENT '{comment}'
        AS SELECT * FROM incoming
    """)
    print(f"{table}: {spark.table(table).count():,} rows")


# COMMAND ----------

path = f"{volume}/clinvar/{clinvar_version}/variant_summary_{clinvar_version}.txt.gz"
clinvar = (
    spark.read.option("header", True)
    .option("sep", "\t")
    .option("quote", "\u0000")  # tab-separated, no quoting
    .csv(path)
)
save(lineage(clinvar, path, clinvar_version), "clinvar_variant_summary", f"ClinVar variant_summary {clinvar_version}")

# COMMAND ----------

path = f"{volume}/uniprot-human/{uniprot_version}/UP000005640_9606.fasta.gz"
schema = StructType(
    [
        StructField("accession", StringType()),
        StructField("entry_name", StringType()),
        StructField("reviewed", BooleanType()),
        StructField("protein_name", StringType()),
        StructField("gene_name", StringType()),
        StructField("length", IntegerType()),
        StructField("sequence", StringType()),
    ]
)
proteome = spark.createDataFrame(read_fasta(path), schema)
save(lineage(proteome, path, uniprot_version), "uniprot_proteome", f"UniProt human reference proteome {uniprot_version}")

# COMMAND ----------

path = f"{volume}/uniprot-human/{uniprot_version}/UP000005640_9606.idmapping.gz"
idmapping = spark.read.option("sep", "\t").csv(path).toDF("accession", "id_type", "id")
save(lineage(idmapping, path, uniprot_version), "uniprot_idmapping", f"UniProt ID mapping {uniprot_version}")
