# Databricks notebook source
# MAGIC %md
# MAGIC # Export the released scores to the AWS lake
# MAGIC
# MAGIC Free Edition cannot write to our S3 bucket, so notebook 10a signs one temporary upload link per file
# MAGIC and puts them into the volume (`ml/files/_export/links.json`). This task:
# MAGIC 1. writes `variant_effect` with Spark, cut into ranges of proteins and sorted inside each file, so a
# MAGIC    query for one protein reads one small piece of one file; and `proteins`, one row per protein;
# MAGIC 2. uploads every file to its link;
# MAGIC 3. uploads the manifest last: the files, their rows and sizes, which the laptop checks against S3.

# COMMAND ----------

import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(".."))
from lakehouse import export

dbutils.widgets.text("catalog", "qafi_dev")
dbutils.widgets.text("source", "gold.fct_variant_effect")
catalog, source = dbutils.widgets.get("catalog"), dbutils.widgets.get("source")
folder = f"/Volumes/{catalog}/ml/files/_export"
with open(f"{folder}/links.json") as f:
    plan = json.load(f)
release = plan["release"]

scores = spark.table(f"{catalog}.{source}")
found = sorted(r["release"] for r in scores.select("release").distinct().collect())
if found != [release]:
    raise ValueError(f"{source} holds the releases {found}; the upload links are for {release}")

# COMMAND ----------

# The release goes into the folder name (release=...), not into the files.
columns = [
    "uniprot_id", "gene_name", "entry_name", "position", "ref_aa", "alt_aa", "variant",
    "stability_score", "function_score", "zero_shot", "stability_head_version", "function_trees_version",
]  # fmt: skip
local = f"{folder}/{release}"
shutil.rmtree(local, ignore_errors=True)
(
    scores.select(*columns)
    .repartitionByRange(len(plan["variant_effect"]), "uniprot_id")  # file i holds a range of proteins
    .sortWithinPartitions("uniprot_id", "position", "alt_aa")
    .write.option("compression", "zstd")
    .option("parquet.block.size", 16 * 1024 * 1024)  # row groups of ~16 MB: what one lookup reads
    .parquet(f"{local}/variant_effect")
)
proteins = (
    spark.table(f"{catalog}.gold.dim_protein")
    .join(scores.select("uniprot_id").distinct(), "uniprot_id")
    .select("uniprot_id", "gene_name", "entry_name", "protein_name", "length", "has_dms")
    .orderBy("uniprot_id")
)
proteins.coalesce(1).write.option("compression", "zstd").parquet(f"{local}/proteins")

# COMMAND ----------

parts = sorted(glob.glob(f"{local}/variant_effect/part-*.parquet"))
parts = [p for p in parts if export.describe(p, "")["rows"]]  # a range can come out empty: nothing to upload
if not parts or len(parts) > len(plan["variant_effect"]):
    raise RuntimeError(f"{len(parts)} files written for {len(plan['variant_effect'])} upload links")
files = []
for path, link in zip(parts, plan["variant_effect"], strict=False):  # spare links stay unused
    export.put(path, link["url"])
    files.append(export.describe(path, link["key"]))
    print(f"uploaded {link['key']}: {files[-1]['rows']:,} rows, {files[-1]['bytes'] / 1e6:.0f} MB")
(path,) = glob.glob(f"{local}/proteins/part-*.parquet")
export.put(path, plan["proteins"]["url"])
versions = scores.select("stability_head_version", "function_trees_version").distinct().collect()
written = export.manifest(
    release, f"{catalog}.{source}", files, export.describe(path, plan["proteins"]["key"]),
    [row.asDict() for row in versions],
)  # fmt: skip
export.put(json.dumps(written, indent=1).encode(), plan["manifest"]["url"], "application/json")  # last
shutil.rmtree(local)
print(f"release {release}: {written['variants']:,} variants of {written['proteins']:,} proteins in S3")
