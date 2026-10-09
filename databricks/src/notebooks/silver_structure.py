# Databricks notebook source
# MAGIC %pip install biopython==1.88

# COMMAND ----------

# MAGIC %md
# MAGIC # Silver: per-residue structure features of every human protein (notebook 07d)
# MAGIC
# MAGIC Each protein's AlphaFold model(s) go through the tested functions in `lakehouse/structure.py`
# MAGIC (`groupBy("uniprot_id").applyInPandas`): pLDDT, solvent accessibility, contact counts, half-sphere
# MAGIC exposure, C-alpha coordinates and the neighbours within 10 Å. Long proteins are stitched from their
# MAGIC fragments. Features are computed for the whole proteome, so training and proteome-wide prediction use
# MAGIC exactly the same values.
# MAGIC
# MAGIC Incremental: each run computes only proteins not done yet, in batches that are saved one by one, so an
# MAGIC interrupted run resumes where it stopped and a repeated run has nothing to do. `rebuild=true` starts
# MAGIC over (a new AlphaFold version).
# MAGIC - `silver.residue_structure`: one row per residue (UniProt numbering)
# MAGIC - `silver.alphafold_proteins`: one row per protein: ok, or why it has no features

# COMMAND ----------

import gzip
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(".."))
from lakehouse.structure import COLUMNS, protein_features

dbutils.widgets.text("catalog", "qafi_dev")
dbutils.widgets.text("limit", "0")  # at most this many new proteins, for a trial run; 0 = all
dbutils.widgets.text("batch_size", "2000")  # proteins per batch; each batch is saved before the next starts
dbutils.widgets.text("proteins_per_task", "20")  # small tasks keep every core busy and let serverless scale out
dbutils.widgets.text("rebuild", "false")  # true: forget earlier results and compute everything again
catalog = dbutils.widgets.get("catalog")
limit = int(dbutils.widgets.get("limit"))
batch_size = int(dbutils.widgets.get("batch_size"))
proteins_per_task = int(dbutils.widgets.get("proteins_per_task"))
table = f"{catalog}.silver.residue_structure"
proteins = f"{catalog}.silver.alphafold_proteins"
batch = f"{catalog}.silver.residue_structure_batch"

if dbutils.widgets.get("rebuild").lower() == "true":
    spark.sql(f"DROP TABLE IF EXISTS {table}")
    spark.sql(f"DROP TABLE IF EXISTS {proteins}")
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {table} (
        uniprot_id string, position int, aa string, plddt double, sasa double, rsa double,
        contacts_8 int, contacts_12 int, hse_up int, hse_down int, ca_x double, ca_y double, ca_z double,
        neighbours array<int>, distances array<double>, fragment int)
    CLUSTER BY (uniprot_id)
    COMMENT 'Per-residue AlphaFold structure features on UniProt numbering (notebook 07d)'
""")
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {proteins}
        (uniprot_id string, status string, residues bigint, fragments bigint, mean_plddt double)
    COMMENT 'AlphaFold proteins: residues with structure features, or why there are none (notebook 07d)'
""")
models = spark.sql(f"""
    SELECT m.uniprot_id, m.fragment, m.pdb_gz, u.sequence
    FROM {catalog}.bronze.alphafold_models m
    JOIN {catalog}.bronze.uniprot_proteome u ON u.accession = m.uniprot_id
""")
todo = [
    row.uniprot_id
    for row in models.select("uniprot_id").distinct().join(spark.table(proteins), "uniprot_id", "left_anti")
    .orderBy("uniprot_id").collect()
]
todo = todo[:limit] if limit else todo
print(f"{len(todo):,} proteins to compute")

OUTPUT = ["uniprot_id", "status", *COLUMNS]
schema = (
    "uniprot_id string, status string, position int, aa string, plddt double, sasa double, rsa double, "
    "contacts_8 int, contacts_12 int, hse_up int, hse_down int, ca_x double, ca_y double, ca_z double, "
    "neighbours array<int>, distances array<double>, fragment int"
)


def one_protein(rows: pd.DataFrame) -> pd.DataFrame:
    uniprot_id = rows["uniprot_id"].iloc[0]
    fragments = [(int(f), gzip.decompress(b).decode()) for f, b in zip(rows["fragment"], rows["pdb_gz"], strict=True)]
    try:
        features, status = protein_features(fragments, rows["sequence"].iloc[0])
    except Exception as error:  # one unreadable model must not stop the other twenty thousand
        features, status = pd.DataFrame(columns=COLUMNS), f"error: {type(error).__name__}"
    if features.empty:  # keep the protein, with the reason, so it shows up in alphafold_proteins
        return pd.DataFrame([{c: None for c in OUTPUT} | {"uniprot_id": uniprot_id, "status": status}])
    return features.assign(uniprot_id=uniprot_id, status=status)[OUTPUT]


# COMMAND ----------

# The work is heavy in computation but small in bytes, so Spark on its own would cut it into a few large
# tasks and leave most cores idle. repartition cuts each batch into tasks of about proteins_per_task proteins.
for first in range(0, len(todo), batch_size):
    chunk = todo[first : first + batch_size]
    ids = spark.createDataFrame([(i,) for i in chunk], "uniprot_id string")
    (
        models.join(ids, "uniprot_id")
        .repartition(max(1, -(-len(chunk) // proteins_per_task)), "uniprot_id")
        .groupBy("uniprot_id")
        .applyInPandas(one_protein, schema)
        .write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(batch)
    )
    spark.sql(f"INSERT INTO {table} SELECT * EXCEPT (status) FROM {batch} WHERE position IS NOT NULL")
    spark.sql(f"""
        INSERT INTO {proteins}
        SELECT uniprot_id, first(status), count(position), count(DISTINCT fragment), round(avg(plddt), 1)
        FROM {batch} GROUP BY uniprot_id
    """)
    print(f"{min(first + batch_size, len(todo)):,} of {len(todo):,} proteins done")
spark.sql(f"DROP TABLE IF EXISTS {batch}")
display(spark.sql(f"SELECT status, count(*) AS proteins, sum(residues) AS residues FROM {proteins} GROUP BY status"))
