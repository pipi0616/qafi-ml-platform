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
dbutils.widgets.text("limit", "0")  # proteins to process, for a trial run; 0 = all
catalog = dbutils.widgets.get("catalog")
limit = int(dbutils.widgets.get("limit"))

models = spark.sql(f"""
    SELECT m.uniprot_id, m.fragment, m.pdb_gz, u.sequence
    FROM {catalog}.bronze.alphafold_models m
    JOIN {catalog}.bronze.uniprot_proteome u ON u.accession = m.uniprot_id
""")
if limit:
    chosen = models.select("uniprot_id").distinct().orderBy("uniprot_id").limit(limit)
    models = models.join(chosen, "uniprot_id")

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
        return pd.DataFrame([{"uniprot_id": uniprot_id, "status": status}]).reindex(columns=OUTPUT)
    return features.assign(uniprot_id=uniprot_id, status=status)[OUTPUT]


table = f"{catalog}.silver.residue_structure"
(
    models.groupBy("uniprot_id")
    .applyInPandas(one_protein, schema)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .clusterBy("uniprot_id")
    .saveAsTable(table)
)

# COMMAND ----------

proteins = f"{catalog}.silver.alphafold_proteins"
spark.sql(f"""
    CREATE OR REPLACE TABLE {proteins}
    COMMENT 'AlphaFold proteins: residues with structure features, or why there are none (notebook 07d)'
    AS SELECT uniprot_id, first(status) AS status, count(position) AS residues,
              count(DISTINCT fragment) AS fragments, round(avg(plddt), 1) AS mean_plddt
    FROM {table} GROUP BY uniprot_id
""")
spark.sql(f"DELETE FROM {table} WHERE position IS NULL")
spark.sql(f"COMMENT ON TABLE {table} IS 'Per-residue AlphaFold structure features on UniProt numbering (notebook 07d)'")
display(spark.sql(f"SELECT status, count(*) AS proteins, sum(residues) AS residues FROM {proteins} GROUP BY status"))
