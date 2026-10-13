# Databricks notebook source
# MAGIC %pip install transformers==5.19.0

# COMMAND ----------

# MAGIC %md
# MAGIC # Silver: ESM-2 language-model features of every human protein (notebook 07e)
# MAGIC
# MAGIC Runs on a serverless GPU (A10). ESM-2 (650 million parameters) reads every protein of the UniProt
# MAGIC reference proteome once, through the tested functions in `lakehouse/language_model.py`:
# MAGIC - `silver.residue_language_model`: every residue's log-probabilities of the 20 amino acids, in the
# MAGIC   order ACDEFGHIKLMNPQRSTVWY. A substitution's zero-shot score is
# MAGIC   logprobs[mutant] - logprobs[wild type].
# MAGIC - `silver.residue_embedding`: the last layer's 1,280 numbers for every residue of the labelled
# MAGIC   proteins, the input of the neural networks in notebook 08 (the whole proteome would take 60 GB)
# MAGIC - `silver.language_model_proteins`: one row per protein: the sequence and model version that were read
# MAGIC
# MAGIC Incremental: a protein is read again only when its sequence or the model changes, or when it has been
# MAGIC labelled since and has no embeddings yet. Each batch replaces its proteins' rows and then marks them
# MAGIC done, so an interrupted run resumes cleanly. `rebuild=true` starts over.
# MAGIC
# MAGIC It needs a GPU quota. Free Edition has none, so there the proteins are read on a laptop and loaded by
# MAGIC the `load_language_model` task, which runs before this one; this task then reads only what is missing.

# COMMAND ----------

import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(".."))
from lakehouse import language_model as lm

dbutils.widgets.text("catalog", "qafi_dev")
dbutils.widgets.text("model_version", lm.REVISION)  # the Hugging Face commit, also the model's folder in raw
dbutils.widgets.text("limit", "0")  # at most this many proteins, for a trial run; 0 = all
dbutils.widgets.text("batch_size", "1000")  # proteins per batch; each batch is saved before the next starts
dbutils.widgets.text("tokens_per_batch", "16384")  # amino acids the GPU reads at once
dbutils.widgets.text("rebuild", "false")  # true: forget earlier results and read every protein again
catalog = dbutils.widgets.get("catalog")
version = dbutils.widgets.get("model_version")
limit = int(dbutils.widgets.get("limit"))
batch_size = int(dbutils.widgets.get("batch_size"))
tokens_per_batch = int(dbutils.widgets.get("tokens_per_batch"))
residues = f"{catalog}.silver.residue_language_model"
embeddings = f"{catalog}.silver.residue_embedding"
proteins = f"{catalog}.silver.language_model_proteins"
staging = f"/Volumes/{catalog}/raw/files/_staging/language_model"

if dbutils.widgets.get("rebuild").lower() == "true":
    for table in (residues, embeddings, proteins):
        spark.sql(f"DROP TABLE IF EXISTS {table}")
for statement in lm.table_ddl(catalog):
    spark.sql(statement)


def replace(table: str, name: str) -> None:
    """The batch's proteins lose their rows in ``table`` and get the ones staged under ``name``: running it
    twice gives the same table."""
    batch = f"SELECT uniprot_id FROM parquet.`{staging}/proteins`"
    spark.sql(f"DELETE FROM {table} WHERE uniprot_id IN ({batch})")
    if os.path.exists(f"{staging}/{name}"):
        spark.sql(f"INSERT INTO {table} SELECT * FROM parquet.`{staging}/{name}`")


todo = spark.sql(f"""
    SELECT p.accession AS uniprot_id, p.sequence, l.uniprot_id IS NOT NULL AS labelled
    FROM {catalog}.bronze.uniprot_proteome p
    LEFT JOIN (SELECT DISTINCT uniprot_id FROM {catalog}.gold.fct_variant_label) l
        ON l.uniprot_id = p.accession
    LEFT JOIN {proteins} d
        ON d.uniprot_id = p.accession AND d.sequence_md5 = md5(p.sequence) AND d.model_version = '{version}'
    WHERE d.uniprot_id IS NULL OR (l.uniprot_id IS NOT NULL AND NOT d.embedded)
    ORDER BY p.accession
""").toPandas()
todo = todo.head(limit) if limit else todo
print(f"{len(todo):,} proteins to read, {todo['labelled'].sum():,} of them labelled")
if len(todo):
    started = time.time()
    model, tokenizer = lm.load(f"/Volumes/{catalog}/raw/files/datasets/esm2-650m/{version}", revision=None)
    print(f"model loaded on {next(model.parameters()).device} in {time.time() - started:.0f} s")

# COMMAND ----------

started = time.time()
for first in range(0, len(todo), batch_size):
    chunk = todo.iloc[first : first + batch_size].reset_index(drop=True)
    results = [None] * len(chunk)
    for wanted, with_vectors in ((chunk["labelled"], True), (~chunk["labelled"], False)):
        index = np.flatnonzero(wanted)
        sequences = chunk["sequence"].iloc[index].tolist()
        read = lm.read_proteins(sequences, model, tokenizer, with_vectors, tokens_per_batch)
        for k, result in zip(index, read, strict=True):
            results[k] = result

    shutil.rmtree(staging, ignore_errors=True)
    for name in ("residues", "embeddings", "proteins"):
        os.makedirs(f"{staging}/{name}")
    ids, sequences = chunk["uniprot_id"].tolist(), chunk["sequence"].tolist()
    lm.residue_frame(ids, sequences, [r[0] for r in results]).to_parquet(f"{staging}/residues/part.parquet")
    labelled = chunk[chunk["labelled"]]
    if len(labelled):
        vectors = [results[k][1] for k in labelled.index]
        frame = lm.residue_frame(labelled["uniprot_id"], labelled["sequence"].tolist(), vectors, "embedding")
        frame.drop(columns="aa").to_parquet(f"{staging}/embeddings/part.parquet")
    else:
        os.rmdir(f"{staging}/embeddings")
    read_at = pd.Timestamp.now(tz="UTC")
    lm.protein_frame(ids, sequences, version, chunk["labelled"]).assign(read_at=read_at).to_parquet(
        f"{staging}/proteins/part.parquet"
    )

    replace(residues, "residues")
    replace(embeddings, "embeddings")
    replace(proteins, "proteins")  # last: only now does the batch count as done
    seconds = time.time() - started
    print(f"{first + len(chunk):,} of {len(todo):,} proteins read ({seconds / 60:.1f} min)")
shutil.rmtree(staging, ignore_errors=True)
display(
    spark.sql(f"""
        SELECT model_version, count(*) AS proteins, sum(residues) AS residues,
               sum(CAST(embedded AS INT)) AS embedded
        FROM {proteins} GROUP BY model_version
    """)
)
