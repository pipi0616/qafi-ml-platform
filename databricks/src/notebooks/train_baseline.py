# Databricks notebook source
# MAGIC %pip install xgboost==3.4.1

# COMMAND ----------

# MAGIC %md
# MAGIC # The baseline model: cross-validated, every run recorded in MLflow (notebook 07f)
# MAGIC
# MAGIC Reads the training side of `gold.fct_variant_features` and, for each run named in `runs`
# MAGIC (`qafi_core.baseline.RUNS`), cross-validates gradient-boosted trees with proteins held out whole.
# MAGIC The science is in `qafi_core.baseline`, tested on a laptop; this notebook only feeds it and records
# MAGIC what comes back.
# MAGIC
# MAGIC Each run becomes one MLflow run with everything needed to redo and to compare it:
# MAGIC - **parameters**: features, kinds of data trained on, model settings, the version of the feature table
# MAGIC - **metrics**: median Spearman over datasets, overall and per kind of data, and the paired comparison
# MAGIC   with ESM-2's zero-shot score
# MAGIC - **artifacts**: the score of every dataset; for `baseline`, feature importances and the model itself,
# MAGIC   trained on all training proteins
# MAGIC
# MAGIC Tables, replaced per run so a repeated run changes nothing:
# MAGIC - `ml.cv_scores`: one row per run per dataset
# MAGIC - `ml.cv_predictions`: the baseline's held-out prediction of every training variant
# MAGIC
# MAGIC Test proteins are never read.

# COMMAND ----------

import os
import sys

import mlflow
import pandas as pd

sys.path.insert(0, os.path.abspath("../../../src"))  # qafi_core: the science code, copied here by the bundle
from qafi_core import baseline, features

dbutils.widgets.text("catalog", "qafi_dev")
dbutils.widgets.text("experiment", "")  # the MLflow experiment's workspace path (the bundle creates it)
dbutils.widgets.text("runs", "all")  # names from qafi_core.baseline.RUNS, comma-separated; all = every one
dbutils.widgets.text("n_folds", "5")
dbutils.widgets.text("code_version", "")  # git commit the bundle was deployed from
catalog = dbutils.widgets.get("catalog")
runs = dbutils.widgets.get("runs")
n_folds = int(dbutils.widgets.get("n_folds"))
names = list(baseline.RUNS) if runs == "all" else [name.strip() for name in runs.split(",")]

table = f"{catalog}.gold.fct_variant_features"
version = spark.sql(f"DESCRIBE HISTORY {table} LIMIT 1").first()["version"]
train = spark.table(table).where("split = 'train'").toPandas()
print(
    f"{table} version {version}: {len(train):,} training variants, {train.uniprot_id.nunique()} proteins, "
    f"{train.dataset_id.nunique()} datasets, {train.similarity_group.nunique()} groups of relatives"
)

# COMMAND ----------

scores_table = f"{catalog}.ml.cv_scores"
predictions_table = f"{catalog}.ml.cv_predictions"
spark.sql(f"""CREATE TABLE IF NOT EXISTS {scores_table}
    (run_name string, mlflow_run_id string, dataset_id string, data_kind string, uniprot_id string,
     variants bigint, spearman double)
    COMMENT 'Cross-validation score of every dataset, per model run (notebook 07f)'""")
spark.sql(f"""CREATE TABLE IF NOT EXISTS {predictions_table}
    (run_name string, dataset_id string, uniprot_id string, uniprot_position int, ref_aa string,
     alt_aa string, fold int, predicted double)
    COMMENT 'Held-out prediction of every training variant by the baseline (notebook 07f)'""")


def save(frame: pd.DataFrame, target: str, run_name: str) -> None:
    """Replace this run's rows in ``target``."""
    spark.sql(f"DELETE FROM {target} WHERE run_name = '{run_name}'")
    columns = [field.name for field in spark.table(target).schema]
    spark.createDataFrame(frame.assign(run_name=run_name)[columns]).write.mode("append").saveAsTable(target)


def record(run_name: str, scores: pd.DataFrame, reference: pd.DataFrame | None) -> None:
    """What every run records: where the data came from, its scores, and how it compares."""
    mlflow.set_tags({"chapter": "07f", "code_version": dbutils.widgets.get("code_version")})
    mlflow.log_params(
        {
            "feature_table": table,
            "feature_table_version": version,
            "n_folds": n_folds,
            "proteins": train.uniprot_id.nunique(),
            "variants": len(train),
        }
    )
    mlflow.log_metrics(baseline.summarise(scores))
    if reference is not None:
        mlflow.log_metrics({f"vs_zero_shot_{k}": v for k, v in baseline.compare(scores, reference).items()})
    mlflow.log_table(scores.round(4), "dataset_scores.json")
    save(scores.assign(mlflow_run_id=mlflow.active_run().info.run_id), scores_table, run_name)


mlflow.set_experiment(dbutils.widgets.get("experiment"))

# The bar: ESM-2's zero-shot score alone, no training. Every trained model is compared with it.
zero_shot = baseline.dataset_scores(train, "esm_delta")
with mlflow.start_run(run_name="esm2_zero_shot"):
    mlflow.log_params({"model": "none: esm_delta as it is", "features": "esm_delta"})
    record("esm2_zero_shot", zero_shot, None)
print("esm2_zero_shot", baseline.summarise(zero_shot))

# COMMAND ----------

for name in names:
    config = baseline.RUNS[name]
    columns = features.select(config.get("features", "all"))
    train_on = config["train_on"].split(",") if "train_on" in config else None
    with mlflow.start_run(run_name=name):
        mlflow.log_params(
            {
                "model": "xgboost",
                "feature_groups": config.get("features", "all"),
                "features": ",".join(columns),
                "train_on": config.get("train_on", "all"),
                **baseline.PARAMS,
            }
        )
        cv = baseline.cross_validate(train, columns, n_folds=n_folds, train_on=train_on)
        scores = baseline.dataset_scores(cv)
        record(name, scores, zero_shot)
        if name == "baseline":
            # The model others will load: trained on every training protein, with its inputs described.
            model = baseline.fit(train, columns)
            mlflow.log_table(baseline.importance(model, columns), "feature_importance.json")
            mlflow.xgboost.log_model(
                model,
                name="model",
                signature=mlflow.models.infer_signature(train[columns], model.predict(train[columns].head(100))),
                input_example=train[columns].head(5),
            )
            save(cv[[*baseline.KEY, "ref_aa", "fold", "predicted"]], predictions_table, name)
    print(name, baseline.summarise(scores))

# COMMAND ----------

display(
    spark.sql(f"""
    SELECT run_name, data_kind, count(*) AS datasets, round(median(spearman), 3) AS median_spearman
    FROM {scores_table} GROUP BY run_name, data_kind ORDER BY data_kind, median_spearman DESC""")
)
