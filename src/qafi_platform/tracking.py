"""Recording model runs made outside Databricks (chapter 08): a laptop's GPU trains, the platform remembers.

A run trained on a laptop leaves the same records as the Databricks training job of chapter 07f, in the
same places, so the two can be compared side by side:
  - an MLflow run in the workspace's experiment: parameters, metrics, tables, the code version
  - its score for every dataset in ``ml.cv_scores``
  - its held-out predictions in ``ml.cv_predictions``

``w`` is a ``databricks.sdk.WorkspaceClient``; MLflow talks to the same workspace through the CLI profile.
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
from contextlib import contextmanager

import numpy as np
import pandas as pd

from qafi_core import baseline
from qafi_platform.lakehouse import sql

# MLflow prints hints and advertisements for features we do not use; keep a notebook's output to results.
for _name in (
    "MLFLOW_DISABLE_AGENT_HINT=1",
    "MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false",
    "_MLFLOW_ENABLE_UC_TRACE_UPSELL=false",
):
    os.environ.setdefault(*_name.split("="))

PREDICTION_COLUMNS = ["dataset_id", "uniprot_id", "uniprot_position", "ref_aa", "alt_aa", "fold", "predicted"]


def code_version() -> str:
    """The git commit of the code being run, marked ``-dirty`` when files differ from that commit."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    changed = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    return commit + ("-dirty" if changed else "")


def tensor_signature(inputs: dict[str, int], outputs: int):
    """What a PyTorch model takes and gives, in MLflow's terms: named inputs with their widths.

    Stored with the model, it lets MLflow check inputs when the model is loaded, and the Unity Catalog
    registry (chapter 09) only accepts models that have one.
    """
    from mlflow.models import ModelSignature
    from mlflow.types import Schema, TensorSpec

    return ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("float32"), (-1, width), name) for name, width in inputs.items()]),
        outputs=Schema([TensorSpec(np.dtype("float32"), (-1, outputs))]),
    )


def read_scores(w, run_name: str, catalog: str = "qafi_dev") -> pd.DataFrame:
    """A run's cross-validation score of every dataset, from ``ml.cv_scores``."""
    scores = sql(
        w,
        f"""SELECT dataset_id, data_kind, uniprot_id, variants, spearman
            FROM {catalog}.ml.cv_scores WHERE run_name = '{run_name}'""",
    )
    return scores.astype({"variants": int, "spearman": float})


def save_scores(w, scores: pd.DataFrame, run_name: str, run_id: str, catalog: str = "qafi_dev") -> None:
    """Replace the run's rows in ``ml.cv_scores``."""
    rows = ", ".join(
        f"('{run_name}', '{run_id}', '{r.dataset_id}', '{r.data_kind}', '{r.uniprot_id}', "
        f"{r.variants}, {r.spearman})"
        for r in scores.itertuples()
    )
    sql(w, f"DELETE FROM {catalog}.ml.cv_scores WHERE run_name = '{run_name}'")
    sql(w, f"INSERT INTO {catalog}.ml.cv_scores VALUES {rows}")


def save_predictions(w, predictions: pd.DataFrame, run_name: str, catalog: str = "qafi_dev") -> int:
    """Replace the run's rows in ``ml.cv_predictions``; returns the number of rows now in the table.

    Too many rows for an INSERT statement, so they travel as a parquet file: uploaded to the volume
    ``ml.files``, then read into the table by the warehouse.
    """
    path = f"/Volumes/{catalog}/ml/files/cv_predictions/{run_name}.parquet"
    data = io.BytesIO()
    predictions[PREDICTION_COLUMNS].to_parquet(data, index=False)
    data.seek(0)
    w.files.upload(path, data, overwrite=True)
    table = f"{catalog}.ml.cv_predictions"
    sql(w, f"DELETE FROM {table} WHERE run_name = '{run_name}'")
    sql(
        w,
        f"""INSERT INTO {table}
            SELECT '{run_name}', dataset_id, uniprot_id, CAST(uniprot_position AS INT), ref_aa, alt_aa,
                   CAST(fold AS INT), CAST(predicted AS DOUBLE)
            FROM read_files('{path}', format => 'parquet')""",
    )
    return int(sql(w, f"SELECT count(*) AS n FROM {table} WHERE run_name = '{run_name}'")["n"][0])


def latest_runs(experiment: str, profile: str = "qafi") -> pd.DataFrame:
    """The newest MLflow run of each name in ``experiment``, one row per run, best overall score first.

    The four score columns are renamed all, domainome, proteingym and mavedb; every other MLflow column
    (params.*, metrics.*, run_id) is kept. MLflow never overwrites: a name that was run twice has two runs,
    and this takes the later one.
    """
    import mlflow

    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_experiment(experiment)
    runs = mlflow.search_runs(order_by=["start_time DESC"]).drop_duplicates("tags.mlflow.runName")
    short = {"metrics.spearman_median": "all"} | {
        f"metrics.spearman_median_{kind}": kind for kind in ("domainome", "proteingym", "mavedb")
    }
    runs = runs.rename(columns=short).set_index("tags.mlflow.runName").rename_axis("run")
    return runs.sort_values("all", ascending=False)


def comparison(runs: pd.DataFrame, run_name: str, reference: str) -> pd.DataFrame:
    """The paired comparison of ``run_name`` with ``reference`` logged with the run, per kind of data.

    better = share of datasets where the run scores higher, difference = its mean gain in Spearman,
    p_value = how likely such gains are if the two were equal.
    """
    prefix = f"metrics.vs_{reference}_"
    logged = runs.loc[run_name].filter(like=prefix).rename(lambda name: name.replace(prefix, ""))
    columns = {
        what: logged.filter(like=what).rename(lambda name, what=what: name.replace(what + "_", ""))
        for what in ("better", "difference", "p_value")
    }
    return pd.DataFrame(columns).astype(float).round(3)


def cv_run(
    w,
    experiment: str,
    run_name: str,
    params: dict,
    scores: pd.DataFrame,
    references: dict[str, pd.DataFrame] | None = None,
    tables: dict[str, pd.DataFrame] | None = None,
    model=None,
    signature=None,
    chapter: str = "",
    catalog: str = "qafi_dev",
    profile: str = "qafi",
) -> str:
    """Record one cross-validated run in MLflow and in ``ml.cv_scores``; returns the MLflow run id.

    ``references`` are other runs' scores to compare with, paired by dataset: each gives the metrics
    ``vs_<name>_better_<kind>``, ``..._difference_...`` and ``..._p_value_...``. ``tables`` are logged as
    artifacts; ``model`` is a trained PyTorch module, logged with its ``signature`` so it can be loaded
    again later.
    """
    import mlflow

    logging.getLogger("mlflow").setLevel(logging.ERROR)  # its progress chatter hides the two links it prints
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"chapter": chapter, "code_version": code_version(), "trained_on": "laptop"})
        mlflow.log_params(params)
        mlflow.log_metrics(baseline.summarise(scores))
        for name, reference in (references or {}).items():
            comparison = baseline.compare(scores, reference)
            mlflow.log_metrics({f"vs_{name}_{key}": value for key, value in comparison.items()})
        mlflow.log_table(scores.round(4), "dataset_scores.json")
        for name, table in (tables or {}).items():
            mlflow.log_table(table, f"{name}.json")
        if model is not None:
            mlflow.pytorch.log_model(model, name="model", signature=signature)
        save_scores(w, scores, run_name, run.info.run_id, catalog)
    return run.info.run_id


def _mlflow(experiment: str, profile: str):
    import mlflow

    logging.getLogger("mlflow").setLevel(logging.ERROR)
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_experiment(experiment)
    return mlflow


@contextmanager
def tuning_run(experiment: str, run_name: str, chapter: str = "", profile: str = "qafi"):
    """A parent MLflow run for a hyperparameter search; yields a function that records one trial.

    Each trial becomes a child run (nested), so the search shows as one row that opens into its trials.
    """
    mlflow = _mlflow(experiment, profile)
    with mlflow.start_run(run_name=run_name) as parent:
        mlflow.set_tags({"chapter": chapter, "code_version": code_version(), "trained_on": "laptop"})

        def log_trial(number: int, settings: dict, summary: dict) -> None:
            with mlflow.start_run(run_name=f"trial_{number:02d}", nested=True):
                mlflow.log_params(settings)
                mlflow.log_metrics(summary)

        yield parent, log_trial


def model_run(
    experiment: str,
    run_name: str,
    params: dict,
    model,
    flavor: str,
    signature=None,
    dicts: dict[str, dict] | None = None,
    chapter: str = "",
    profile: str = "qafi",
) -> str:
    """Record a final model (trained on all training proteins) with its settings; returns its model URI.

    ``flavor`` is the MLflow flavor that saves it (pytorch, xgboost); ``dicts`` are small JSON files
    stored next to it, such as the training distributions that turn outputs into percentiles.
    """
    mlflow = _mlflow(experiment, profile)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"chapter": chapter, "code_version": code_version(), "final_model": "true"})
        mlflow.log_params(params)
        logged = getattr(mlflow, flavor).log_model(model, name="model", signature=signature)
        for name, content in (dicts or {}).items():
            mlflow.log_dict(content, f"{name}.json")
    return logged.model_uri  # the address 09b registers from


def experiment_run(
    experiment: str,
    run_name: str,
    params: dict,
    metrics: dict,
    tables: dict[str, pd.DataFrame] | None = None,
    chapter: str = "",
    profile: str = "qafi",
) -> str:
    """Record one run that is not a cross-validation: its settings, its numbers and its result tables.

    ``cv_run`` expects a score per labelled dataset and writes ``ml.cv_scores``; the stability chapter
    (14a) scores domains of another dataset entirely, so it records here instead. Everything still lands
    in the same MLflow experiment, with the same code-version tag, so runs stay comparable and findable.
    """
    mlflow = _mlflow(experiment, profile)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"chapter": chapter, "code_version": code_version(), "trained_on": "laptop"})
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for name, table in (tables or {}).items():
            mlflow.log_table(table, f"{name}.json")
    return run.info.run_id


def test_set_opened(experiment: str, profile: str = "qafi") -> pd.DataFrame:
    """Runs that already read the locked test proteins (tag test_set = opened); empty if none did."""
    mlflow = _mlflow(experiment, profile)
    return mlflow.search_runs(filter_string="tags.test_set = 'opened'")


def test_run(
    experiment: str,
    per_model: dict[str, pd.DataFrame],
    gate: pd.DataFrame,
    models: dict[str, str],
    chapter: str = "",
    force: bool = False,
    profile: str = "qafi",
) -> str:
    """Record the one evaluation on the test proteins; refuses if one was recorded before.

    The test proteins answer one question once. Looking again after changing something would let the
    test results steer the model, which is what the locked split exists to prevent; ``force`` exists only
    for deliberate, documented exceptions. ``models`` names the logged models that were tested (their
    URIs), so that the registry (09b) can register exactly those.
    """
    earlier = test_set_opened(experiment, profile)
    if len(earlier) and not force:
        raise RuntimeError(f"the test set was already opened by run {earlier['run_id'].iloc[0]}")
    mlflow = _mlflow(experiment, profile)
    with mlflow.start_run(run_name="test_evaluation") as run:
        mlflow.set_tags({"chapter": chapter, "code_version": code_version(), "test_set": "opened"})
        mlflow.log_params({f"model_{name}": uri for name, uri in models.items()})
        for name, scores in per_model.items():
            mlflow.log_metrics({f"{name}_{k}": v for k, v in baseline.summarise(scores).items()})
            mlflow.log_table(scores.round(4), f"test_scores_{name}.json")
        mlflow.log_table(gate.reset_index(), "gate.json")
        mlflow.log_metric("gate_passed", float(gate["passed"].all()))
    return run.info.run_id
