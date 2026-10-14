"""The model registry (chapter 09b): released models get a name, numbered versions and aliases.

The registry is Unity Catalog's: a model lives next to the tables, as ``catalog.schema.model``, with the
same permissions and lineage. Every registration of the same name adds a version (1, 2, 3 ...); versions
are never overwritten. An alias is a movable label on one version: whoever scores the proteome asks for
``@champion`` and gets whichever version currently holds it, so releasing a new model or rolling back is
moving a label, not changing any code.

  register        a logged model (from 09a) becomes a new version of a registered model
  gate_result     the release rules as checked on the test proteins (09a's test run)
  promote         move an alias to a version
  load            the model behind an alias
  champions       which version each alias points to
  distributions   the percentile tables stored with the version's run
"""

from __future__ import annotations

import logging

import pandas as pd

from qafi_platform import tracking  # noqa: F401  (sets MLflow's environment quietly)

NAMES = {"stability": "stability_head", "trees": "function_trees"}  # in schema ml of the catalog


def _mlflow(profile: str = "qafi"):
    import mlflow

    logging.getLogger("mlflow").setLevel(logging.ERROR)
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")
    return mlflow


def full_name(model: str, catalog: str = "qafi_dev") -> str:
    """``catalog.ml.<name>`` for a key of ``NAMES``."""
    return f"{catalog}.ml.{NAMES[model]}"


def register(
    model_uri: str, name: str, description: str = "", tags: dict | None = None, profile: str = "qafi"
):
    """Register the logged model ``model_uri`` under ``name``; returns the new version (a string)."""
    mlflow = _mlflow(profile)
    version = mlflow.register_model(model_uri, name).version
    client = mlflow.MlflowClient()
    if description:
        client.update_model_version(name, version, description=description)
    for key, value in (tags or {}).items():
        client.set_model_version_tag(name, version, key, str(value))
    return str(version)


def gate_result(experiment: str, profile: str = "qafi") -> tuple[bool, pd.DataFrame, dict[str, str], str]:
    """The test evaluation: did it pass every release rule, the rules with their evidence, the models it
    tested (name -> model URI) and its run id.

    Raises if the test proteins have not been evaluated yet: without that evidence nothing is released.
    """
    runs = tracking.test_set_opened(experiment, profile)
    if not len(runs):
        raise RuntimeError("no test evaluation recorded yet: run chapter 09a first")
    run = runs.iloc[0]
    mlflow = _mlflow(profile)
    rules = mlflow.load_table("gate.json", run_ids=[run["run_id"]]).drop(columns=["run_id"], errors="ignore")
    tested = {k.removeprefix("params.model_"): v for k, v in run.items() if k.startswith("params.model_")}
    return bool(run["metrics.gate_passed"] == 1.0), rules.set_index("rule"), tested, run["run_id"]


def promote(name: str, version: str, alias: str = "champion", profile: str = "qafi") -> None:
    """Put ``alias`` on ``version``; the version that held it before loses it."""
    _mlflow(profile).MlflowClient().set_registered_model_alias(name, alias, version)


def versions(name: str, profile: str = "qafi") -> pd.DataFrame:
    """Every version of ``name``: number, aliases, the run that produced it, tags, creation time."""
    client = _mlflow(profile).MlflowClient()
    rows = []
    for v in client.search_model_versions(f"name = '{name}'"):
        detail = client.get_model_version(name, v.version)
        rows.append(
            {
                "version": int(v.version),
                "aliases": ", ".join(detail.aliases or []),
                "run_id": detail.run_id,
                "created": pd.Timestamp(detail.creation_timestamp, unit="ms"),
                **{f"tag {k}": val for k, val in (detail.tags or {}).items()},
            }
        )
    return pd.DataFrame(rows).sort_values("version", ascending=False, ignore_index=True)


def load(name: str, alias: str = "champion", flavor: str = "pyfunc", profile: str = "qafi"):
    """The model that ``alias`` points to, loaded with ``flavor`` (pytorch, xgboost, pyfunc)."""
    mlflow = _mlflow(profile)
    return getattr(mlflow, flavor).load_model(f"models:/{name}@{alias}")


def champions(alias: str = "champion", catalog: str = "qafi_dev", profile: str = "qafi") -> dict[str, str]:
    """The version ``alias`` points to for each released model, by key of ``NAMES`` (kept with the scores)."""
    client = _mlflow(profile).MlflowClient()
    return {
        key: str(client.get_model_version_by_alias(full_name(key, catalog), alias).version) for key in NAMES
    }


def distributions(name: str, alias: str = "champion", profile: str = "qafi") -> dict[str, list[float]]:
    """The percentile tables stored with the run that produced the version ``alias`` points to."""
    mlflow = _mlflow(profile)
    run_id = mlflow.MlflowClient().get_model_version_by_alias(name, alias).run_id
    return mlflow.artifacts.load_dict(f"runs:/{run_id}/distributions.json")
