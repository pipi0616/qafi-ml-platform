import sys
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from qafi_platform import registry, tracking


@pytest.fixture
def fake(monkeypatch):
    """MLflow's registry, kept in a dict: versions per name, tags, aliases."""
    state = {"versions": {}, "aliases": {}, "tags": {}, "descriptions": {}}

    class Client:
        def update_model_version(self, name, version, description):
            state["descriptions"][name, version] = description

        def set_model_version_tag(self, name, version, key, value):
            state["tags"][name, version, key] = value

        def set_registered_model_alias(self, name, alias, version):
            state["aliases"][name, alias] = version

        def get_model_version_by_alias(self, name, alias):
            version = state["aliases"][name, alias]
            return NS(run_id=f"run-of-{version}", version=version)

    def register_model(uri, name):
        state["versions"].setdefault(name, []).append(uri)
        return NS(version=len(state["versions"][name]))

    mlflow = NS(
        set_tracking_uri=lambda uri: None, set_registry_uri=lambda uri: state.update(registry=uri),
        register_model=register_model, MlflowClient=Client,
        artifacts=NS(load_dict=lambda uri: {"from": uri}),
        load_table=lambda name, run_ids: pd.DataFrame({"rule": ["r"], "passed": [True], "run_id": run_ids}),
    )  # fmt: skip
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    return state


def test_each_registration_adds_a_version_with_its_tags(fake):
    assert registry.register("runs:/a/model", "qafi_dev.ml.x", "first", {"gate": "passed"}) == "1"
    assert registry.register("runs:/b/model", "qafi_dev.ml.x") == "2"
    assert fake["registry"] == "databricks-uc://qafi"
    assert (
        fake["tags"]["qafi_dev.ml.x", 1, "gate"] == "passed"
        and fake["descriptions"]["qafi_dev.ml.x", 1] == "first"
    )


def test_an_alias_moves_and_the_percentile_tables_follow_it(fake):
    registry.promote("qafi_dev.ml.x", "1")
    registry.promote("qafi_dev.ml.x", "2")
    assert fake["aliases"]["qafi_dev.ml.x", "champion"] == "2"
    assert registry.distributions("qafi_dev.ml.x") == {"from": "runs:/run-of-2/distributions.json"}
    assert registry.full_name("trees") == "qafi_dev.ml.function_trees"
    registry.promote(registry.full_name("stability"), "3")
    registry.promote(registry.full_name("trees"), "1")
    assert registry.champions() == {"stability": "3", "trees": "1"}


def test_nothing_is_released_without_a_test_evaluation(fake, monkeypatch):
    monkeypatch.setattr(tracking, "test_set_opened", lambda experiment, profile="qafi": pd.DataFrame())
    with pytest.raises(RuntimeError, match="no test evaluation"):
        registry.gate_result("/exp")
    run = {"run_id": "t1", "metrics.gate_passed": 1.0, "params.model_stability": "models:/m-1", "tags.x": "y"}
    monkeypatch.setattr(tracking, "test_set_opened", lambda experiment, profile="qafi": pd.DataFrame([run]))
    passed, rules, tested, run_id = registry.gate_result("/exp")
    assert passed and run_id == "t1" and tested == {"stability": "models:/m-1"} and rules.loc["r", "passed"]
