import sys
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from qafi_platform import tracking


@pytest.fixture
def scores():
    return pd.DataFrame(
        {
            "dataset_id": [f"d{i}" for i in range(12)],
            "data_kind": ["domainome"] * 6 + ["mavedb"] * 6,
            "uniprot_id": "P1",
            "variants": 100,
            "spearman": np.linspace(0.3, 0.6, 12),
        }
    )


@pytest.fixture
def statements(monkeypatch):
    """Every SQL statement the code sends, instead of a warehouse."""
    sent = []

    def fake_sql(w, statement):
        sent.append(" ".join(statement.split()))
        return pd.DataFrame({"n": ["3"]})

    monkeypatch.setattr(tracking, "sql", fake_sql)
    return sent


def test_scores_replace_the_rows_of_their_run(scores, statements):
    tracking.save_scores(None, scores.head(2), "mlp", "run1")
    assert statements[0] == "DELETE FROM qafi_dev.ml.cv_scores WHERE run_name = 'mlp'"
    assert statements[1].startswith(
        "INSERT INTO qafi_dev.ml.cv_scores VALUES ('mlp', 'run1', 'd0', 'domainome'"
    )
    assert statements[1].count("('mlp'") == 2


def test_predictions_travel_as_a_parquet_file_through_the_volume(statements):
    uploaded = {}
    w = NS(files=NS(upload=lambda path, data, overwrite: uploaded.update({path: pd.read_parquet(data)})))
    predictions = pd.DataFrame({"uniprot_position": [1, 2, 3], "predicted": [0.1, 0.2, 0.3]}).assign(
        dataset_id="d0", uniprot_id="P1", ref_aa="A", alt_aa="C", fold=0, label=9.9
    )
    assert tracking.save_predictions(w, predictions, "mlp") == 3
    path = "/Volumes/qafi_dev/ml/files/cv_predictions/mlp.parquet"
    assert list(uploaded[path].columns) == tracking.PREDICTION_COLUMNS  # the label does not travel
    assert statements[0] == "DELETE FROM qafi_dev.ml.cv_predictions WHERE run_name = 'mlp'"
    assert f"FROM read_files('{path}', format => 'parquet')" in statements[1]


def test_a_run_is_recorded_with_its_comparisons_and_code_version(scores, statements, monkeypatch):
    logged = {"params": {}, "metrics": {}, "tags": {}, "tables": [], "models": []}

    class Run:
        info = NS(run_id="run1")

        def __enter__(self):
            return self

        def __exit__(self, *error):
            return False

    fake = NS(
        set_tracking_uri=lambda uri: logged.update(uri=uri),
        set_experiment=lambda name: logged.update(experiment=name),
        start_run=lambda run_name: logged.update(run_name=run_name) or Run(),
        set_tags=logged["tags"].update,
        log_params=logged["params"].update,
        log_metrics=logged["metrics"].update,
        log_table=lambda table, name: logged["tables"].append(name),
        pytorch=NS(log_model=lambda model, name, signature: logged["models"].append((model, signature))),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setattr(tracking, "code_version", lambda: "abc123-dirty")
    worse = scores.assign(spearman=scores["spearman"] - 0.05)
    extras = {"references": {"baseline": worse}, "tables": {"extra": scores}}
    run_id = tracking.cv_run(
        None, "/Users/me/exp", "mlp", {"epochs": 4}, scores, **extras, model="m", signature="s", chapter="08a"
    )
    assert (
        run_id == "run1" and logged["uri"] == "databricks://qafi" and logged["experiment"] == "/Users/me/exp"
    )
    assert logged["tags"] == {"chapter": "08a", "code_version": "abc123-dirty", "trained_on": "laptop"}
    assert logged["params"] == {"epochs": 4}
    assert logged["metrics"]["spearman_median_mavedb"] == pytest.approx(0.5318, abs=1e-4)
    assert logged["metrics"]["vs_baseline_better_domainome"] == 1.0
    assert logged["metrics"]["vs_baseline_difference_mavedb"] == pytest.approx(0.05)
    assert logged["tables"] == ["dataset_scores.json", "extra.json"]
    assert logged["models"] == [("m", "s")]
    assert "INSERT INTO qafi_dev.ml.cv_scores VALUES ('mlp', 'run1'" in statements[1]


def test_the_signature_names_each_input_with_its_width():
    signature = tracking.tensor_signature({"embedding": 1280, "features": 16}, 20)
    inputs = {spec.name: spec.shape for spec in signature.inputs}
    assert inputs == {"embedding": (-1, 1280), "features": (-1, 16)}
    assert tuple(signature.outputs.to_dict()[0]["tensor-spec"]["shape"]) == (-1, 20)


def test_the_code_version_says_when_files_differ_from_the_commit(monkeypatch):
    answers = {"rev-parse": "abc123\n", "status": " M file.py\n"}
    monkeypatch.setattr(tracking.subprocess, "run", lambda cmd, **kw: NS(stdout=answers[cmd[1]]))
    assert tracking.code_version() == "abc123-dirty"
    answers["status"] = ""
    assert tracking.code_version() == "abc123"


def test_scores_come_back_as_numbers(monkeypatch):
    text = pd.DataFrame(
        {
            "dataset_id": ["d0"],
            "data_kind": ["mavedb"],
            "uniprot_id": ["P1"],
            "variants": ["120"],
            "spearman": ["0.41"],
        }
    )
    monkeypatch.setattr(tracking, "sql", lambda w, statement: text)
    scores = tracking.read_scores(None, "baseline")
    assert scores["variants"].tolist() == [120] and scores["spearman"].tolist() == [0.41]


def test_the_newest_run_of_each_name_is_read_with_its_comparisons(monkeypatch):
    found = pd.DataFrame(
        {
            "run_id": ["new", "old", "zero"],
            "tags.mlflow.runName": ["mlp", "mlp", "esm2_zero_shot"],  # newest first, as asked for
            "metrics.spearman_median": [0.63, 0.10, 0.45],
            "metrics.spearman_median_domainome": [0.65, 0.1, 0.46],
            "metrics.spearman_median_proteingym": [0.63, 0.1, 0.47],
            "metrics.spearman_median_mavedb": [0.38, 0.1, 0.40],
            "metrics.vs_baseline_better_mavedb": [0.33, None, None],
            "metrics.vs_baseline_difference_mavedb": [-0.017, None, None],
            "metrics.vs_baseline_p_value_mavedb": [0.005, None, None],
            "metrics.vs_baseline_better_domainome": [0.9, None, None],
            "metrics.vs_baseline_difference_domainome": [0.079, None, None],
            "metrics.vs_baseline_p_value_domainome": [0.0, None, None],
        }
    )
    asked = {}
    fake = NS(
        set_tracking_uri=lambda uri: None,
        set_experiment=lambda name: asked.update(experiment=name),
        search_runs=lambda order_by: asked.update(order_by=order_by) or found,
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    runs = tracking.latest_runs("/Users/me/exp")
    assert asked == {"experiment": "/Users/me/exp", "order_by": ["start_time DESC"]}
    assert runs.index.tolist() == ["mlp", "esm2_zero_shot"] and runs.loc["mlp", "run_id"] == "new"
    assert runs.loc["mlp", ["all", "domainome", "proteingym", "mavedb"]].tolist() == [0.63, 0.65, 0.63, 0.38]
    table = tracking.comparison(runs, "mlp", "baseline")
    assert table.loc["mavedb"].tolist() == [0.33, -0.017, 0.005]
    assert table.loc["domainome"].tolist() == [0.9, 0.079, 0.0]
