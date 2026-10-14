import pandas as pd

from qafi_core import neural, tuning


def test_the_search_starts_from_the_current_setting_and_reports_every_trial(monkeypatch):
    seen = []

    def fake_cross_validate(df, table, index, features, standardise, settings):
        seen.append(settings)
        score = 1 - abs(settings["learning_rate"] - 5e-4) * 1000  # best near 0.0005
        return df.assign(predicted=df["label"] * score, fold=0)

    monkeypatch.setattr(neural, "cross_validate", fake_cross_validate)
    df = pd.DataFrame(
        {"dataset_id": ["d"] * 20, "data_kind": "mavedb", "uniprot_id": "P", "label": range(20)}
    )
    recorded = []
    study = tuning.search(df, None, None, ["rsa"], n_trials=4, on_trial=lambda *a: recorded.append(a))
    assert seen[0] == {key: neural.SETTINGS[key] for key in tuning.SPACE}  # trial 0 is the current setting
    assert len(recorded) == 4 and recorded[0][0] == 0
    assert set(study.best_params) == set(tuning.SPACE)
    assert all(256 <= s["hidden"] <= 1024 and 2 <= s["epochs"] <= 8 for s in seen)
