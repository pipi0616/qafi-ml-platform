"""Searching the network's settings with Optuna (chapter 09a), on the training proteins only.

Each trial is one setting of the network: Optuna proposes it, the network is cross-validated with the
folds of chapter 07a, and the median Spearman over all training datasets is the trial's score. The
proposals are not random: after a few trials Optuna models which settings scored well (a Tree-structured
Parzen Estimator) and proposes more like them. The test proteins are never involved.
"""

from __future__ import annotations

from qafi_core import baseline, neural

SPACE = {  # name: (low, high, kind)
    "hidden": (256, 1024, "choice"),
    "dropout": (0.1, 0.4, "float"),
    "learning_rate": (1e-4, 1e-3, "log"),
    "epochs": (2, 8, "int"),
    "weight_decay": (1e-4, 1e-1, "log"),
}


def suggest(trial) -> dict:
    """One setting of the network, proposed by Optuna within ``SPACE``."""
    return {
        "hidden": trial.suggest_categorical("hidden", [256, 512, 1024]),
        "dropout": trial.suggest_float("dropout", *SPACE["dropout"][:2]),
        "learning_rate": trial.suggest_float("learning_rate", *SPACE["learning_rate"][:2], log=True),
        "epochs": trial.suggest_int("epochs", *SPACE["epochs"][:2]),
        "weight_decay": trial.suggest_float("weight_decay", *SPACE["weight_decay"][:2], log=True),
    }


def search(df, table, index, features: list[str], n_trials: int, seed: int = 0, on_trial=None):
    """Run ``n_trials`` trials and return the Optuna study (best_params, trials_dataframe()).

    The first trial is the setting the network has used so far, so the search can only find something
    better. ``on_trial(number, settings, summary)`` is called after each trial, to record it.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial) -> float:
        settings = suggest(trial)
        cv = neural.cross_validate(df, table, index, features=features, standardise=True, settings=settings)
        summary = baseline.summarise(baseline.dataset_scores(cv))
        for name, value in summary.items():
            trial.set_user_attr(name, value)
        if on_trial:
            on_trial(trial.number, settings, summary)
        return summary["spearman_median"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    current = {key: neural.SETTINGS[key] for key in SPACE}
    study.enqueue_trial(current)
    study.optimize(objective, n_trials=n_trials)
    return study
