"""Working from a handful of measurements (notebook 14x): few-shot fitting and active learning.

Chapters 07-09 train on hundreds of proteins and predict a new one with no measurements at all. A protein
engineering project is the other way round: one protein, a budget of a few dozen or a few hundred variants
to measure, and the question of what to make next.

``curve`` measures how well a model does after seeing n measured variants of a protein it was never
trained on, for a few budgets and several draws each. ``campaign`` simulates the loop a lab actually runs:
measure a batch, refit, choose the next batch, repeat; what matters there is not correlation but whether
the best variants have been found.

The models compared are the standard few-shot baselines (Hsu et al. 2022): a ridge regression on the
one-hot encoding of the substitution, the same on the ESM-2 residue vector, and the "augmented" version
that adds a prediction made without any measurement as one more feature.

That prediction -- the **prior** -- is what this project's own released model competes for. Both functions
take a mapping of named priors to columns, so one run compares starting from the free ESM-2 zero-shot
score against starting from the score chapter 09 released: the prior is the line a fitted model has to
beat, and it is also a feature the fitted model gets to use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_PRIORS = {"zero-shot": "predicted"}


def one_hot(df: pd.DataFrame, positions: list[int]) -> np.ndarray:
    """Position and mutant amino acid as indicator columns: the encoding with no biology in it."""
    index = {p: i for i, p in enumerate(positions)}
    out = np.zeros((len(df), len(positions) + len(AMINO_ACIDS)), dtype=np.float32)
    for row, (position, alt) in enumerate(zip(df.position, df.mut_aa, strict=True)):
        if position in index:
            out[row, index[position]] = 1
        out[row, len(positions) + AMINO_ACIDS.index(alt)] = 1
    return out


def features(df: pd.DataFrame, embeddings, positions: list[int], kind: str, prior: str) -> np.ndarray:
    """The inputs of one of the three baselines: one-hot, the ESM-2 vector, or the vector plus a prior."""
    if kind == "one_hot":
        return one_hot(df, positions)
    vectors = embeddings[df.row.values].astype(np.float32)
    if kind == "embedding":
        return vectors
    return np.hstack([vectors, df[[prior]].to_numpy(dtype=np.float32)])


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    """Ridge regression, solved directly; the inputs are standardised first."""
    mean, std = x.mean(0), x.std(0) + 1e-6
    z = (x - mean) / std
    weights = np.linalg.solve(z.T @ z + alpha * len(x) * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    return lambda new: ((new - mean) / std) @ weights + y.mean()


def curve(
    df: pd.DataFrame,
    embeddings,
    budgets=(24, 48, 96, 384),
    draws: int = 10,
    priors: dict[str, str] | None = None,
    label: str = "label",
    seed: int = 0,
) -> pd.DataFrame:
    """How well each model ranks the variants it has not seen, after n of them were measured.

    One row per (budget, draw, model). ``df`` is one assay of one protein, with a ``row`` column pointing
    into ``embeddings`` and every prior's column already in it. Each prior contributes two models: itself,
    unfitted (a flat line -- it never sees the measurements), and the augmented ridge that reads it as an
    extra feature. The one-hot and plain-embedding ridges do not depend on a prior and are fitted once.
    """
    priors = priors or DEFAULT_PRIORS
    rng = np.random.default_rng(seed)
    positions = sorted(df.position.unique())
    rows = []
    for budget in budgets:
        if len(df) < budget + 100:  # keep at least 100 variants to be judged on
            continue
        for draw in range(draws):
            measured = df.iloc[rng.choice(len(df), budget, replace=False)]
            rest = df.drop(measured.index)
            y = measured[label].to_numpy(dtype=np.float32)
            scores = {
                name: stats.spearmanr(rest[label], rest[column]).statistic for name, column in priors.items()
            }
            fits = {"one-hot + ridge": "one_hot", "ESM-2 + ridge": "embedding"}
            fits |= {f"ESM-2 + {name} + ridge": name for name in priors}
            for model, kind in fits.items():
                column = priors.get(kind, "")  # the prior-free encodings ignore it
                key = kind if kind in ("one_hot", "embedding") else "augmented"
                predict = fit_ridge(features(measured, embeddings, positions, key, column), y)
                guess = predict(features(rest, embeddings, positions, key, column))
                scores[model] = stats.spearmanr(rest[label], guess).statistic
            rows.extend(
                {"budget": budget, "draw": draw, "model": k, "spearman": v} for k, v in scores.items()
            )
    return pd.DataFrame(rows)


def campaign(
    df: pd.DataFrame,
    embeddings,
    batch: int = 96,
    rounds: int = 4,
    strategy: str = "greedy",
    prior: str = "predicted",
    label: str = "label",
    seed: int = 0,
) -> pd.DataFrame:
    """One design loop: measure a batch, refit, choose the next batch, repeat.

    ``strategy`` picks the next batch: ``random`` ignores the model (and the prior), ``greedy`` takes the
    best predicted, ``ucb`` adds the spread of an ensemble of ridge fits to favour variants the model is
    unsure about. The first batch is always random, because nothing is fitted yet.

    Returns one row per round with the best label measured so far, as a share of the best in the whole
    landscape: the number a project actually cares about.
    """
    rng = np.random.default_rng(seed)
    positions = sorted(df.position.unique())
    best_possible = df[label].max()
    taken = list(rng.choice(len(df), batch, replace=False))
    rows = []
    for round_number in range(1, rounds + 1):
        measured = df.iloc[taken]
        rows.append(
            {
                "round": round_number,
                "measured": len(taken),
                "strategy": strategy,
                "best found": measured[label].max() / best_possible,
                "top 1% found": float((measured[label] >= df[label].quantile(0.99)).sum()),
            }
        )
        if round_number == rounds:
            break
        rest = df.drop(df.index[taken])
        if strategy == "random":
            chosen = rng.choice(len(rest), batch, replace=False)
        else:
            x = features(measured, embeddings, positions, "augmented", prior)
            y = measured[label].to_numpy(dtype=np.float32)
            new = features(rest, embeddings, positions, "augmented", prior)
            guesses = np.array([fit_ridge(x, y, alpha)(new) for alpha in (0.3, 1.0, 3.0)])
            score = guesses.mean(0) + (guesses.std(0) if strategy == "ucb" else 0)
            chosen = np.argsort(-score)[:batch]
        taken += [df.index.get_loc(i) for i in rest.index[chosen]]
    return pd.DataFrame(rows)
