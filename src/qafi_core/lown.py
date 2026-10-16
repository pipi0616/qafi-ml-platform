"""Working from a handful of measurements (notebook 14d): few-shot fitting and active learning.

Chapters 07-09 train on hundreds of proteins and predict a new one with no measurements at all. A protein
engineering project is the other way round: one protein, a budget of a few dozen or a few hundred variants
to measure, and the question of what to make next.

``curve`` measures how well a model does after seeing n measured variants of a protein it was never
trained on, for a few budgets and several draws each. ``campaign`` simulates the loop a lab actually runs:
measure a batch, refit, choose the next batch, repeat; what matters there is not correlation but whether
the best variants have been found.

The models compared are the standard few-shot baselines (Hsu et al. 2022): a ridge regression on the
one-hot encoding of the substitution, the same on the ESM-2 residue vector, and the "augmented" version
that adds the zero-shot prediction as one more feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def one_hot(df: pd.DataFrame, positions: list[int]) -> np.ndarray:
    """Position and mutant amino acid as indicator columns: the encoding with no biology in it."""
    index = {p: i for i, p in enumerate(positions)}
    out = np.zeros((len(df), len(positions) + len(AMINO_ACIDS)), dtype=np.float32)
    for row, (position, alt) in enumerate(zip(df.position, df.mut_aa, strict=True)):
        if position in index:
            out[row, index[position]] = 1
        out[row, len(positions) + AMINO_ACIDS.index(alt)] = 1
    return out


def features(df: pd.DataFrame, embeddings, positions: list[int], kind: str, zero_shot: str) -> np.ndarray:
    """The inputs of one of the three baselines: one-hot, the ESM-2 vector, or the vector plus the score."""
    if kind == "one_hot":
        return one_hot(df, positions)
    vectors = embeddings[df.row.values].astype(np.float32)
    if kind == "embedding":
        return vectors
    return np.hstack([vectors, df[[zero_shot]].to_numpy(dtype=np.float32)])


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
    zero_shot: str = "predicted",
    label: str = "label",
    seed: int = 0,
) -> pd.DataFrame:
    """How well each model ranks the variants it has not seen, after n of them were measured.

    One row per (budget, draw, model). ``df`` is one assay of one protein, with a ``row`` column pointing
    into ``embeddings`` and a zero-shot prediction already in it.
    """
    rng = np.random.default_rng(seed)
    positions = sorted(df.position.unique())
    rows = []
    for budget in budgets:
        if len(df) < budget + 100:  # keep at least 100 variants to be judged on
            continue
        for draw in range(draws):
            measured = df.iloc[rng.choice(len(df), budget, replace=False)]
            rest = df.drop(measured.index)
            scores = {"zero-shot": stats.spearmanr(rest[label], rest[zero_shot]).statistic}
            for kind in ("one_hot", "embedding", "embedding + zero-shot"):
                name = {
                    "one_hot": "one-hot + ridge",
                    "embedding": "ESM-2 + ridge",
                    "embedding + zero-shot": "ESM-2 + zero-shot + ridge",
                }[kind]
                key = "augmented" if kind == "embedding + zero-shot" else kind
                x = features(measured, embeddings, positions, key, zero_shot)
                predict = fit_ridge(x, measured[label].to_numpy(dtype=np.float32))
                guess = predict(features(rest, embeddings, positions, key, zero_shot))
                scores[name] = stats.spearmanr(rest[label], guess).statistic
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
    zero_shot: str = "predicted",
    label: str = "label",
    seed: int = 0,
) -> pd.DataFrame:
    """One design loop: measure a batch, refit, choose the next batch, repeat.

    ``strategy`` picks the next batch: ``random`` ignores the model, ``greedy`` takes the best predicted,
    ``ucb`` adds the spread of an ensemble of ridge fits to favour variants the model is unsure about.
    The first batch is always random, because nothing is fitted yet.

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
            x = features(measured, embeddings, positions, "augmented", zero_shot)
            y = measured[label].to_numpy(dtype=np.float32)
            new = features(rest, embeddings, positions, "augmented", zero_shot)
            guesses = np.array([fit_ridge(x, y, alpha)(new) for alpha in (0.3, 1.0, 3.0)])
            score = guesses.mean(0) + (guesses.std(0) if strategy == "ucb" else 0)
            chosen = np.argsort(-score)[:batch]
        taken += [df.index.get_loc(i) for i in rest.index[chosen]]
    return pd.DataFrame(rows)
