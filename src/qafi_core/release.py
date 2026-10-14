"""What goes into production (chapter 09): the two released scores, how they are made, and the exam.

Two questions, two scores, both between 0 and 1 and comparable across proteins:
  - stability  how likely the variant leaves the protein stable (08a's network on frozen ESM-2 650M)
  - function   how likely it leaves the protein working: the average of three members' percentiles
               (ESM-2's zero-shot score, 07f's trees and the stability network; 08a's ensemble)

A score is a percentile: 0.2 means "more damaging than 80% of the variants the models were trained on".
Each member is turned into a percentile with its own distribution over the training variants, so the
members can be averaged and a score means the same in every protein. (In cross-validation, ranks were
taken within each assay; with percentiles over all training variants the function score is the same
within an assay, 0.428 against 0.432, and also works across proteins.)

The network reads the residue only: its ESM-2 vector and ten residue-level features. It gives the scores
of all 20 amino acids at once, so the proteome (217 million variants) needs one pass per residue.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from qafi_core import baseline
from qafi_core.features import FEATURES

# Features that describe the residue, not the substitution: the network can read a residue once.
RESIDUE_FEATURES = [
    "esm_wt_logprob", "esm_entropy", "plddt", "rsa", "contacts_8", "contacts_12", "hse_up", "hse_down",
    "neighbour_entropy", "relative_position",
]  # fmt: skip
FUNCTION_MEMBERS = ["esm_delta", "trees", "stability"]  # all oriented: higher = more like the wild type
N_QUANTILES = 1001

# Published predictors in the old proteome table, oriented like our labels (higher = more like the wild
# type). Checked on training proteins: with this sign each correlates positively with the labels.
PUBLISHED = {
    "qafi2": ("QAFI2 (the paper)", 1),
    "am_pathogenicity": ("AlphaMissense", -1),
    "eve": ("EVE", -1),
    "popeve": ("popEVE", 1),
    "cpt1": ("CPT-1", -1),
}


def quantiles(values) -> np.ndarray:
    """``N_QUANTILES`` evenly spaced quantiles of ``values``: a distribution in 1,001 numbers."""
    values = np.asarray(values, dtype=np.float64)
    return np.quantile(values[~np.isnan(values)], np.linspace(0, 1, N_QUANTILES))


def percentile(values, table: np.ndarray) -> np.ndarray:
    """Where each value falls in the distribution ``table`` (from ``quantiles``): 0 to 1."""
    return np.interp(np.asarray(values, dtype=np.float64), table, np.linspace(0, 1, len(table)))


def distributions(df: pd.DataFrame) -> dict[str, list[float]]:
    """The training distribution of every function member and of the stability network's output."""
    return {name: quantiles(df[name]).tolist() for name in FUNCTION_MEMBERS}


def scores(df: pd.DataFrame, tables: dict[str, list[float]]) -> pd.DataFrame:
    """``df`` (columns esm_delta, trees, stability) with the two released scores added, both 0 to 1."""
    members = {name: percentile(df[name], np.asarray(tables[name])) for name in FUNCTION_MEMBERS}
    return df.assign(
        stability_score=members["stability"],
        function_score=sum(members.values()) / len(members),
    )


def test_scores(df: pd.DataFrame, columns: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Each named prediction column scored per dataset (Spearman), as in cross-validation."""
    return {
        name: baseline.dataset_scores(df.assign(predicted=df[column])) for name, column in columns.items()
    }


def against_published(df: pd.DataFrame, ours: dict[str, str], published: pd.DataFrame) -> pd.DataFrame:
    """Our scores next to the published predictors, on exactly the variants every one of them scored.

    ``published`` is the old proteome table (one row per protein variant). Each predictor is compared on
    its own shared variants: a dataset is scored by both, and the table gives the median Spearman of each,
    the share of datasets where ours is higher and the paired Wilcoxon p-value, per kind of data.
    """
    key = ["uniprot_id", "uniprot_position", "ref_aa", "alt_aa"]
    joined = df.merge(
        published.rename(columns={"position": "uniprot_position"}),
        on=key,
        how="inner",
        validate="many_to_one",
    )
    rows = []
    for column, (name, sign) in PUBLISHED.items():
        have = joined[joined[column].notna()].assign(
            published=lambda d, c=column, s=sign: s * d[c].astype(float)
        )
        if have.empty:
            continue
        theirs = baseline.dataset_scores(have.assign(predicted=have["published"]))
        for our_name, our_column in ours.items():
            mine = baseline.dataset_scores(have.assign(predicted=have[our_column]))
            both = mine.merge(theirs, on=["dataset_id", "data_kind"], suffixes=("", "_published"))
            for kind, g in both.groupby("data_kind"):
                gain = g["spearman"] - g["spearman_published"]
                rows.append(
                    {
                        "published": name,
                        "ours": our_name,
                        "data_kind": kind,
                        "datasets": len(g),
                        "variants": int(g["variants"].sum()),
                        "ours_median": g["spearman"].median(),
                        "published_median": g["spearman_published"].median(),
                        "ours_better": (gain > 0).mean(),
                        "p_value": stats.wilcoxon(gain).pvalue
                        if len(g) > 1 and (gain != 0).any()
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows).round(3)


GATE = {
    # rule: (score, kind of data, must beat, reason)
    "stability beats zero-shot on stability assays": ("stability_score", "domainome", "esm_delta"),
    "stability beats the trees on stability assays": ("stability_score", "domainome", "trees"),
    "function beats zero-shot on functional assays": ("function_score", "mavedb", "esm_delta"),
}


def gate(per_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The release rules, checked on the test proteins: each released score must beat the simpler one.

    ``per_model`` maps a prediction column to its per-dataset test scores. A rule passes when the released
    score is higher on more than half of the datasets of that kind and the paired Wilcoxon test gives
    p below 0.05. Returns one row per rule with the evidence.
    """
    rows = []
    for rule, (released, kind, simpler) in GATE.items():
        comparison = baseline.compare(per_model[released], per_model[simpler])
        better, p = comparison[f"better_{kind}"], comparison[f"p_value_{kind}"]
        rows.append(
            {
                "rule": rule,
                "datasets better": better,
                "mean gain": comparison[f"difference_{kind}"],
                "p_value": p,
                "passed": bool(better > 0.5 and p < 0.05),
            }
        )
    return pd.DataFrame(rows).set_index("rule")


def features_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The 16 feature columns the trees read, as floats, in their training order."""
    return df[FEATURES].astype(np.float32)
