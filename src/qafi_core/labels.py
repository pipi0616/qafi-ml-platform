"""Training labels (notebook 07c): checks of the new labels against the old QAFI paper's labels.

The labels themselves are built in dbt (gold.dim_label_dataset, gold.fct_variant_label); this module only
compares them with the paper's training set, variant by variant.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats


def compare_with_paper(new: pd.DataFrame, paper: pd.DataFrame, min_variants: int = 50) -> pd.DataFrame:
    """For each new dataset sharing variants with the paper: Pearson correlation and the two median labels.

    ``new`` has dataset_id, uniprot_id, uniprot_position, ref_aa, alt_aa, label; ``paper`` is the paper's
    training set (uniprot, protein, pos, first, second, score_log_normalized). High correlation with a shifted
    median means the same ranking on a different calibration; low correlation usually means the two came from
    different assays of the same protein.
    """
    both = new.merge(
        paper,
        left_on=["uniprot_id", "uniprot_position", "ref_aa", "alt_aa"],
        right_on=["uniprot", "pos", "first", "second"],
    )
    rows = [
        {
            "protein": protein,
            "dataset_id": dataset,
            "variants": len(g),
            "pearson": stats.pearsonr(g["label"], g["score_log_normalized"])[0],
            "median_new": g["label"].median(),
            "median_paper": g["score_log_normalized"].median(),
        }
        for (dataset, protein), g in both.groupby(["dataset_id", "protein"])
        if len(g) >= min_variants
    ]
    return pd.DataFrame(rows).sort_values("pearson", ascending=False, ignore_index=True)
