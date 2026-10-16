"""Deep mutational scanning from reads (notebook 14b): read the pipeline's fitness and check it against
the published one.

The Nextflow pipeline in ``pipelines/dms_fitness`` turns sequencing reads into a fitness value for every
variant (DiMSum's ``fitness_*.txt`` tables). This module reads those tables into one frame and lines them
up, variant by variant, with the fitness the authors deposited in GEO: the same amino-acid sequence on both
sides, so a reproduced number can be compared with the published one directly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# DiMSum writes one space-separated table per kind of variant (the wild type, synonymous changes,
# single and double amino-acid substitutions); all share aa_seq, Nham_aa, fitness and sigma.
TABLES = ("fitness_wildtype.txt", "fitness_silent.txt", "fitness_singles.txt", "fitness_doubles.txt")


def read_fitness(folder: str | Path) -> pd.DataFrame:
    """Every variant DiMSum scored, from the wild type, singles and doubles tables in ``folder``.

    Columns: aa_seq, Nham_aa (number of amino-acid changes), WT (True for the wild type), fitness, sigma
    (its standard error). Doubles have both a plain and a Bayesian-corrected fitness in DiMSum's output;
    the plain one is used, as in the deposited table, unless only the corrected one is there. The wild type
    itself has no fitness column (it is 0 by definition), so it is filled in.
    """
    folder = Path(folder)
    frames = []
    for name in TABLES:
        path = folder / name
        if not path.exists():
            continue
        t = pd.read_csv(path, sep=r"\s+")  # space separated, NA for missing
        fitness = "fitness" if "fitness" in t else "fitness_uncorr"
        sigma = "sigma" if "sigma" in t else "sigma_uncorr"
        if fitness not in t:  # the wild-type table: fitness is 0 by definition
            t[fitness], t[sigma] = 0.0, 0.0
        frames.append(
            pd.DataFrame(
                {
                    "aa_seq": t["aa_seq"],
                    "Nham_aa": t["Nham_aa"] if "Nham_aa" in t else 0,
                    "WT": t["WT"].fillna(False).astype(bool) if "WT" in t else False,
                    "fitness": t[fitness],
                    "sigma": t[sigma],
                }
            )
        )
    if not frames:
        raise FileNotFoundError(f"no DiMSum fitness tables in {folder}")
    return pd.concat(frames, ignore_index=True).drop_duplicates("aa_seq")


def read_published(path: str | Path) -> pd.DataFrame:
    """The authors' deposited fitness table (GEO supplementary file): aa_seq, Nham_aa, WT, fitness, sigma."""
    t = pd.read_csv(path, sep="\t")
    t["WT"] = t["WT"].fillna(False).astype(bool)
    return t


def compare(ours: pd.DataFrame, published: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Line up our fitness with the published one by amino-acid sequence.

    Returns the joined table and a summary: how many variants each side has, how many both have, and the
    Pearson correlation of fitness overall and per number of mutations.
    """
    joined = ours.merge(published, on="aa_seq", suffixes=("_ours", "_published"))
    summary = {
        "ours": len(ours),
        "published": len(published),
        "shared": len(joined),
        "only_ours": len(ours) - len(joined),
        "only_published": len(published) - len(joined),
        "pearson": joined["fitness_ours"].corr(joined["fitness_published"]),
        "max_abs_difference": (joined["fitness_ours"] - joined["fitness_published"]).abs().max(),
    }
    for n, group in joined.groupby("Nham_aa_published"):
        if len(group) > 1:
            summary[f"pearson_{int(n)}_mutations"] = group["fitness_ours"].corr(group["fitness_published"])
            summary[f"shared_{int(n)}_mutations"] = len(group)
    return joined, summary
