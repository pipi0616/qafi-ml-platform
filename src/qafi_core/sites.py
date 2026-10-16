"""A map of what each position of a protein is for (notebook 14e).

14c gives every substitution two energies. Averaged over the twenty amino acids, they say something about
the **position** rather than the substitution:

    structural    changing it costs folding energy: the residue holds the protein together
    functional    changing it costs little folding energy but a lot of functional energy: the protein
                  still folds, it just stops working -- an active site, a ligand pocket, an interface
    tolerant      neither: the position can be changed

The functional class is the interesting one. It is where a drug that stabilises the protein will not help,
and it is where a binding pocket tends to be. ``classify`` builds the map, ``enrichment`` checks it against
positions already known to be functional, and ``pocket_rank`` asks where a known drug pocket sits in the
ranking of a protein's own positions.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

STRUCTURAL, FUNCTIONAL, TOLERANT, MIXED = "structural", "functional", "tolerant", "mixed"

# UniProt writes its feature columns as "ACT_SITE 123; /note=..."; the number is the position.
FEATURE = re.compile(r"(?:ACT_SITE|BINDING|SITE)\s+(\d+)")


def known_sites(table: pd.DataFrame) -> pd.DataFrame:
    """Positions UniProt annotates as active site, binding site or site, one row each.

    ``table`` is UniProt's TSV with the columns Entry, "Active site", "Binding site", "Site". Ranges
    (``BINDING 10..14``) are expanded; a position annotated twice appears once.
    """
    rows = []
    for entry, *columns in table[["Entry", "Active site", "Binding site", "Site"]].itertuples(index=False):
        for kind, text in zip(("active", "binding", "site"), columns, strict=True):
            if not isinstance(text, str):
                continue
            for part in FEATURE.finditer(text):
                rows.append({"uniprot_id": entry, "position": int(part.group(1)), "annotation": kind})
    return pd.DataFrame(rows).drop_duplicates(["uniprot_id", "position"])


def classify(
    variants: pd.DataFrame,
    fold: str = "ddg_fold",
    func: str = "ddg_func",
    quantile: float = 0.25,
) -> pd.DataFrame:
    """One row per position: its mean two energies and what the position is for.

    A position counts as sensitive in one energy when its mean is in the worst ``quantile`` of that
    protein's positions. Thresholds are per protein, because proteins differ in how tolerant they are
    overall and the question here is which positions stand out *within* a protein.
    """
    by_position = (
        variants.groupby(["uniprot_id", "position"])[[fold, func]]
        .mean()
        .reset_index()
        .rename(columns={fold: "fold_cost", func: "func_cost"})
    )
    worst = by_position.groupby("uniprot_id")[["fold_cost", "func_cost"]].transform(
        lambda column: column.quantile(quantile)
    )
    structural = by_position.fold_cost <= worst.fold_cost
    functional = by_position.func_cost <= worst.func_cost
    by_position["position_is"] = np.select(
        [structural & ~functional, functional & ~structural, structural & functional],
        [STRUCTURAL, FUNCTIONAL, MIXED],
        TOLERANT,
    )
    return by_position


def enrichment(positions: pd.DataFrame, annotated: pd.DataFrame) -> pd.DataFrame:
    """How much more often each class of position is an annotated functional site than chance.

    Returns one row per class with the share of its positions that are annotated, the odds ratio against
    every other position, and Fisher's exact p-value.
    """
    from scipy import stats

    known = set(zip(annotated.uniprot_id, annotated.position, strict=True))
    marked = positions.assign(
        annotated=[(u, p) in known for u, p in zip(positions.uniprot_id, positions.position, strict=True)]
    )
    rows = []
    for name, group in marked.groupby("position_is"):
        other = marked[marked.position_is != name]
        table = [
            [group.annotated.sum(), (~group.annotated).sum()],
            [other.annotated.sum(), (~other.annotated).sum()],
        ]
        odds, p = stats.fisher_exact(table, alternative="greater")
        rows.append(
            {
                "position_is": name,
                "positions": len(group),
                "annotated": int(group.annotated.sum()),
                "share annotated": group.annotated.mean(),
                "odds ratio": odds,
                "p": p,
            }
        )
    return pd.DataFrame(rows).sort_values("share annotated", ascending=False, ignore_index=True)


def pocket_rank(
    positions: pd.DataFrame, uniprot_id: str, pocket: list[int], column: str = "func_cost"
) -> dict:
    """Where a known pocket's residues sit in the ranking of one protein's positions by ``column``.

    A pocket the model knew nothing about should still come out near the top if the map is useful. The
    percentile is 0 for the most sensitive position of the protein and 1 for the least.
    """
    from scipy import stats

    protein = positions[positions.uniprot_id == uniprot_id]
    if protein.empty:
        raise ValueError(f"no positions for {uniprot_id}")
    rank = protein[column].rank(pct=True)
    inside = protein.position.isin(pocket)
    if not inside.any():
        raise ValueError(f"none of the pocket positions are in the map for {uniprot_id}")
    return {
        "uniprot_id": uniprot_id,
        "pocket residues": int(inside.sum()),
        "protein positions": len(protein),
        "median percentile": float(rank[inside].median()),
        "median percentile elsewhere": float(rank[~inside].median()),
        "p": float(
            stats.mannwhitneyu(protein[column][inside], protein[column][~inside], alternative="less").pvalue
        ),
    }
