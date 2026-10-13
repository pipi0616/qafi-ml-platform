"""How models are tested (chapters 07-09): proteins are held out whole, and related proteins stay together.

The model is meant for proteins without any DMS data, so it is always tested on proteins it has not seen.
Three rules follow:
  - all variants of a protein are on one side, training or testing; variants are never split at random
  - related proteins (similar sequences) are on the same side as well, or testing becomes partly memory
  - the final test proteins are chosen once and locked; nothing is tuned on them

``similar_groups`` puts related proteins into one group; ``split_proteins`` sends a fixed share of the
groups to the test set and ``lock_split`` writes that split once; ``group_folds`` makes cross-validation
folds by group; ``per_protein`` scores each held-out protein. The summary is the median over proteins, as
in ProteinGym and the QAFI paper.
"""

from __future__ import annotations

import hashlib
import math
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LAMBDA, K = 0.267, 0.041  # BLAST's alignment statistics for BLOSUM62 with gap costs 11/1
# Two proteins are relatives when their best local alignment passes all three, as in sequence clustering
# tools (MMseqs2, CD-HIT): unlikely by chance, at least 30% identical, covering half of the shorter protein.
MAX_EVALUE = 1e-5
MIN_IDENTITY = 0.3
MIN_COVERAGE = 0.5


@cache
def _aligner():
    from Bio.Align import PairwiseAligner, substitution_matrices

    return PairwiseAligner(
        mode="local",
        substitution_matrix=substitution_matrices.load("BLOSUM62"),
        open_gap_score=-11,
        extend_gap_score=-1,
    )


def evalue(a: str, b: str) -> float:
    """How often an alignment as good as the best one of ``a`` and ``b`` arises by chance (BLAST statistics).

    A short protein matched against a very long one finds some similar stretch by chance alone; the E-value
    accounts for the lengths, so only real relatives (shared domains included) come out small.
    """
    bits = (LAMBDA * _aligner().score(a, b) - math.log(K)) / math.log(2)
    return len(a) * len(b) * 2.0**-bits


def related(a: str, b: str) -> bool:
    """Whether two proteins are relatives (see the thresholds above).

    The E-value alone is not enough: without the low-complexity masking BLAST applies, repetitive stretches
    of long proteins look significant, and chains of such matches merge unrelated proteins.
    """
    if evalue(a, b) >= MAX_EVALUE:
        return False
    counts = _aligner().align(a, b)[0].counts()
    aligned = counts.identities + counts.mismatches
    return counts.identities >= MIN_IDENTITY * aligned and aligned >= MIN_COVERAGE * min(len(a), len(b))


def similar_groups(sequences: dict[str, str]) -> dict[str, str]:
    """For each protein, the group of related proteins it belongs to, named after its first member.

    Groups follow chains of relations: A related to B and B to C puts A, B and C together.
    """
    names = sorted(sequences)
    group = {name: name for name in names}

    def root(name: str) -> str:
        while group[name] != name:
            name = group[name]
        return name

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if related(sequences[a], sequences[b]):
                group[max(root(a), root(b))] = min(root(a), root(b))
    return {name: root(name) for name in names}


def split_proteins(groups: dict[str, str], test_share: float = 0.2, salt: str = "qafi-test-v1") -> dict:
    """'train' or 'test' per protein: whole groups go to the test set, picked by a fixed hash of their name.

    A hash instead of a random draw: a group always lands on the same side, also when proteins are added
    later, so the test set never changes without anyone noticing.
    """

    def bucket(name: str) -> int:
        return int(hashlib.sha256(f"{salt}:{name}".encode()).hexdigest(), 16) % 1000

    return {protein: "test" if bucket(g) < test_share * 1000 else "train" for protein, g in groups.items()}


def lock_split(path: Path, sequences: dict[str, str]) -> pd.DataFrame:
    """The train/test split of the proteins: computed and written to ``path`` once, only read afterwards.

    The file is committed to git. Changing the test set after seeing results would let those results leak
    into it, so an existing file is never recomputed.
    """
    if path.exists():
        return pd.read_csv(path)
    groups = similar_groups(sequences)
    split = split_proteins(groups)
    table = pd.DataFrame(
        {
            "uniprot_id": list(groups),
            "similarity_group": list(groups.values()),
            "split": [split[protein] for protein in groups],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def representative_groups(sequences: dict[str, str]) -> dict[str, str]:
    """For each protein, the group it belongs to, named after the group's representative.

    Unlike :func:`similar_groups`, relations are not followed in chains: the longest protein left becomes a
    representative, every protein related to that representative itself joins its group, and the group is
    then closed. Chains merge unrelated families, because multi-domain proteins share single domains: one
    zinc finger protein reaches a kinase through a third protein that has both. This is the clustering that
    CD-HIT and MMseqs2 perform (a greedy set cover), and the one sequence splits in the field are built on.
    """
    left = sorted(sequences, key=lambda name: (-len(sequences[name]), name))
    group = {}
    while left:
        representative, *rest = left
        group[representative] = representative
        left = []
        for name in rest:
            if related(sequences[representative], sequences[name]):
                group[name] = representative
            else:
                left.append(name)
    return group


def extend_split(path: Path, sequences: dict[str, str]) -> pd.DataFrame:
    """Add proteins to the locked split in ``path``; rows already there are never touched.

    A new protein related to a protein already in the split joins its group and its side, so relatives are
    never split apart. The remaining new proteins are grouped among themselves by
    :func:`representative_groups`, and each group draws its side from the same hash as :func:`split_proteins`.
    """
    locked = pd.read_csv(path)
    known = dict(zip(locked.uniprot_id, locked.similarity_group, strict=True))
    side = dict(zip(locked.uniprot_id, locked.split, strict=True))
    new = [name for name in sorted(sequences) if name not in known]
    rows = []
    for protein in new:
        relatives = [q for q in sorted(known) if q in sequences and related(sequences[q], sequences[protein])]
        if relatives:
            relative = relatives[0]
            rows.append({"uniprot_id": protein, "similarity_group": known[relative], "split": side[relative]})
    joined = {row["uniprot_id"] for row in rows}
    rest = {name: sequences[name] for name in new if name not in joined}
    groups = representative_groups(rest)
    split = split_proteins(groups)
    rows += [
        {"uniprot_id": name, "similarity_group": groups[name], "split": split[name]} for name in sorted(rest)
    ]
    if not rows:
        return locked
    table = pd.concat([locked, pd.DataFrame(rows)], ignore_index=True)
    table = table.sort_values("uniprot_id", ignore_index=True)
    table.to_csv(path, index=False)
    return table


def group_folds(groups: pd.Series, n_folds: int = 5) -> pd.Series:
    """Cross-validation fold for each row; every group falls in exactly one fold."""
    from sklearn.model_selection import GroupKFold

    fold = pd.Series(-1, index=groups.index)
    for k, (_, test) in enumerate(GroupKFold(n_splits=n_folds).split(groups, groups=groups)):
        fold.iloc[test] = k
    return fold


def per_protein(df: pd.DataFrame, label: str, prediction: str, protein: str = "protein") -> pd.DataFrame:
    """Pearson and Spearman correlation of label and prediction within each protein (at least 10 variants)."""
    rows = []
    for name, g in df.groupby(protein):
        if len(g) < 10 or g[label].nunique() < 2 or g[prediction].nunique() < 2:
            continue
        rows.append(
            {
                protein: name,
                "variants": len(g),
                "pearson": stats.pearsonr(g[label], g[prediction])[0],
                "spearman": stats.spearmanr(g[label], g[prediction])[0],
            }
        )
    return pd.DataFrame(rows)


def compare_splits(
    df: pd.DataFrame, features: list[str], label: str, protein: str, n_folds: int = 5, seed: int = 0
) -> pd.DataFrame:
    """The same model tested two ways: variants split at random, and proteins held out whole.

    Returns the median per-protein Pearson and Spearman of each. With a random split the model has already
    seen other variants of the same protein, even of the same position, so its score overstates what it can
    do for a new protein.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold, KFold

    ways = {
        "variants split at random": KFold(n_folds, shuffle=True, random_state=seed).split(df),
        "proteins held out whole": GroupKFold(n_folds).split(df, groups=df[protein]),
    }
    results = {}
    for way, splits in ways.items():
        predicted = np.full(len(df), np.nan)
        for train, test in splits:
            model = HistGradientBoostingRegressor(random_state=seed)
            model.fit(df.iloc[train][features], df.iloc[train][label])
            predicted[test] = model.predict(df.iloc[test][features])
        scores = per_protein(df.assign(predicted=predicted), label, "predicted", protein)
        results[way] = {
            "median Pearson": scores["pearson"].median(),
            "median Spearman": scores["spearman"].median(),
        }
    return pd.DataFrame(results).T.round(3)
