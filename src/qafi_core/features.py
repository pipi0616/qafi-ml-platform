"""The features of a variant (chapter 07f): which numbers the model sees, and the lookup tables behind them.

Three families, all computable from raw data for any human protein, so training and proteome-wide
prediction use the same values:
  - substitution     what changes when one amino acid replaces another (lookup tables, below)
  - language model   how surprising the new amino acid is to ESM-2, and how conserved the position
  - structure        where the residue sits in the AlphaFold model, and how conserved its neighbours are

The numbers themselves are computed in SQL by dbt (``gold.dim_residue`` and ``gold.fct_variant_features``).
This module holds what that SQL cannot: the two lookup tables, which dbt loads as seeds, and the list of
feature columns, which the models read.
"""

from __future__ import annotations

import pandas as pd

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # the order of ESM-2's 20 log-probabilities (lakehouse.language_model)

# Residue volume in cubic angstrom (Zamyatnin 1972), the table the QAFI paper used.
VOLUME = {
    "A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9, "G": 60.1, "H": 153.2, "I": 166.7, "K": 168.6,
    "L": 166.7, "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4, "S": 89.0, "T": 116.1, "V": 140.0,
    "W": 227.8, "Y": 193.6,
}  # fmt: skip
CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1}  # at neutral pH; histidine is mostly uncharged

FEATURE_GROUPS = {
    "substitution": ["blosum62", "d_hydrophobicity", "d_volume", "d_charge"],
    "language_model": ["esm_delta", "esm_wt_logprob", "esm_entropy"],
    "structure": ["plddt", "rsa", "contacts_8", "contacts_12", "hse_up", "hse_down"],
    "neighbourhood": ["neighbour_entropy", "d_volume_buried", "relative_position"],
}
FEATURES = [name for group in FEATURE_GROUPS.values() for name in group]


def select(groups: str = "all") -> list[str]:
    """Feature columns of the named groups, e.g. ``"language_model,structure"``; ``"all"`` gives every one."""
    if groups == "all":
        return list(FEATURES)
    return [name for group in groups.split(",") for name in FEATURE_GROUPS[group.strip()]]


def amino_acid_table() -> pd.DataFrame:
    """One row per amino acid: Kyte-Doolittle hydrophobicity, volume and charge (the seed ``amino_acid``)."""
    from Bio.SeqUtils.ProtParamData import kd

    return pd.DataFrame(
        {
            "aa": list(AMINO_ACIDS),
            "hydrophobicity": [kd[aa] for aa in AMINO_ACIDS],
            "volume": [VOLUME[aa] for aa in AMINO_ACIDS],
            "charge": [CHARGE.get(aa, 0) for aa in AMINO_ACIDS],
        }
    )


def blosum62_table() -> pd.DataFrame:
    """One row per ordered pair of amino acids with its BLOSUM62 score (the seed ``blosum62``)."""
    from Bio.Align import substitution_matrices

    matrix = substitution_matrices.load("BLOSUM62")
    return pd.DataFrame(
        [
            {"ref_aa": ref, "alt_aa": alt, "score": int(matrix[ref][alt])}
            for ref in AMINO_ACIDS
            for alt in AMINO_ACIDS
        ]
    )
