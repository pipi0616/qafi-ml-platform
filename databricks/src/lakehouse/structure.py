"""AlphaFold structures -> per-residue features and neighbourhoods (notebook 07d).

For each protein, from its AlphaFold model:
  plddt                   AlphaFold's confidence in the residue's position, 0-100 (the B-factor column)
  sasa, rsa               solvent-accessible surface area (Shrake-Rupley), and the same relative to the
                          residue type's maximum (Tien et al. 2013): 0 = buried, about 1 = fully exposed
  contacts_8, contacts_12 residues whose C-beta (C-alpha for glycine) lies within 8 or 12 Å
  hse_up, hse_down        half-sphere exposure (Hamelryck 2005): C-alpha neighbours within 13 Å on the side
                          chain's side, and on the opposite side
  neighbours, distances   residues within 10 Å (C-beta): the edges of the protein graph for notebook 08
  ca_x, ca_y, ca_z        C-alpha coordinates (for long proteins, each fragment has its own frame)

Proteins longer than 2,700 residues come as overlapping fragments. Each fragment is placed on UniProt
numbering by finding its sequence in the UniProt sequence, and each residue takes its features from the
fragment in which it lies furthest from the ends, where the prediction is most reliable.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}  # fmt: skip
# Theoretical maximum accessible surface area per residue type, Å² (Tien et al. 2013, PLoS One 8:e80635)
MAX_ASA = {
    "A": 129, "R": 274, "N": 195, "D": 193, "C": 167, "Q": 225, "E": 223, "G": 104, "H": 224, "I": 197,
    "L": 201, "K": 236, "M": 224, "F": 240, "P": 159, "S": 155, "T": 172, "W": 285, "Y": 263, "V": 174,
}  # fmt: skip
NEIGHBOUR_DISTANCE = 10.0  # Å between C-beta atoms: an edge of the protein graph
COLUMNS = [
    "position", "aa", "plddt", "sasa", "rsa", "contacts_8", "contacts_12", "hse_up", "hse_down",
    "ca_x", "ca_y", "ca_z", "neighbours", "distances", "fragment",
]  # fmt: skip


def fragment_features(pdb_text: str) -> pd.DataFrame:
    """Per-residue features of one AlphaFold model, numbered as in the file."""
    from Bio.PDB import HSExposureCB, PDBParser
    from Bio.PDB.SASA import ShrakeRupley

    model = PDBParser(QUIET=True).get_structure("model", io.StringIO(pdb_text))[0]
    ShrakeRupley().compute(model, level="R")
    HSExposureCB(model, radius=13.0)  # writes EXP_HSE_B_U / _D into each residue's xtra
    residues = [r for r in model.get_residues() if r.id[0] == " " and "CA" in r]
    positions = np.array([r.id[1] for r in residues])
    aa = [THREE_TO_ONE.get(r.get_resname(), "X") for r in residues]
    ca = np.array([r["CA"].coord for r in residues], dtype=float)
    cb = np.array([(r["CB"] if "CB" in r else r["CA"]).coord for r in residues], dtype=float)
    distance = np.linalg.norm(cb[:, None, :] - cb[None, :, :], axis=-1)
    np.fill_diagonal(distance, np.inf)
    sasa = np.array([r.sasa for r in residues], dtype=float)
    close = [np.flatnonzero(row < NEIGHBOUR_DISTANCE) for row in distance]
    return pd.DataFrame(
        {
            "position": positions,
            "aa": aa,
            "plddt": [r["CA"].get_bfactor() for r in residues],
            "sasa": sasa.round(2),
            "rsa": [
                round(s / MAX_ASA[a], 4) if a in MAX_ASA else None for s, a in zip(sasa, aa, strict=True)
            ],
            "contacts_8": (distance < 8).sum(axis=1),
            "contacts_12": (distance < 12).sum(axis=1),
            "hse_up": [r.xtra.get("EXP_HSE_B_U") for r in residues],
            "hse_down": [r.xtra.get("EXP_HSE_B_D") for r in residues],
            "ca_x": ca[:, 0].round(3),
            "ca_y": ca[:, 1].round(3),
            "ca_z": ca[:, 2].round(3),
            "neighbours": [positions[i].tolist() for i in close],
            "distances": [distance[k, i].round(2).tolist() for k, i in enumerate(close)],
        }
    )


def protein_features(fragments: list[tuple[int, str]], uniprot_sequence: str) -> tuple[pd.DataFrame, str]:
    """Per-residue features of one protein on UniProt numbering, and 'ok' or the reason there are none.

    ``fragments`` holds (fragment number, PDB text) pairs: one pair for most proteins, several for long ones.
    """
    pieces = []
    for number, text in sorted(fragments):
        features = fragment_features(text)
        sequence = "".join(features["aa"])
        start = uniprot_sequence.find(sequence)
        if not sequence or start < 0 or uniprot_sequence.find(sequence, start + 1) >= 0:
            return pd.DataFrame(columns=COLUMNS), "sequence differs from UniProt"
        shift = start + 1 - int(features["position"].iloc[0])  # file numbering -> UniProt numbering
        features["position"] += shift
        features["neighbours"] = [[p + shift for p in n] for n in features["neighbours"]]
        features["fragment"] = number
        first, last = features["position"].min(), features["position"].max()
        features["centrality"] = np.minimum(features["position"] - first, last - features["position"])
        pieces.append(features)
    if not pieces:
        return pd.DataFrame(columns=COLUMNS), "no model"
    combined = (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["position", "centrality"], ascending=[True, False])
        .drop_duplicates("position")
    )
    covered = len(combined) == len(uniprot_sequence)
    return combined[COLUMNS].reset_index(drop=True), "ok" if covered else "covers part of the sequence"
