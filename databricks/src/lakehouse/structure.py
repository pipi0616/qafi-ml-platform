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
import math

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


def sphere_points(n: int = 100) -> np.ndarray:
    """n points spread evenly over the unit sphere (golden spiral), exactly where Biopython puts them."""
    points, z, longitude = np.zeros((n, 3), dtype=np.float32), 1 - 1 / n, 0.0
    for k in range(n):
        r = (1 - z * z) ** 0.5
        points[k] = math.cos(longitude) * r, math.sin(longitude) * r, z
        z -= 2.0 / n
        longitude += math.pi * (3 - 5**0.5)
    return points


def accessible_surface_area(coords: np.ndarray, radii: np.ndarray, probe: float = 1.4) -> np.ndarray:
    """Solvent-accessible surface area of each atom, Å² (Shrake & Rupley 1973).

    Points are spread over a sphere around each atom (the atom's radius plus a water molecule's); a point is
    accessible unless it lies inside another atom's sphere. The same points and radii as
    Bio.PDB.SASA.ShrakeRupley, so the same areas, but every point is tested at once with a k-d tree instead
    of atom by atom in a Python loop: about twice as fast.
    """
    from scipy.spatial import KDTree

    sphere = sphere_points().astype(float)
    radii = radii + probe
    atoms = KDTree(coords)
    accessible = np.empty(len(coords))
    for first in range(0, len(coords), 1000):  # 1,000 atoms (100,000 points) at a time keeps memory small
        last = min(first + 1000, len(coords))
        points = sphere[None, :, :] * radii[first:last, None, None] + coords[first:last, None, :]
        points = points.reshape(-1, 3)
        tree = KDTree(points, balanced_tree=False, compact_nodes=False)  # quicker to build, used once
        pairs = tree.sparse_distance_matrix(atoms, radii.max(), output_type="ndarray")
        inside = (pairs["v"] <= radii[pairs["j"]]) & (pairs["j"] != first + pairs["i"] // len(sphere))
        buried = np.zeros(len(points), dtype=bool)
        buried[pairs["i"][inside]] = True
        accessible[first:last] = len(sphere) - buried.reshape(-1, len(sphere)).sum(axis=1)
    return accessible * (radii * radii * (4 * np.pi / len(sphere)))


def side_chain_direction(residue) -> np.ndarray:
    """C-alpha -> C-beta; for glycine, the N atom turned -120 degrees about the C-alpha -> C axis.

    The same pseudo C-beta as Bio.PDB.HSExposureCB, so half-sphere exposure matches Biopython exactly.
    """
    from Bio.PDB.vectors import rotaxis

    ca = residue["CA"].get_vector()
    if residue.get_resname() != "GLY":
        return (residue["CB"].get_vector() - ca).get_array()
    n, c = residue["N"].get_vector() - ca, residue["C"].get_vector() - ca
    return n.left_multiply(rotaxis(-np.pi * 120.0 / 180.0, c)).get_array()


def half_sphere_exposure(
    ca: np.ndarray, direction: np.ndarray, radius: float = 13.0
) -> tuple[np.ndarray, ...]:
    """C-alpha neighbours within ``radius`` on the side-chain side (up) and the opposite side (down).

    Hamelryck 2005, as Bio.PDB.HSExposureCB computes it for one unbroken chain, but as one matrix operation
    instead of a Python loop over residue pairs: about 100 times faster on a typical protein.
    """
    from scipy.spatial.distance import cdist

    near = cdist(ca, ca) < radius
    np.fill_diagonal(near, False)
    # j is on i's side-chain side when (CA_j - CA_i) . direction_i > 0: an angle below 90 degrees
    up = direction @ ca.T > np.einsum("ij,ij->i", direction, ca)[:, None]
    return (near & up).sum(axis=1), (near & ~up).sum(axis=1)


def fragment_features(pdb_text: str) -> pd.DataFrame:
    """Per-residue features of one AlphaFold model, numbered as in the file."""
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ATOMIC_RADII
    from scipy.spatial.distance import cdist

    model = PDBParser(QUIET=True).get_structure("model", io.StringIO(pdb_text))[0]
    every = list(model.get_residues())
    atoms = [a for r in every for a in r]
    area = accessible_surface_area(
        np.array([a.coord for a in atoms], dtype=float), np.array([ATOMIC_RADII[a.element] for a in atoms])
    )
    residue_area = np.bincount(np.repeat(np.arange(len(every)), [len(r) for r in every]), weights=area)
    keep = [k for k, r in enumerate(every) if r.id[0] == " " and "CA" in r]
    residues = [every[k] for k in keep]
    sasa = residue_area[keep]
    positions = np.array([r.id[1] for r in residues])
    aa = [THREE_TO_ONE.get(r.get_resname(), "X") for r in residues]
    ca = np.array([r["CA"].coord for r in residues], dtype=float)
    cb = np.array([(r["CB"] if "CB" in r else r["CA"]).coord for r in residues], dtype=float)
    distance = cdist(cb, cb)
    np.fill_diagonal(distance, np.inf)
    close = [np.flatnonzero(row < NEIGHBOUR_DISTANCE) for row in distance]
    hse_up, hse_down = half_sphere_exposure(ca, np.array([side_chain_direction(r) for r in residues]))
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
            "hse_up": hse_up,
            "hse_down": hse_down,
            "ca_x": ca[:, 0].round(3),
            "ca_y": ca[:, 1].round(3),
            "ca_z": ca[:, 2].round(3),
            "neighbours": [positions[i].tolist() for i in close],
            "distances": [distance[k, i].round(2).tolist() for k, i in enumerate(close)],
        }
    )


def fragment_sequence(pdb_text: str) -> tuple[int, str]:
    """Number of the first residue and the one-letter sequence, read straight from the C-alpha lines.

    Far cheaper than computing the features, so every fragment of a long protein can be placed first.
    """
    ca = [line for line in pdb_text.splitlines() if line.startswith("ATOM") and line[12:16] == " CA "]
    return (int(ca[0][22:26]) if ca else 0), "".join(THREE_TO_ONE.get(line[17:20], "X") for line in ca)


def protein_features(fragments: list[tuple[int, str]], uniprot_sequence: str) -> tuple[pd.DataFrame, str]:
    """Per-residue features of one protein on UniProt numbering, and 'ok' or the reason there are none.

    ``fragments`` holds (fragment number, PDB text) pairs: one pair for most proteins, several for long ones.
    All fragments are placed first; then each fragment's features are computed and only the residues it
    supplies are kept, so memory grows with the protein's length, not its number of fragments (titin has 166).
    """
    placed = []  # (fragment, text, shift from file to UniProt numbering, first and last UniProt position)
    for number, text in sorted(fragments):
        first, sequence = fragment_sequence(text)
        start = uniprot_sequence.find(sequence)
        if not sequence or start < 0 or uniprot_sequence.find(sequence, start + 1) >= 0:
            return pd.DataFrame(columns=COLUMNS), "sequence differs from UniProt"
        placed.append((number, text, start + 1 - first, start + 1, start + len(sequence)))
    if not placed:
        return pd.DataFrame(columns=COLUMNS), "no model"
    centrality = np.full(len(uniprot_sequence) + 1, -1)  # both indexed by UniProt position
    source = np.zeros(len(uniprot_sequence) + 1, dtype=int)  # the fragment each residue comes from
    for number, _, _, low, high in placed:
        positions = np.arange(low, high + 1)
        from_end = np.minimum(positions - low, high - positions)
        better = from_end > centrality[positions]  # on a tie the earlier fragment stays
        centrality[positions[better]], source[positions[better]] = from_end[better], number
    pieces = []
    for number, text, shift, _, _ in placed:
        features = fragment_features(text)
        features["position"] += shift
        features = features[source[features["position"]] == number]
        neighbours = [[p + shift for p in n] for n in features["neighbours"]]
        pieces.append(features.assign(neighbours=neighbours, fragment=number))
    combined = pd.concat(pieces, ignore_index=True).sort_values("position")
    covered = len(combined) == len(uniprot_sequence)
    return combined[COLUMNS].reset_index(drop=True), "ok" if covered else "covers part of the sequence"
