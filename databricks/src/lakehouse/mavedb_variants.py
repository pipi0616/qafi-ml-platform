"""MaveDB variants -> silver (notebook 07c): classified, on UniProt numbering, one row per protein variant.

For one score set (one target protein):
  1. classify each row from its protein HGVS: missense (p.Leu12Pro), nonsense (p.Leu12Ter; "Ter" looks like
     an amino acid code, so nonsense is tested first), synonymous (p.Leu12=; p.= or _sy/_wt stand for all
     synonymous variants together). Multi-site, insertion and deletion variants are left out.
  2. put positions on UniProt canonical numbering: identical sequences keep them, a fragment adds its start,
     otherwise each position is located by its unique 15-residue window (as in notebook 05). The reference
     residue must then be the UniProt residue at that position.
  3. one row per protein variant: files written at DNA level list a substitution once per codon, so the
     scores of one protein variant are combined by their median.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
    "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
    "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}  # fmt: skip
NONSENSE = re.compile(r"p\.\(?([A-Z][a-z]{2})(\d+)(?:Ter|\*)\)?")
MISSENSE = re.compile(r"p\.\(?([A-Z][a-z]{2})(\d+)(?!Ter)([A-Z][a-z]{2})\)?")
SYNONYMOUS = re.compile(r"p\.\(?([A-Z][a-z]{2})(\d+)=\)?")
ALL_SYNONYMOUS = {"p.=", "p.(=)", "_sy", "_wt"}
WINDOW = 15  # residues around a position used to find it in the UniProt sequence
# The standard genetic code: codon -> amino acid ("*" = stop). Kept here so Databricks needs no Biopython.
BASES = "TCAG"
CODONS = {
    a + b + c: aa
    for (a, b, c), aa in zip(
        ((a, b, c) for a in BASES for b in BASES for c in BASES),
        "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
        strict=True,
    )
}

OUTPUT_COLUMNS = [
    "kind",
    "assay_position",
    "uniprot_position",
    "position_mapping",
    "ref_aa",
    "alt_aa",
    "score",
    "measurements",
    "ref_matches_uniprot",
]


def classify(hgvs_pro: str | None) -> tuple[str, int | None, str | None, str | None]:
    """(kind, position, reference residue, alternative residue) of one protein HGVS string."""
    text = re.sub(r"^[A-Za-z0-9_.]+:(?=p\.)", "", (hgvs_pro or "").strip())  # drop "ENSP0000...:" prefixes
    if text in ALL_SYNONYMOUS:
        return "synonymous", None, None, None
    if (m := NONSENSE.fullmatch(text)) and m[1] in THREE_TO_ONE:
        return "nonsense", int(m[2]), THREE_TO_ONE[m[1]], "*"
    if (m := MISSENSE.fullmatch(text)) and m[1] in THREE_TO_ONE and m[3] in THREE_TO_ONE and m[1] != m[3]:
        return "missense", int(m[2]), THREE_TO_ONE[m[1]], THREE_TO_ONE[m[3]]
    if (m := SYNONYMOUS.fullmatch(text)) and m[1] in THREE_TO_ONE:
        return "synonymous", int(m[2]), THREE_TO_ONE[m[1]], THREE_TO_ONE[m[1]]
    return "other", None, None, None


def target_protein(sequence: str | None, sequence_type: str | None) -> str:
    """The target as a protein sequence: DNA is translated (standard genetic code) up to the first stop."""
    if not sequence:
        return ""
    if (sequence_type or "").lower() != "dna":
        return sequence.upper()
    dna = sequence.upper()
    protein = []
    for i in range(0, len(dna) - 2, 3):
        residue = CODONS.get(dna[i : i + 3], "X")
        if residue == "*":
            break
        protein.append(residue)
    return "".join(protein)


def position_map(target: str, uniprot: str) -> tuple[dict[int, int], str]:
    """Target position -> UniProt position (both 1-based), and how they were matched."""
    if not target or not uniprot:
        return {}, "unmapped"
    if target == uniprot:
        return {i: i for i in range(1, len(target) + 1)}, "identical"
    start = uniprot.find(target)
    if start >= 0 and uniprot.find(target, start + 1) < 0:
        return {i: i + start for i in range(1, len(target) + 1)}, "fragment"
    mapping = {}
    for i in range(1, len(target) + 1):
        first = max(1, i - WINDOW // 2)
        window = target[first - 1 : first - 1 + WINDOW]
        hit = uniprot.find(window)
        if len(window) == WINDOW and hit >= 0 and uniprot.find(window, hit + 1) < 0:
            mapping[i] = hit + 1 + (i - first)
    return mapping, "window" if mapping else "unmapped"


def numbering(
    positions: pd.Series,
    refs: pd.Series,
    target: str,
    uniprot: str,
    offset: int | None,
    sequence_type: str | None,
) -> tuple[dict[int, int], str]:
    """The numbering under which the variants' reference residues best agree with UniProt.

    MAVE-HGVS positions should count from the start of the target sequence, but some submitters numbered
    them along the full-length protein, or MaveDB records an offset to the reference. Each candidate is
    tried and the one where most reference residues match UniProt wins (ties go to the sequence match).
    """
    candidates = [position_map(target, uniprot)[::-1]]
    candidates.append(("UniProt numbering", {i: i for i in range(1, len(uniprot) + 1)}))
    if offset:
        steps = {offset, offset // 3} if (sequence_type or "").lower() == "dna" else {offset}
        for step in sorted(steps):
            candidates.append(("MaveDB offset", {i: i + step for i in range(1, len(uniprot) + 1)}))

    def agreement(mapping: dict[int, int]) -> float:
        hits = [
            0 < mapping.get(int(p), 0) <= len(uniprot) and uniprot[mapping[int(p)] - 1] == r
            for p, r in zip(positions, refs, strict=True)
        ]
        return sum(hits) / len(hits) if hits else 0.0

    name, mapping = max(candidates, key=lambda c: agreement(c[1]))  # max keeps the first of equal scores
    return mapping, name if mapping else "unmapped"


def process_score_set(
    rows: pd.DataFrame, target: str, uniprot: str, offset: int | None = None, sequence_type: str | None = None
) -> pd.DataFrame:
    """Classified variants of one score set on UniProt numbering, one row per protein variant.

    ``rows`` has the columns ``hgvs_pro`` and ``score`` (text, as in bronze).
    """
    parsed = pd.DataFrame(
        [classify(h) for h in rows["hgvs_pro"]],
        columns=["kind", "assay_position", "ref_aa", "alt_aa"],
        index=rows.index,
    )
    parsed["score"] = pd.to_numeric(rows["score"], errors="coerce")
    parsed = parsed[(parsed["kind"] != "other") & parsed["score"].notna()]
    if parsed.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    located = parsed[parsed["assay_position"].notna()]
    mapping, method = numbering(
        located["assay_position"], located["ref_aa"], target, uniprot, offset, sequence_type
    )
    parsed["uniprot_position"] = parsed["assay_position"].map(
        lambda p: mapping.get(int(p)) if pd.notna(p) else None
    )
    parsed["position_mapping"] = np.where(parsed["uniprot_position"].notna(), method, "unmapped")
    parsed.loc[parsed["assay_position"].isna(), "position_mapping"] = "all synonymous"
    keys = ["kind", "assay_position", "uniprot_position", "position_mapping", "ref_aa", "alt_aa"]
    out = parsed.groupby(keys, dropna=False)["score"].agg(score="median", measurements="size").reset_index()
    out["ref_matches_uniprot"] = [
        (uniprot[int(pos) - 1] == ref) if pd.notna(pos) and ref and 0 < int(pos) <= len(uniprot) else None
        for pos, ref in zip(out["uniprot_position"], out["ref_aa"], strict=True)
    ]
    for column in ("assay_position", "uniprot_position"):
        out[column] = out[column].astype("Int64")
    return out[OUTPUT_COLUMNS]
