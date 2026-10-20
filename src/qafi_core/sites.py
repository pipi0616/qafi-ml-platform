"""A map of what each position of a protein is for (notebook 14d).

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
    by_position["position_is"] = by_position.groupby("uniprot_id", group_keys=False)[
        ["fold_cost", "func_cost"]
    ].apply(lambda p: pd.Series(position_classes(p.fold_cost, p.func_cost, quantile), index=p.index))
    return by_position


def score(models, residues: pd.DataFrame, sequences, read, chunk: int = 200_000) -> pd.DataFrame:
    """Both energies of every possible substitution of some proteins, averaged over the trained models.

    ``models`` are 14c's (model, scaling, codes) triples; ``residues`` is the proteome residue table of the
    proteins (uniprot_id, position, aa and the structure features); ``sequences`` maps UniProt id to
    sequence; ``read`` turns a list of sequences into (log-probabilities, residue vectors) per sequence with
    the pinned language model. Also returns the language model's own log-odds (``esm_delta``), the
    one-number baseline the two energies are compared with.
    """
    from qafi_core import energy, stability

    have = sorted(residues.uniprot_id.unique())
    seqs = [sequences[u] for u in have]
    out = read(seqs)
    logprobs = np.concatenate([r[0] for r in out])
    vectors = np.concatenate([r[1] for r in out]).astype(np.float16)
    index = pd.DataFrame(
        {
            "domain": np.repeat(have, [len(s) for s in seqs]),
            "position": np.concatenate([np.arange(1, len(s) + 1) for s in seqs]),
        }
    )
    structure = residues[
        ["uniprot_id", "position", "aa", "plddt", "rsa", "contacts_8", "contacts_12", "hse_up", "hse_down"]
    ].rename(columns={"uniprot_id": "domain"})
    wild_type = index.merge(structure, on=["domain", "position"], how="left").aa.to_numpy()
    every = pd.DataFrame(
        {
            "domain": np.repeat(index.domain.to_numpy(), 20),
            "position": np.repeat(index.position.to_numpy(), 20),
            "wt_aa": np.repeat(wild_type, 20),
            "mut_aa": np.tile(list(energy.AMINO_ACIDS), len(index)),
        }
    ).dropna(subset=["wt_aa"])
    every = every[every.wt_aa != every.mut_aa].assign(ddg=0.0)
    every = stability.add_features(every, structure, index, logprobs, strict=False)
    fold, func = [], []
    for start in range(0, len(every), chunk):
        part = every.iloc[start : start + chunk]
        f, g = zip(
            *(energy.energies(m, s, part, vectors, energy.FEATURES) for m, s, _ in models), strict=True
        )
        fold.append(np.mean(f, axis=0))
        func.append(np.mean(g, axis=0))
    every["ddg_fold"], every["ddg_func"] = np.concatenate(fold), np.concatenate(func)
    return every.rename(columns={"domain": "uniprot_id"})


def position_classes(fold_cost, func_cost, quantile: float = 0.25) -> np.ndarray:
    """The class of each position of ONE protein from its two costs (lower = costlier, as the energies).

    A position is sensitive in an energy when it is in the worst ``quantile`` of the positions given. The
    same rule labels measured costs, so a predicted map and a measured one can be compared class by class.
    """
    fold_cost, func_cost = pd.Series(np.asarray(fold_cost)), pd.Series(np.asarray(func_cost))
    structural = (fold_cost <= fold_cost.quantile(quantile)).to_numpy()
    functional = (func_cost <= func_cost.quantile(quantile)).to_numpy()
    return np.select(
        [structural & ~functional, functional & ~structural, structural & functional],
        [STRUCTURAL, FUNCTIONAL, MIXED],
        TOLERANT,
    )


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


# --- checking the map against measurements and structures ---------------------------------------------


def agreement(measured, predicted, permutations: int = 10_000, seed: int = 0) -> dict:
    """How well predicted position classes match classes made from measurements, against chance.

    Chance is what two independent labellings with the same class sizes would share; kappa is the agreement
    above chance as a share of what is possible above chance. The permutation p-value shuffles the
    predicted labels. ``recall`` is, for each measured class, the share the prediction recovers, next to the
    share a random labelling with the predicted class sizes would recover.
    """
    measured, predicted = np.asarray(measured), np.asarray(predicted)
    rng = np.random.default_rng(seed)
    agree = float((measured == predicted).mean())
    classes = [STRUCTURAL, FUNCTIONAL, MIXED, TOLERANT]
    share_m = {c: float((measured == c).mean()) for c in classes}
    share_p = {c: float((predicted == c).mean()) for c in classes}
    chance = sum(share_m[c] * share_p[c] for c in classes)
    null = np.array([(measured == rng.permutation(predicted)).mean() for _ in range(permutations)])
    recall = {
        c: (float(((predicted == c) & (measured == c)).sum() / (measured == c).sum()), share_p[c])
        for c in classes
        if (measured == c).any()
    }
    return {
        "positions": len(measured),
        "agreement": agree,
        "chance": chance,
        "kappa": (agree - chance) / (1 - chance),
        "p": float((1 + (null >= agree).sum()) / (1 + permutations)),
        "recall": recall,
    }


def beyond_abundance(abundance, activity) -> np.ndarray:
    """The part of an activity readout that the abundance readout of the same variants does not explain.

    Activity rises with abundance (a protein that is not there cannot work), but not along a straight line.
    An increasing isotonic fit of activity on abundance takes out everything abundance can explain; what is
    left measures function on its own, which is what the functional energy is meant to predict.
    """
    from sklearn.isotonic import IsotonicRegression

    abundance, activity = np.asarray(abundance, float), np.asarray(activity, float)
    fit = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(abundance, activity)
    return activity - fit.predict(abundance)


# BioLiP's ligand codes for metal ions. Many metals hold a fold together (zinc fingers), so they are kept
# apart from the ligands, peptides and nucleic acids a protein binds to do its job.
METALS = frozenset(
    "LI NA K RB CS MG CA SR BA MN FE FE2 FE3 CO CU CU1 CU3 NI ZN CD HG PB AG AU PT MO W V CR AL GA SB".split()
)
_RESIDUE = re.compile(r"([A-Z])(\d+)")


def receptor_numbering(receptor: str, protein: str) -> dict[int, int]:
    """Map 1-based positions of a structure's chain sequence onto 1-based positions of the full protein.

    Structures often cover part of a protein, carry engineered mutations, or use another isoform's
    numbering, so the chain is aligned to the protein rather than trusted. Only positions whose amino acid
    is the same in both are kept.
    """
    start = protein.find(receptor)
    if start >= 0:
        return {i + 1: start + i + 1 for i in range(len(receptor))}
    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner(
        mode="local", match_score=2, mismatch_score=-1, open_gap_score=-5, extend_gap_score=-0.5
    )
    best = aligner.align(receptor, protein)[0]
    out = {}
    for (r0, r1), (p0, _) in zip(*best.aligned, strict=True):
        for i in range(r1 - r0):
            if receptor[r0 + i] == protein[p0 + i]:
                out[r0 + i + 1] = p0 + i + 1
    return out


def read_biolip(path, uniprot_ids=None) -> pd.DataFrame:
    """BioLiP's annotation file (BioLiP.txt.gz), keeping the columns ``biolip_sites`` uses.

    Columns are named by BioLiP's 1-based numbers. ``uniprot_ids`` keeps only the lines of those proteins.
    """
    wanted = [1, 3, 5, 9, 18, 21]
    table = pd.read_csv(path, sep="\t", header=None, quoting=3, dtype=str, usecols=[c - 1 for c in wanted])
    table.columns = wanted
    if uniprot_ids is not None:
        table = table[table[18].isin(set(uniprot_ids))]
    return table.reset_index(drop=True)


def biolip_sites(table: pd.DataFrame, sequences: dict[str, str]) -> pd.DataFrame:
    """Protein positions in contact with a biologically relevant ligand, from BioLiP's annotation file.

    ``table`` holds BioLiP's columns by their 1-based numbers (1 PDB id, 3 resolution, 5 ligand code,
    9 binding residues numbered along the chain sequence, 18 UniProt id, 21 chain sequence);
    ``sequences`` maps UniProt id to the full protein sequence. One row per (structure, ligand, position);
    ligands that are metal ions are flagged.
    """
    rows = []
    numbering = {}
    for pdb, resolution, ligand, residues, uniprot_id, receptor in table[[1, 3, 5, 9, 18, 21]].itertuples(
        index=False
    ):
        protein = sequences.get(uniprot_id)
        if protein is None or not isinstance(residues, str) or not isinstance(receptor, str):
            continue
        key = (uniprot_id, receptor)
        if key not in numbering:
            numbering[key] = receptor_numbering(receptor, protein)
        to_protein = numbering[key]
        for aa, number in _RESIDUE.findall(residues):
            position = to_protein.get(int(number))
            if position is not None and protein[position - 1] == aa:
                rows.append(
                    {
                        "uniprot_id": uniprot_id,
                        "position": position,
                        "pdb": pdb,
                        "resolution": float(resolution),
                        "ligand": ligand,
                        "metal": ligand in METALS,
                    }
                )
    return pd.DataFrame(rows, columns=["uniprot_id", "position", "pdb", "resolution", "ligand", "metal"])


def ligand_names(codes) -> pd.DataFrame:
    """Names of ligand codes from the PDB's chemical dictionary, with DrugBank's name and groups if any.

    One query to RCSB's GraphQL service per 200 codes. A drug is found by name in ``name``, ``drugbank``
    or ``synonyms`` (lower case), never by guessing a code.
    """
    import json
    import urllib.request

    query = (
        "query($ids: [String!]!) { chem_comps(comp_ids: $ids) { rcsb_id chem_comp { name } "
        "drugbank { drugbank_info { name drug_groups } } rcsb_chem_comp_synonyms { name } } }"
    )
    codes = sorted(set(codes))
    rows = []
    for start in range(0, len(codes), 200):
        body = json.dumps({"query": query, "variables": {"ids": codes[start : start + 200]}}).encode()
        request = urllib.request.Request(
            "https://data.rcsb.org/graphql", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            found = json.load(response)["data"]["chem_comps"]
        for item in found:
            if item is None:
                continue
            drugbank = (item.get("drugbank") or {}).get("drugbank_info") or {}
            rows.append(
                {
                    "ligand": item["rcsb_id"],
                    "name": ((item.get("chem_comp") or {}).get("name") or "").lower(),
                    "drugbank": (drugbank.get("name") or "").lower(),
                    "groups": ",".join(drugbank.get("drug_groups") or []),
                    "synonyms": "|".join(
                        (s.get("name") or "").lower() for s in item.get("rcsb_chem_comp_synonyms") or []
                    ),
                }
            )
    return pd.DataFrame(rows, columns=["ligand", "name", "drugbank", "groups", "synonyms"])


def drug_pocket(sites_table: pd.DataFrame, uniprot_id: str, ligands: set[str]) -> tuple[str, list[int]]:
    """The binding residues of one drug on one protein: its best-resolved structure in ``biolip_sites``.

    Returns the PDB id and the protein positions. Structures without a resolution (NMR) come last.
    """
    hits = sites_table[(sites_table.uniprot_id == uniprot_id) & sites_table.ligand.isin(ligands)]
    if hits.empty:
        raise ValueError(f"no structure of {uniprot_id} with {sorted(ligands)}")
    order = hits.assign(r=hits.resolution.where(hits.resolution > 0, np.inf)).sort_values(["r", "pdb"])
    pdb = order.pdb.iloc[0]
    return pdb, sorted(order[order.pdb == pdb].position.unique().tolist())
