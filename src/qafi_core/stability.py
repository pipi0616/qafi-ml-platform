"""Folding free energies in kcal/mol (notebook 14a): the mega-scale dataset and its published split.

Chapters 07-09 predict a rank: a number between 0 and 1 that orders the variants of one protein. This
chapter predicts a physical quantity instead, the change in folding free energy, in kcal/mol, so the
prediction can be compared with a measurement and with the published stability predictors.

The data is Tsuboyama et al. 2023 (Nature), 776,298 high-quality folding stabilities of 479 small domains
measured by cDNA display proteolysis. The split is the one published with ThermoMPNN: domains clustered by
MMseqs2 at 25% sequence identity, whole clusters sent to train, validation or test. Reusing it is what
makes our number comparable with ThermoMPNN's and SPURS', which both report on exactly these 28 test
domains; inventing our own split would throw that away.

``megascale`` applies the three filters the ThermoMPNN paper describes and returns one row per single
substitution. ``load_splits`` reads the published split file. Sign convention is Tsuboyama's throughout
(positive ddG = stabilising); ``THERMOMPNN_SIGN`` documents the opposite convention used by ThermoMPNN.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import pandas as pd

# Tsuboyama's ddG_ML is positive when the mutation stabilises the fold. ThermoMPNN negates it
# (datasets.py: ``ddG = -float(row.ddG_ML)``), so its published numbers are on the opposite sign.
# Correlations are unaffected in magnitude; RMSE is not. We keep Tsuboyama's sign and say so.
THERMOMPNN_SIGN = -1

MUTATION = re.compile(r"^([A-Z])(\d+)([A-Z])$")

# Counts published in the ThermoMPNN paper (PNAS 2024), reproduced by ``megascale`` and checked in the
# tests: every filtering step has to land on these, or the data or the filters have changed.
PUBLISHED_COUNTS = {
    "rows": 776_298,  # the file itself: datasets 2 and 3 together
    "with_ddg": 607_839,  # after dropping ddG_ML == '-'
    "single_substitutions": 391_090,  # after dropping insertions, deletions and multiple mutations
    # rows belonging to the 298 domains of the published split, wild type included:
    "on_split_domains": 272_712,
    "test_rows": 28_312,  # the test domains' share of those, wild type included
    "test_mutations": 28_172,  # the same without the 140 wild-type rows: what a model actually predicts
    "domains": {"train": 239, "val": 31, "test": 28},
}


class _NoCodeUnpickler(pickle.Unpickler):
    """An unpickler that refuses everything except the numpy array machinery.

    Loading a pickle normally runs whatever code it names: it is a program, not a data file. The published
    split is a dict of lists and numpy arrays of domain names, so allowing exactly three numpy entry points
    and refusing every other import reads it without giving the file a way to run anything.
    """

    # numpy 1 wrote "numpy.core", numpy 2 writes "numpy._core"; the published file is from numpy 1 and a
    # file written here is from numpy 2, so both spellings have to be allowed.
    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"refused to import {module}.{name} while reading a split file")
        return getattr(__import__(module, fromlist=[name]), name)


def load_splits(path: str | Path) -> dict[str, list[str]]:
    """The published train / val / test domain names, read from ThermoMPNN's ``mega_splits.pkl``.

    The file also holds five cross-validation folds and a FireProt-specific training list; only the three
    main splits are returned, as plain lists of ``<name>.pdb`` strings.
    """
    with open(path, "rb") as f:
        splits = _NoCodeUnpickler(f).load()
    return {name: [str(x) for x in splits[name]] for name in ("train", "val", "test")}


def megascale(csv_path: str | Path, splits: dict[str, list[str]]) -> pd.DataFrame:
    """One row per single substitution of the split's domains, with its folding free energy change.

    The three filters are the ones the ThermoMPNN paper describes, in its order:
      1. drop ``ddG_ML == '-'``: the fit did not give a reliable free energy;
      2. drop insertions, deletions and multiple mutations (``ins``, ``del``, ``:`` in ``mut_type``);
      3. keep only the 298 domains of the published split, which excludes the perturbed wild-type
         backgrounds the authors also measured, and drop the wild-type rows themselves.

    Columns: domain, split, mut_type, position (1-based in the domain), wt_aa, mut_aa, ddg (kcal/mol,
    positive = stabilising), dg (the variant's own folding free energy) and seq (the wild-type sequence).
    """
    df = pd.read_csv(
        csv_path,
        usecols=["ddG_ML", "mut_type", "WT_name", "aa_seq", "dG_ML"],
        low_memory=False,
    )
    df = df[df.ddG_ML != "-"]
    df = df[~df.mut_type.str.contains("ins|del|:", regex=True)]

    where = {name: split for split, names in splits.items() for name in names}
    df = df[df.WT_name.isin(where)]

    wild_type = df.mut_type == "wt"
    sequences = df[wild_type].groupby("WT_name").aa_seq.first()
    df = df[~wild_type]

    parsed = df.mut_type.str.extract(MUTATION)
    if parsed.isna().any(axis=None):
        bad = df.mut_type[parsed.isna().any(axis=1)].unique()[:5]
        raise ValueError(f"mut_type values that are not a single substitution: {list(bad)}")

    out = pd.DataFrame(
        {
            "domain": df.WT_name.values,
            "split": df.WT_name.map(where).values,
            "mut_type": df.mut_type.values,
            "position": parsed[1].astype(int).values,
            "wt_aa": parsed[0].values,
            "mut_aa": parsed[2].values,
            "ddg": df.ddG_ML.astype(float).values,
            "dg": df.dG_ML.astype(float).values,
            "seq": df.WT_name.map(sequences).values,
        }
    )
    check_positions(out)
    return out.reset_index(drop=True)


def check_positions(df: pd.DataFrame) -> None:
    """Raise unless every row's wild-type amino acid is the one its sequence has at that position."""
    at_position = [seq[pos - 1] for seq, pos in zip(df.seq, df.position, strict=True)]
    wrong = df[pd.Series(at_position, index=df.index) != df.wt_aa]
    if len(wrong):
        raise ValueError(
            f"{len(wrong)} rows name an amino acid the sequence does not have, e.g. {wrong.iloc[0].to_dict()}"
        )


# --- the model ------------------------------------------------------------------------------------

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AMINO_ACIDS)}

# Beside the ESM-2 vector of the mutated residue: how buried and how well predicted the position is,
# how crowded its neighbourhood, and what the language model thinks of the substitution.
FEATURES = [
    "rsa",
    "plddt",
    "contacts_8",
    "contacts_12",
    "hse_up",
    "hse_down",
    "esm_delta",
    "esm_wt",
    "esm_entropy",
]

# Same shape as 08a's network, with one difference that matters: the target is kcal/mol, not a
# standardised rank, so the loss is a plain mean squared error in physical units and the number the
# network writes out is the prediction itself.
SETTINGS = {"hidden": 512, "dropout": 0.2, "epochs": 4, "lr": 3e-4, "weight_decay": 0.01, "batch": 4096}
SEEDS = (0, 1, 2)

# Published on the very same 28 test domains, for the chapter's comparison table.
PUBLISHED_TEST_SCORES = {
    # ThermoMPNN: Dieckhaus et al., PNAS 121(6):e2314853121 (2024), Table 1; median per-protein
    # Spearman 0.77 is from SPURS' Fig. 2b, which re-ran it.
    "ThermoMPNN": {"spearman": 0.725, "pearson": 0.754, "rmse": 0.708, "median_spearman": 0.77},
    # SPURS: Li & Luo, Nat Commun 17:891 (2025), Supplementary Table 1 and Fig. 2b. No RMSE published.
    "SPURS": {"spearman": 0.74, "pearson": 0.78, "median_spearman": 0.83},
}


def add_features(
    mutations: pd.DataFrame,
    structure: pd.DataFrame,
    residue_index: pd.DataFrame,
    logprobs,
    strict: bool = True,
) -> pd.DataFrame:
    """Join each mutation to its residue's structure features and ESM-2 log-probabilities.

    ``residue_index`` has one row per residue (domain, position) in the order of the ``logprobs`` and
    embedding matrices; the returned frame gains a ``row`` column pointing into them, so the caller can
    index the embeddings without another join.
    """
    import numpy as np

    index = residue_index.reset_index(drop=True)
    rows = index.assign(row=np.arange(len(index))).set_index(["domain", "position"]).row
    where = pd.MultiIndex.from_arrays([mutations.domain, mutations.position])
    out = mutations.copy()
    out["row"] = rows.reindex(where).values
    if out.row.isna().any():
        raise ValueError(f"{int(out.row.isna().sum())} mutations have no residue in the index")
    out["row"] = out.row.astype(int)

    got = structure.set_index(["domain", "position"]).reindex(where)
    for column in ["rsa", "plddt", "contacts_8", "contacts_12", "hse_up", "hse_down"]:
        out[column] = got[column].values
    if strict and not (got["aa"].values == out.wt_aa.values).all():
        raise ValueError("the structure's amino acid differs from the mutation's wild type")

    wt = out.wt_aa.map(AA_INDEX).values
    alt = out.mut_aa.map(AA_INDEX).values
    out["esm_delta"] = logprobs[out.row.values, alt] - logprobs[out.row.values, wt]
    out["esm_wt"] = logprobs[out.row.values, wt]
    out["esm_entropy"] = -(np.exp(logprobs) * logprobs)[out.row.values].sum(1)
    # The zero-parameter baseline the chapter has to beat: conservation weighted by how buried the
    # position is (Tsishyn et al. 2025, "residue conservation and solvent accessibility are (almost)
    # all you need"), with ESM-2 log-odds standing in for the alignment's log-odds ratio.
    out["buried"] = 1 - out.rsa.clip(0, 1)
    out["rsalor"] = out.buried * out.esm_delta
    return out


def per_domain(df: pd.DataFrame, prediction: str, label: str = "ddg", minimum: int = 10) -> pd.DataFrame:
    """Spearman and Pearson correlation inside each domain with at least ``minimum`` mutations."""
    from scipy import stats

    rows = [
        {
            "domain": domain,
            "mutations": len(g),
            "spearman": stats.spearmanr(g[label], g[prediction]).statistic,
            "pearson": stats.pearsonr(g[label], g[prediction])[0],
        }
        for domain, g in df.groupby("domain")
        if len(g) >= minimum
    ]
    return pd.DataFrame(rows)


def score(df: pd.DataFrame, prediction: str, label: str = "ddg") -> dict:
    """How good one prediction is: pooled correlations, the median domain, and the error in kcal/mol."""
    import numpy as np
    from scipy import stats

    by_domain = per_domain(df, prediction, label)
    return {
        "spearman": stats.spearmanr(df[label], df[prediction]).statistic,
        "pearson": stats.pearsonr(df[label], df[prediction])[0],
        "median_spearman": by_domain.spearman.median(),
        "median_pearson": by_domain.pearson.median(),
        "rmse": float(np.sqrt(((df[label] - df[prediction]) ** 2).mean())),
        "domains": len(by_domain),
        "mutations": len(df),
    }


def device():
    """The fastest processor available: an NVIDIA GPU, a Mac's GPU, or the CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else torch.device("cpu"))


def _module():
    import torch
    from torch import nn

    class Energy(nn.Module):
        """ESM-2's vector of the mutated residue plus a few features, read out as kcal/mol.

        The last layer has one output per amino acid and only the mutant's output is kept, so the
        network answers "what would this position cost if it were each of the twenty" in one pass.
        """

        def __init__(self, width: int, n_features: int, hidden: int, dropout: float):
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.body = nn.Sequential(
                nn.Linear(width + n_features, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, len(AMINO_ACIDS)),
            )

        def forward(self, vectors, features, alt):
            x = torch.cat([self.norm(vectors), features], dim=1)
            return self.body(x).gather(1, alt[:, None]).squeeze(1)

    return Energy


def _tensors(df: pd.DataFrame, embeddings, mean, std, dev):
    import numpy as np
    import torch

    features = (df[FEATURES].values.astype(np.float32) - mean) / std
    return (
        torch.tensor(embeddings[df.row.values].astype(np.float32), device=dev),
        torch.tensor(features.astype(np.float32), device=dev),
        torch.tensor(df.mut_aa.map(AA_INDEX).values, device=dev),
        torch.tensor(df.ddg.values.astype(np.float32), device=dev),
    )


def fit(df: pd.DataFrame, embeddings, settings: dict | None = None, seed: int = 0):
    """Train one network on ``df``; returns it together with the feature scaling it was trained with."""
    import numpy as np
    import torch
    from torch import nn

    settings = {**SETTINGS, **(settings or {})}
    dev = device()
    torch.manual_seed(seed)
    values = df[FEATURES].values.astype(np.float32)
    mean, std = values.mean(0), values.std(0) + 1e-6
    vectors, features, alt, target = _tensors(df, embeddings, mean, std, dev)

    model = _module()(vectors.shape[1], len(FEATURES), settings["hidden"], settings["dropout"]).to(dev)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    for _ in range(settings["epochs"]):
        model.train()
        order = torch.randperm(len(target), device=dev)
        for start in range(0, len(target), settings["batch"]):
            rows = order[start : start + settings["batch"]]
            optimiser.zero_grad()
            nn.functional.mse_loss(model(vectors[rows], features[rows], alt[rows]), target[rows]).backward()
            optimiser.step()
    return model, (mean, std)


def predict(model, scaling, df: pd.DataFrame, embeddings):
    """Predicted ddG in kcal/mol for every row of ``df``."""
    import torch

    mean, std = scaling
    vectors, features, alt, _ = _tensors(df, embeddings, mean, std, device())
    model.eval()
    with torch.inference_mode():
        return model(vectors, features, alt).cpu().numpy()


def fit_ensemble(df: pd.DataFrame, embeddings, settings: dict | None = None, seeds=SEEDS) -> list:
    """One network per seed. Averaging a few runs of the same design was worth a little in 08a."""
    return [fit(df, embeddings, settings, seed) for seed in seeds]


def predict_ensemble(models: list, df: pd.DataFrame, embeddings):
    """The average prediction of an ensemble, in kcal/mol."""
    import numpy as np

    return np.mean([predict(model, scaling, df, embeddings) for model, scaling in models], axis=0)
