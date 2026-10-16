"""Two energies instead of one score (notebook 14c): the physical readout layer.

A mutation changes two free energies of a protein:
  ddg_fold  how much more (or less) stable the folded state is, in kcal/mol
  ddg_func  how much more (or less) the folded protein binds or does its job

Experiments never measure these directly. They measure a phenotype -- how much protein a cell keeps, how
well it binds, how active it is -- and the phenotype depends on the energies through the share of molecules
in each state (Boltzmann). This module turns energies into those shares, and the shares into each assay's
readout, so a network that predicts the two energies can be trained on every kind of assay at once:

  abundance assays (Domainome, VAMP-seq, AbundancePCA)  see the folded share only
  activity and binding assays                           see the share that is folded AND functional

Sign convention, as in Tsuboyama 2023 and 14a: free energies are stabilities, positive = favourable. A
wild type with g_fold = 2 kcal/mol is folded 97% of the time at 30 C; a mutation with ddg_fold = -3 makes
it g_fold = -1 and leaves 16% folded.

This is the thermodynamic model of Faure et al. 2022 and MoCHI (Faure & Lehner 2024), which fit these
energies to one protein's measurements at a time. What is new here is that the energies come from a
network that has never seen the protein, so they can be predicted for any protein.
"""

from __future__ import annotations

import math

import pandas as pd

RT = 0.6  # kcal/mol: gas constant x temperature at about 30 C, where the yeast selections are grown


def fraction_folded(g_fold, rt: float = RT):
    """Share of molecules folded, for a folding stability ``g_fold`` (kcal/mol, positive = stable).

    Two states, unfolded and folded, with weights 1 and exp(g_fold / RT): the folded share is a sigmoid.
    Works on floats, numpy arrays and torch tensors alike.
    """
    return _sigmoid(g_fold / rt)


def fraction_functional(g_fold, g_func, rt: float = RT):
    """Share of molecules folded and functional (bound, active), three states.

    Unfolded, folded and folded-functional have weights 1, exp(g_fold/RT) and exp((g_fold + g_func)/RT),
    so the functional share is 1 / (1 + exp(-g_func/RT) * (1 + exp(-g_fold/RT))). A protein that cannot
    fold cannot function; a very stable one is limited only by g_func.
    """
    return 1 / (1 + _exp(-g_func / rt) * (1 + _exp(-g_fold / rt)))


def _sigmoid(x):
    if hasattr(x, "sigmoid"):  # torch
        return x.sigmoid()
    try:
        import numpy as np

        return 1 / (1 + np.exp(-x))
    except ImportError:  # pragma: no cover
        return 1 / (1 + math.exp(-x))


def _exp(x):
    if hasattr(x, "exp"):  # torch
        return x.exp()
    import numpy as np

    return np.exp(x)


# --- the model ------------------------------------------------------------------------------------

ABUNDANCE, ACTIVITY, DDG = "abundance", "activity", "ddg"


def build(
    width: int, n_features: int, n_proteins: int, n_datasets: int, hidden: int = 512, dropout: float = 0.2
):
    """The two-energy network with its readout layer.

    ``width`` is the residue vector's size (ESM-2: 1280), ``n_features`` the extra features per mutation.
    Learned besides the network: a wild-type folding stability per protein, and per dataset a wild-type
    functional energy plus the scale and offset that map a share of molecules onto that assay's numbers.
    """
    import torch
    from torch import nn

    class TwoEnergies(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.body = nn.Sequential(
                nn.Linear(width + n_features, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            # one output per amino acid and per energy: what each of the 20 would cost at this position
            self.fold = nn.Linear(128, 20)
            self.func = nn.Linear(128, 20)
            # wild-type energies start where typical small domains are: stable, and mostly functional
            self.g_fold_wt = nn.Parameter(torch.full((n_proteins,), 2.0))
            self.g_func_wt = nn.Parameter(torch.full((n_datasets,), 1.0))
            self.scale = nn.Parameter(torch.ones(n_datasets))
            self.offset = nn.Parameter(torch.zeros(n_datasets))

        def energies(self, vectors, features, alt):
            """Predicted ddg_fold and ddg_func of each mutation, kcal/mol (positive = favourable)."""
            h = self.body(torch.cat([self.norm(vectors), features], dim=1))
            pick = alt[:, None]
            return self.fold(h).gather(1, pick).squeeze(1), self.func(h).gather(1, pick).squeeze(1)

        def forward(self, vectors, features, alt, protein, dataset, kind, pin_folding: bool = False):
            """What each row's assay would read: ddG itself for ddg rows, a scaled share for the others.

            ``kind`` per row: 0 a ddG in kcal/mol, 1 abundance, 2 activity, 3 a stability score (KIND_CODE).

            With ``pin_folding``, an activity row cannot change the folding energy: only the assays that
            see folding alone (abundance and kcal/mol) decide it, and the functional energy is left to
            explain whatever activity shows beyond that. This is the two-step fit of MoCHI, written as one
            network; without it the two energies are free to trade off and come out nearly identical.
            """
            ddg_fold, ddg_func = self.energies(vectors, features, alt)
            if pin_folding:
                ddg_fold = torch.where(kind == KIND_CODE[ACTIVITY], ddg_fold.detach(), ddg_fold)
            g_fold = self.g_fold_wt[protein] + ddg_fold
            g_func = self.g_func_wt[dataset] + ddg_func
            share = torch.where(kind == 1, fraction_folded(g_fold), fraction_functional(g_fold, g_func))
            share = torch.where(kind == 3, ddg_fold, share)  # stability scores are linear in the energy
            readout = self.offset[dataset] + self.scale[dataset] * share
            return torch.where(kind == 0, ddg_fold, readout)

    return TwoEnergies()


KIND_CODE = {DDG: 0, ABUNDANCE: 1, ACTIVITY: 2}


# --- which readout each assay gets -----------------------------------------------------------------

STABILITY_SCORE = "stability_score"
KIND_CODE[STABILITY_SCORE] = 3

# Words in a MaveDB title that mean the assay counts how much protein the cell keeps (it folds, is not
# degraded, reaches the surface), rather than what the protein does. Checked by hand against every title
# in the training and test sets of 14c; tests/unit/test_energy.py pins the decisions that matter.
ABUNDANCE_WORDS = (
    "vamp-seq",
    "abundance",
    "surface expression",
    "cell surface",
    "surface localization",
    "protein stability",
)


def readout_kind(data_kind: str, title: str | None) -> str:
    """How an assay sees the two energies: abundance, activity or a stability score.

    Domainome is AbundancePCA throughout. The ProteinGym assays we use are all Tsuboyama 2023 stabilities,
    already on a scale linear in the folding energy (a stability score, not a share of molecules). For the
    other MaveDB assays the title decides: abundance words above, otherwise activity.
    """
    if data_kind == "domainome":
        return ABUNDANCE
    if data_kind == "proteingym":
        return STABILITY_SCORE
    text = (title or "").lower()
    return ABUNDANCE if any(word in text for word in ABUNDANCE_WORDS) else ACTIVITY


# --- training -------------------------------------------------------------------------------------

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SETTINGS = {
    "hidden": 512,
    "dropout": 0.2,
    "epochs": 4,
    "lr": 3e-4,
    "lr_readout": 3e-2,  # the per-protein and per-dataset parameters are few and need bigger steps
    "weight_decay": 0.01,
    "batch": 4096,
    "kcal_weight": 0.25,  # share of the loss given to the kcal/mol stability data (14a's Tsuboyama set)
    # Most positions of a protein are not functional sites, so the functional energy should stay at zero
    # unless the data demand otherwise. Without this the network is free to split an effect between the
    # two energies any way it likes, and they come out nearly the same number (measured: 0.82 Spearman).
    "func_penalty": 0.0,
    "pin_folding": True,  # activity assays do not move the folding energy (see the model's forward)
}


def tables(rows: pd.DataFrame) -> dict:
    """Integer codes for proteins, datasets and kinds, and each dataset's starting offset and scale.

    ``rows`` has one row per measurement: protein, dataset, kind (KIND_CODE keys), label, row (index into
    the embedding matrix), mut_aa and the feature columns. Labels of kcal/mol rows are used as they are;
    every other dataset gets an offset and scale that start at its 2nd and 98th percentiles, so the readout
    begins on the right scale and only has to be refined.
    """
    proteins = {p: i for i, p in enumerate(sorted(rows.protein.unique()))}
    datasets = {d: i for i, d in enumerate(sorted(rows.dataset.unique()))}
    low = rows.groupby("dataset").label.quantile(0.02)
    high = rows.groupby("dataset").label.quantile(0.98)
    return {
        "proteins": proteins,
        "datasets": datasets,
        "offset": [float(low[d]) for d in datasets],
        "scale": [float(max(high[d] - low[d], 1e-3)) for d in datasets],
    }


def weights(rows: pd.DataFrame, kcal_weight: float) -> pd.Series:
    """Each dataset weighs the same in the loss, whatever its size; the kcal/mol data get ``kcal_weight``.

    The label of each non-kcal dataset is also divided by its spread, so a dataset on a 0-1 scale and one on
    a 0-100 scale count alike.
    """
    per_dataset = 1 / rows.groupby("dataset").label.transform("size")
    label = rows.groupby("dataset").label
    high, low = label.transform(lambda y: y.quantile(0.98)), label.transform(lambda y: y.quantile(0.02))
    spread = (high - low).clip(1e-3)
    w = per_dataset / spread**2
    kcal = rows.kind == DDG
    w[~kcal] *= (1 - kcal_weight) / w[~kcal].sum() if (~kcal).any() else 0
    w[kcal] *= kcal_weight / w[kcal].sum() if kcal.any() else 0
    return w / w.mean()


def _tensors(rows, codes, embeddings, features, mean, std, dev):
    import numpy as np
    import torch

    aa = {a: i for i, a in enumerate(AMINO_ACIDS)}
    as_long = lambda x: torch.tensor(np.asarray(x), dtype=torch.long, device=dev)  # noqa: E731
    return {
        "vectors": torch.tensor(embeddings[rows.row.values].astype(np.float32), device=dev),
        "features": torch.tensor(((rows[features].values - mean) / std).astype(np.float32), device=dev),
        "alt": as_long(rows.mut_aa.map(aa).values),
        "protein": as_long(rows.protein.map(codes["proteins"]).values),
        "dataset": as_long(rows.dataset.map(codes["datasets"]).values),
        "kind": as_long(rows.kind.map(KIND_CODE).values),
        "label": torch.tensor(rows.label.values.astype(np.float32), device=dev),
    }


def fit(rows: pd.DataFrame, embeddings, features: list[str], settings: dict | None = None, seed: int = 0):
    """Train the two-energy model on every kind of measurement at once; returns (model, scaling, codes)."""
    import numpy as np
    import torch

    from qafi_core.stability import device

    settings = {**SETTINGS, **(settings or {})}
    dev = device()
    torch.manual_seed(seed)
    codes = tables(rows)
    values = rows[features].values.astype(np.float32)
    mean, std = values.mean(0), values.std(0) + 1e-6
    data = _tensors(rows, codes, embeddings, features, mean, std, dev)
    w = torch.tensor(weights(rows, settings["kcal_weight"]).values.astype(np.float32), device=dev)

    model = build(
        data["vectors"].shape[1],
        len(features),
        len(codes["proteins"]),
        len(codes["datasets"]),
        settings["hidden"],
        settings["dropout"],
    ).to(dev)
    with torch.no_grad():
        model.offset.copy_(torch.tensor(codes["offset"]))
        model.scale.copy_(torch.tensor(codes["scale"]))
    readout = [model.g_fold_wt, model.g_func_wt, model.offset, model.scale]
    network = [p for p in model.parameters() if all(p is not r for r in readout)]
    optimiser = torch.optim.AdamW(
        [
            {"params": network, "lr": settings["lr"], "weight_decay": settings["weight_decay"]},
            {"params": readout, "lr": settings["lr_readout"], "weight_decay": 0.0},
        ]
    )
    n = len(rows)
    for _ in range(settings["epochs"]):
        model.train()
        order = torch.randperm(n, device=dev)
        for start in range(0, n, settings["batch"]):
            i = order[start : start + settings["batch"]]
            optimiser.zero_grad()
            out = model(
                *(data[k][i] for k in ("vectors", "features", "alt", "protein", "dataset", "kind")),
                pin_folding=settings["pin_folding"],
            )
            loss = (w[i] * (out - data["label"][i]) ** 2).mean()
            if settings["func_penalty"]:
                _, ddg_func = model.energies(data["vectors"][i], data["features"][i], data["alt"][i])
                loss = loss + settings["func_penalty"] * ddg_func.abs().mean()
            loss.backward()
            optimiser.step()
    return model, (mean, std), codes


def energies(model, scaling, rows: pd.DataFrame, embeddings, features: list[str]):
    """Predicted ddg_fold and ddg_func (kcal/mol, positive = favourable) for any mutations, seen or not."""
    import numpy as np
    import torch

    from qafi_core.stability import device

    mean, std = scaling
    dev = device()
    aa = {a: i for i, a in enumerate(AMINO_ACIDS)}
    vectors = torch.tensor(embeddings[rows.row.values].astype(np.float32), device=dev)
    feats = torch.tensor(((rows[features].values - mean) / std).astype(np.float32), device=dev)
    alt = torch.tensor(rows.mut_aa.map(aa).values, dtype=torch.long, device=dev)
    model.eval()
    with torch.inference_mode():
        fold, func = model.energies(vectors, feats, alt)
    return fold.cpu().numpy(), func.cpu().numpy()


# --- putting the training data together -----------------------------------------------------------

# The 14a features, under the names the mega-scale table uses; 07f's feature table calls one of them
# esm_wt_logprob.
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


def stack(stability_rows, stability_vectors, label_rows, label_index, label_vectors, titles: dict):
    """One table of measurements and one matrix of residue vectors, from the two sources of 14c.

    ``stability_rows``: 14a's mega-scale mutations with features (domain, mut_aa, ddg, row, FEATURES); they
    become kcal/mol rows of one dataset. ``label_rows``: 07f's labelled variants (dataset_id, data_kind,
    uniprot_id, uniprot_position, alt_aa, label, features); their residue vectors are found through
    ``label_index`` (uniprot_id, position, row) in ``label_vectors``. ``titles`` maps dataset_id to its
    title, which decides each assay's readout. The label vectors are appended after the stability ones,
    so their rows are shifted by the size of the first matrix.
    """
    import numpy as np

    kcal = pd.DataFrame(
        {
            "protein": stability_rows.domain.values,
            "dataset": "tsuboyama_2023_kcal",
            "kind": DDG,
            "label": stability_rows.ddg.values,
            "row": stability_rows.row.values,
            "mut_aa": stability_rows.mut_aa.values,
            **{f: stability_rows[f].values for f in FEATURES},
        }
    )
    where = label_index.set_index(["uniprot_id", "position"]).row
    keys = pd.MultiIndex.from_arrays([label_rows.uniprot_id, label_rows.uniprot_position])
    rows = where.reindex(keys).values
    if np.isnan(rows.astype(float)).any():
        raise ValueError("some labelled variants have no residue vector")
    labelled = pd.DataFrame(
        {
            "protein": label_rows.uniprot_id.values,
            "dataset": label_rows.dataset_id.values,
            "kind": [
                readout_kind(k, titles.get(d))
                for k, d in zip(label_rows.data_kind, label_rows.dataset_id, strict=True)
            ],
            "label": label_rows.label.values,
            "row": rows.astype(int) + len(stability_vectors),
            "mut_aa": label_rows.alt_aa.values,
            **{f: label_rows["esm_wt_logprob" if f == "esm_wt" else f].values for f in FEATURES},
        }
    )
    return pd.concat([kcal, labelled], ignore_index=True), np.concatenate([stability_vectors, label_vectors])
