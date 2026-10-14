"""A small neural network on ESM-2 residue embeddings (chapter 08a): the first PyTorch model.

ESM-2 describes every residue by 1,280 numbers (chapter 07e). ``Head`` reads that description, optionally
with the 16 features of chapter 07f, and gives 20 numbers: its prediction for each amino acid the residue
could change into. ESM-2 itself is not trained here; only the head is.

The exam is the one of chapter 07a, and the scores are computed by the functions of ``baseline``, so
this model and the tree baseline are compared on identical terms.

  embedding_table   the embeddings of the wanted proteins as one matrix on disk, and where each residue is
  Head              the network
  train, predict    the training loop and its counterpart
  cross_validate    every variant predicted by a network that saw neither its protein nor its relatives
  fit               the network trained on all rows
  learning_curve    loss and held-out score after each epoch: shows when more training starts to hurt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from qafi_core import baseline, validation
from qafi_core.features import AMINO_ACIDS, FEATURES

EMBEDDING_SIZE = 1280
SETTINGS = {
    "hidden": 512,  # units of the first layer; the second has a quarter
    "dropout": 0.2,
    "epochs": 4,  # passes over the training variants; more makes held-out proteins worse (learning_curve)
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "batch_size": 4096,
    "seed": 0,
}
# The runs of chapter 08a: each adds one thing to the one before.
RUNS = {
    "mlp_embedding": {},
    "mlp_embedding_features": {"features": FEATURES},
    "mlp_standardised": {"features": FEATURES, "standardise": True},
}


def parameters(config: dict) -> dict:
    """What a run of ``RUNS`` should record about itself: its inputs and every training setting."""
    return {
        "model": "mlp on ESM-2 embeddings",
        "inputs": "embedding + 16 features" if config.get("features") else "embedding",
        "standardise_labels": config.get("standardise", False),
        **SETTINGS,
        "device": str(device()),
    }


def device() -> torch.device:
    """The fastest processor PyTorch can use here: an NVIDIA GPU, an Apple GPU, or the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def embedding_table(parts: Path, proteins, path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """The embeddings of ``proteins`` as one matrix (residues x 1280) at ``path``, and each residue's row.

    ``parts`` is a parquet file or folder with uniprot_id, position and embedding. The matrix is written
    once, as 16-bit numbers (half the size; the fifth decimal is not needed), and afterwards only opened.
    Returns the matrix, read from disk on demand, and a table uniprot_id, position, row.
    """
    import pyarrow.dataset as ds

    path = Path(path)
    index_path = path.with_suffix(".parquet")
    if not (path.exists() and index_path.exists()):
        data = ds.dataset(parts).filter(ds.field("uniprot_id").isin(sorted(proteins)))
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float16, shape=(data.count_rows(), EMBEDDING_SIZE)
        )
        index, done = [], 0
        for batch in data.to_batches(columns=["uniprot_id", "position", "embedding"], batch_size=20_000):
            values = batch.column("embedding").flatten().to_numpy(zero_copy_only=False)
            matrix[done : done + len(batch)] = values.reshape(len(batch), EMBEDDING_SIZE)
            index.append(batch.select(["uniprot_id", "position"]).to_pandas())
            done += len(batch)
        matrix.flush()
        pd.concat(index, ignore_index=True).assign(row=np.arange(done)).to_parquet(index_path)
    return np.load(path, mmap_mode="r"), pd.read_parquet(index_path)


def standardised(df: pd.DataFrame) -> pd.Series:
    """Each label minus the mean of its dataset, divided by the dataset's spread.

    Assays differ in how far apart their scores lie. With raw labels the network must also guess each
    assay's scale; standardised, it only has to get the order within an assay right, which is what the
    score measures.
    """
    groups = df.groupby("dataset_id")["label"]
    return (df["label"] - groups.transform("mean")) / (groups.transform("std") + 1e-6)


class Head(nn.Module):
    """From a residue's embedding (and features) to a score for each of the 20 amino acids.

    Two layers with a non-linearity between them; dropout switches off random units during training, so
    the network cannot rely on any single one. The mean and spread used to scale the features are stored
    with the network: a saved model scales new inputs exactly as it did in training.
    """

    def __init__(self, mean=None, std=None, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.register_buffer("mean", mean if mean is not None else torch.zeros(0))
        self.register_buffer("std", std if std is not None else torch.ones(0))
        self.norm = nn.LayerNorm(EMBEDDING_SIZE)
        self.body = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE + len(self.mean), hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden // 4, len(AMINO_ACIDS))

    def forward(self, embedding: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm(embedding.float())
        if len(self.mean):
            x = torch.cat([x, (features - self.mean) / self.std], dim=1)
        return self.out(self.body(x))


def _tensors(df: pd.DataFrame, index: pd.DataFrame, features, target: pd.Series, dev) -> dict:
    """The variants as tensors on the device: where each one's embedding is, what it changes to, and so on."""
    rows = df[["uniprot_id", "uniprot_position"]].merge(
        index, left_on=["uniprot_id", "uniprot_position"], right_on=["uniprot_id", "position"], how="left"
    )["row"]
    if rows.isna().any():
        raise ValueError(f"{int(rows.isna().sum())} variants have no embedding")
    data = {
        "row": torch.tensor(rows.to_numpy(dtype=np.int64), device=dev),
        "alt": torch.tensor(df["alt_aa"].map(AMINO_ACIDS.index).to_numpy(dtype=np.int64), device=dev),
        "target": torch.tensor(target.to_numpy(dtype=np.float32), device=dev),
        "weight": torch.tensor(baseline.weights(df).astype(np.float32), device=dev),
        "features": None,
    }
    if features:
        data["features"] = torch.tensor(df[list(features)].to_numpy(dtype=np.float32), device=dev)
    return data


def _new_head(data: dict, rows: torch.Tensor, settings: dict, dev) -> Head:
    """An untrained network; its feature scaling comes from the training rows only."""
    mean = std = None
    if data["features"] is not None:
        mean, std = data["features"][rows].mean(0), data["features"][rows].std(0) + 1e-6
    return Head(mean, std, settings["hidden"], settings["dropout"]).to(dev)


def _outputs(model: Head, table: torch.Tensor, data: dict, rows: torch.Tensor) -> torch.Tensor:
    """The network's score for each of these variants: of its 20 outputs, the one of the new amino acid."""
    features = data["features"][rows] if data["features"] is not None else None
    scores = model(table[data["row"][rows]], features)
    return scores.gather(1, data["alt"][rows, None]).squeeze(1)


def train(model: Head, table: torch.Tensor, data: dict, rows: torch.Tensor, settings: dict, each_epoch=None):
    """Train ``model`` on the variants ``rows``; returns the mean loss of each epoch.

    One step: predict a batch, measure the weighted squared error, work out how every weight should change
    to shrink it (``backward``), change them a little (``step``). An epoch is one pass over all variants,
    in a new random order. ``each_epoch(epoch, loss)`` is called after each one.
    """
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    history = []
    for epoch in range(settings["epochs"]):
        model.train()
        order = rows[torch.randperm(len(rows), device=rows.device)]
        total, batches = 0.0, order.split(settings["batch_size"])
        for batch in batches:
            error = _outputs(model, table, data, batch) - data["target"][batch]
            weight = data["weight"][batch]
            loss = (weight * error**2).sum() / weight.sum()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item()
        history.append(total / len(batches))
        if each_epoch:
            each_epoch(epoch + 1, history[-1])
    return history


def predict(model: Head, table: torch.Tensor, data: dict, rows: torch.Tensor) -> np.ndarray:
    """The trained network's scores for the variants ``rows``, with dropout off and no gradients kept."""
    model.eval()
    with torch.no_grad():
        return torch.cat([_outputs(model, table, data, batch) for batch in rows.split(16384)]).cpu().numpy()


def _prepare(df, table, index, features, standardise, settings):
    dev = device()
    settings = {**SETTINGS, **(settings or {})}
    df = df.reset_index(drop=True)
    target = standardised(df) if standardise else df["label"]
    if not isinstance(table, torch.Tensor):
        table = torch.tensor(np.array(table), device=dev)  # read from disk into (GPU) memory, still 16-bit
    return df, table, _tensors(df, index, features, target, dev), settings, dev


def cross_validate(
    df: pd.DataFrame,
    table,
    index: pd.DataFrame,
    features: list[str] | None = None,
    standardise: bool = False,
    settings: dict | None = None,
    n_folds: int = 5,
) -> pd.DataFrame:
    """``df`` with ``fold`` and ``predicted`` added, like ``baseline.cross_validate``.

    The folds come from the same function as the baseline's, so both models are tested on the same splits.
    """
    df, table, data, settings, dev = _prepare(df, table, index, features, standardise, settings)
    fold = validation.group_folds(df["similarity_group"], n_folds)
    predicted = np.full(len(df), np.nan, dtype=np.float32)
    for k in range(n_folds):
        torch.manual_seed(settings["seed"] + k)
        inside = torch.tensor(np.flatnonzero(fold != k), device=dev)
        held_out = torch.tensor(np.flatnonzero(fold == k), device=dev)
        model = _new_head(data, inside, settings, dev)
        train(model, table, data, inside, settings)
        predicted[fold == k] = predict(model, table, data, held_out)
    return df.assign(fold=fold, predicted=predicted)


def fit(df, table, index, features=None, standardise: bool = False, settings: dict | None = None) -> Head:
    """The network trained on every row of ``df``: the one to save and to use on new proteins."""
    df, table, data, settings, dev = _prepare(df, table, index, features, standardise, settings)
    torch.manual_seed(settings["seed"])
    everything = torch.arange(len(df), device=dev)
    model = _new_head(data, everything, settings, dev)
    train(model, table, data, everything, settings)
    return model.cpu()


def score(model: Head, df: pd.DataFrame, table, index: pd.DataFrame, features=None) -> np.ndarray:
    """A trained network's prediction for every variant of ``df``, whose residues must be in ``table``."""
    dev = device()
    rows = df[["uniprot_id", "uniprot_position"]].merge(
        index, left_on=["uniprot_id", "uniprot_position"], right_on=["uniprot_id", "position"], how="left"
    )["row"]
    if rows.isna().any():
        raise ValueError(f"{int(rows.isna().sum())} variants have no embedding")
    table = torch.tensor(np.array(table), device=dev)
    row = torch.tensor(rows.to_numpy(np.int64), device=dev)
    alt = torch.tensor(df["alt_aa"].map(AMINO_ACIDS.index).to_numpy(np.int64), device=dev)
    values = torch.tensor(df[list(features)].to_numpy(np.float32), device=dev) if features else None
    model = model.to(dev).eval()
    out = []
    with torch.no_grad():
        for batch in torch.arange(len(df), device=dev).split(16384):
            scores = model(table[row[batch]], values[batch] if values is not None else None)
            out.append(scores.gather(1, alt[batch, None]).squeeze(1).cpu())
    return torch.cat(out).numpy()


def learning_curve(
    df, table, index, features=None, standardise: bool = False, settings: dict | None = None, fold: int = 0
) -> pd.DataFrame:
    """Loss on the training proteins and score on held-out proteins after each epoch.

    Trains on four folds and watches the fifth. The loss keeps falling, but the held-out score peaks early:
    from then on the network is learning its training proteins by heart. This is how ``epochs`` was chosen.
    """
    df, table, data, settings, dev = _prepare(df, table, index, features, standardise, settings)
    folds = validation.group_folds(df["similarity_group"], 5)
    inside = torch.tensor(np.flatnonzero(folds != fold), device=dev)
    held_out = torch.tensor(np.flatnonzero(folds == fold), device=dev)
    watched = df[folds == fold]
    torch.manual_seed(settings["seed"])
    model = _new_head(data, inside, settings, dev)
    rows = []

    def record(epoch: int, loss: float) -> None:
        scores = baseline.dataset_scores(watched.assign(predicted=predict(model, table, data, held_out)))
        rows.append({"epoch": epoch, "training_loss": loss, "held_out_spearman": scores["spearman"].median()})

    train(model, table, data, inside, settings, each_epoch=record)
    return pd.DataFrame(rows).set_index("epoch").round(4)
