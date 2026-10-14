"""A graph neural network on AlphaFold residue graphs (chapter 08b).

A protein is a graph: every residue is a node, and two residues are joined by an edge when they lie within
10 angstrom of each other in the AlphaFold model (chapter 07d). A graph layer lets every residue update its
own description with the average of its neighbours' (message passing); after two layers a residue has
heard from its neighbours' neighbours.

``Network`` is the network of chapter 08a with such layers in the middle, and ``layers=0`` gives it back
without them, so the only difference between two runs is whether residues look at their neighbours. It
reads one of two inputs:
  - ``esm``        the ESM-2 embedding of each residue, as in chapter 08a
  - ``structure``  only the amino acid and six structure features: no language model anywhere

The exam and the scores are those of ``baseline`` and ``neural``.

  protein_graphs   one graph per protein, with its labelled variants attached
  Network          the network
  cross_validate   every variant predicted by a network that saw neither its protein nor its relatives
  learning_curve   loss and held-out score after each epoch
  probe            how much of a structure feature a straight-line read-out of the embedding recovers
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv

from qafi_core import baseline, validation
from qafi_core.features import AMINO_ACIDS, FEATURE_GROUPS, FEATURES
from qafi_core.neural import EMBEDDING_SIZE, standardised

NODE_FEATURES = FEATURE_GROUPS["structure"]  # what the structure-only network knows about a residue
SUBSTITUTION_FEATURES = [*FEATURE_GROUPS["substitution"], "d_volume_buried"]  # ... and about a substitution
INPUTS = {
    # per kind of input: the variant features joined on at the end, and how the network is trained
    "esm": {"features": FEATURES, "hidden": 256, "dropout": 0.2, "learning_rate": 3e-4, "epochs": 10},
    "structure": {
        "features": SUBSTITUTION_FEATURES,
        "hidden": 128,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "epochs": 30,
    },
}
SETTINGS = {"weight_decay": 0.01, "proteins_per_batch": 8, "seed": 0}


def device() -> torch.device:
    """Where graph networks are trained: an NVIDIA GPU if there is one, otherwise the CPU. Not the Apple GPU.

    Every batch of proteins has another size. PyTorch's Apple-GPU backend keeps a compiled copy of the
    computation for each new input size and never lets go of it, so memory grew to 30 GB within minutes
    when this was tried. These networks are small and the graphs trimmed: on the CPU an epoch takes about
    a second, as fast as on the GPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# The runs of chapter 08b: with and without neighbours, for each kind of input.
RUNS = {
    "gnn_esm_0_layers": {"inputs": "esm", "layers": 0},
    "gnn_esm_2_layers": {"inputs": "esm", "layers": 2},
    "gnn_structure_0_layers": {"inputs": "structure", "layers": 0},
    "gnn_structure_2_layers": {"inputs": "structure", "layers": 2},
}


def parameters(config: dict) -> dict:
    """What a run of ``RUNS`` should record about itself."""
    setup = INPUTS[config["inputs"]]
    return {
        "model": "graph network" if config["layers"] else "network without graph layers",
        "inputs": config["inputs"],
        "graph_layers": config["layers"],
        "variant_features": ",".join(setup["features"]),
        **{name: value for name, value in setup.items() if name != "features"},
        **SETTINGS,
        "standardise_labels": True,
        "device": str(device()),
    }


class Protein(Data):
    """One protein's graph together with its labelled variants.

    When PyTorch Geometric joins several proteins into one batch it renumbers the nodes; ``variant_node``
    points at nodes, so it has to be shifted by the same amount as the edges are.
    """

    def __inc__(self, key, value, *args, **kwargs):
        return self.num_nodes if key == "variant_node" else super().__inc__(key, value, *args, **kwargs)


def protein_graphs(
    df: pd.DataFrame, residues: pd.DataFrame, table, index: pd.DataFrame, steps: int = 2
) -> dict[str, Protein]:
    """One graph per protein of ``df``; ``df`` must have a fresh index (0, 1, 2, ...).

    ``residues`` has one row per residue: uniprot_id, position, aa, the six structure features, and
    ``neighbours``, the positions within 10 angstrom. ``table`` and ``index`` are the embedding matrix and
    its residue index (``neural.embedding_table``). Each graph carries, per node, the embedding, the amino
    acid and the structure features; and per labelled variant its node, its new amino acid, its features,
    its standardised label, its weight and its row in ``df``.

    Only residues within ``steps`` edges of a labelled residue are kept. A network with that many graph
    layers cannot hear from residues further away, so its predictions are exactly the same without them;
    most assays cover one domain of a long protein, and this leaves a sixth of the residues.
    """
    row_of = index.set_index(["uniprot_id", "position"])["row"]
    embeddings = torch.from_numpy(np.array(table))
    target, weight = standardised(df).to_numpy(np.float32), baseline.weights(df).astype(np.float32)
    variants_of = df.groupby("uniprot_id", sort=False).indices
    graphs = {}
    for protein, r in residues[residues["uniprot_id"].isin(variants_of)].groupby("uniprot_id", sort=False):
        r = r.sort_values("position")
        positions = r["position"].to_numpy()
        node_of = pd.Series(np.arange(len(positions)), index=positions)
        neighbours = [n if n is not None else [] for n in r["neighbours"]]
        receiver = np.repeat(np.arange(len(positions)), [len(n) for n in neighbours])
        sender = node_of.reindex(np.concatenate(neighbours) if len(receiver) else []).to_numpy()
        receiver, sender = receiver[~np.isnan(sender)], sender[~np.isnan(sender)].astype(np.int64)
        rows = variants_of[protein]
        v = df.iloc[rows]
        labelled = node_of.loc[v["uniprot_position"]].to_numpy(np.int64)

        keep = np.zeros(len(positions), dtype=bool)
        keep[labelled] = True
        for _ in range(steps):
            keep[sender[keep[receiver]]] = True  # whoever sends a message to a kept residue is kept too
        new_number = np.cumsum(keep) - 1
        edges = keep[sender] & keep[receiver]
        embedding_rows = row_of.loc[protein].reindex(positions[keep]).to_numpy(np.int64)
        graphs[protein] = Protein(
            num_nodes=int(keep.sum()),
            embedding=embeddings[torch.tensor(embedding_rows)],
            aa=torch.tensor(r["aa"].map(AMINO_ACIDS.index).to_numpy(np.int64)[keep]),
            structure=torch.tensor(r[NODE_FEATURES].to_numpy(np.float32)[keep]),
            edge_index=torch.tensor(np.stack([new_number[sender[edges]], new_number[receiver[edges]]])),
            variant_node=torch.tensor(new_number[labelled]),
            variant_alt=torch.tensor(v["alt_aa"].map(AMINO_ACIDS.index).to_numpy(np.int64)),
            variant_features=torch.tensor(v[FEATURES].to_numpy(np.float32)),
            variant_target=torch.tensor(target[rows]),
            variant_weight=torch.tensor(weight[rows]),
            variant_row=torch.tensor(rows.astype(np.int64)),
        )
    return graphs


class Network(nn.Module):
    """Describe every residue, let residues exchange descriptions along the edges, then score variants.

    ``encode`` turns a residue's input into ``hidden`` numbers. Each graph layer (GraphSAGE) adds to a
    residue's numbers a function of its own and of its neighbours' average. ``head`` reads the residue of
    a variant, joined with the variant's features, and gives a score for each of the 20 amino acids.
    The constants that scale the features are stored with the network.
    """

    def __init__(
        self, inputs: str, layers: int, scaling: dict[str, torch.Tensor], hidden: int, dropout: float
    ):
        super().__init__()
        self.inputs = inputs
        self.columns = [FEATURES.index(name) for name in INPUTS[inputs]["features"]]
        for name, value in scaling.items():
            self.register_buffer(name, value)
        if inputs == "esm":
            self.norm = nn.LayerNorm(EMBEDDING_SIZE)
            width = EMBEDDING_SIZE
        else:
            width = len(AMINO_ACIDS) + len(NODE_FEATURES)
        self.encode = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout))
        self.convs = nn.ModuleList([SAGEConv(hidden, hidden) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden + len(self.columns), 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, len(AMINO_ACIDS)),
        )

    def forward(self, batch) -> torch.Tensor:
        if self.inputs == "esm":
            nodes = self.norm(batch.embedding.float())
        else:
            scaled = (batch.structure - self.structure_mean) / self.structure_std
            nodes = torch.cat([nn.functional.one_hot(batch.aa, len(AMINO_ACIDS)).float(), scaled], dim=1)
        h = self.encode(nodes)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            h = norm(h + self.dropout(nn.functional.gelu(conv(h, batch.edge_index))))
        features = (batch.variant_features[:, self.columns] - self.features_mean) / self.features_std
        scores = self.head(torch.cat([h[batch.variant_node], features], dim=1))
        return scores.gather(1, batch.variant_alt[:, None]).squeeze(1)


def _new_network(graphs: list[Protein], inputs: str, layers: int, setup: dict, dev) -> Network:
    """An untrained network; its scaling constants come from the training proteins only."""
    columns = [FEATURES.index(name) for name in setup["features"]]
    features = torch.cat([g.variant_features for g in graphs])[:, columns]
    structure = torch.cat([g.structure for g in graphs])
    scaling = {
        "features_mean": features.mean(0),
        "features_std": features.std(0) + 1e-6,
        "structure_mean": structure.mean(0),
        "structure_std": structure.std(0) + 1e-6,
    }
    return Network(inputs, layers, scaling, setup["hidden"], setup["dropout"]).to(dev)


def train(model: Network, graphs: list[Protein], setup: dict, dev, each_epoch=None) -> list[float]:
    """Train on batches of whole proteins; returns the mean loss of each epoch.

    The four steps are those of ``neural.train``. What differs is the batch: a handful of proteins, joined
    by ``DataLoader`` into one big graph without edges between them.
    """
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=setup["learning_rate"], weight_decay=setup["weight_decay"]
    )
    loader = DataLoader(graphs, batch_size=setup["proteins_per_batch"], shuffle=True)
    history = []
    for epoch in range(setup["epochs"]):
        model.train()
        total = 0.0
        for batch in loader:
            batch = batch.to(dev)
            error = model(batch) - batch.variant_target
            loss = (batch.variant_weight * error**2).sum() / batch.variant_weight.sum()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item()
        history.append(total / len(loader))
        if each_epoch:
            each_epoch(epoch + 1, history[-1])
    return history


def predict(model: Network, graphs: list[Protein], dev) -> tuple[np.ndarray, np.ndarray]:
    """The trained network's scores for the variants of ``graphs``, and the rows of ``df`` they belong to."""
    model.eval()
    scores, rows = [], []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=16):
            scores.append(model(batch.to(dev)).cpu().numpy())
            rows.append(batch.variant_row.cpu().numpy())
    return np.concatenate(scores), np.concatenate(rows)


def _folds(df: pd.DataFrame, graphs: dict[str, Protein], n_folds: int):
    df = df.reset_index(drop=True)
    if sum(len(g.variant_row) for g in graphs.values()) != len(df):
        raise ValueError("the graphs were built from another table of variants")
    fold = validation.group_folds(df["similarity_group"], n_folds)
    return df, fold, fold.groupby(df["uniprot_id"]).first()


def cross_validate(
    df: pd.DataFrame,
    graphs: dict[str, Protein],
    inputs: str = "esm",
    layers: int = 2,
    settings: dict | None = None,
    n_folds: int = 5,
) -> pd.DataFrame:
    """``df`` with ``fold`` and ``predicted`` added, like ``baseline.cross_validate``; same folds too."""
    dev = device()
    setup = {**INPUTS[inputs], **SETTINGS, **(settings or {})}
    df, fold, fold_of = _folds(df, graphs, n_folds)
    predicted = np.full(len(df), np.nan, dtype=np.float32)
    for k in range(n_folds):
        torch.manual_seed(setup["seed"] + k)
        inside = [graphs[protein] for protein in fold_of.index[fold_of != k]]
        held_out = [graphs[protein] for protein in fold_of.index[fold_of == k]]
        model = _new_network(inside, inputs, layers, setup, dev)
        train(model, inside, setup, dev)
        scores, rows = predict(model, held_out, dev)
        predicted[rows] = scores
    return df.assign(fold=fold, predicted=predicted)


def learning_curve(
    df, graphs, inputs: str = "esm", layers: int = 2, settings: dict | None = None, fold: int = 0
) -> pd.DataFrame:
    """Loss on the training proteins and score on held-out proteins after each epoch (one fold held out)."""
    dev = device()
    setup = {**INPUTS[inputs], **SETTINGS, **(settings or {})}
    df, folds, fold_of = _folds(df, graphs, 5)
    inside = [graphs[protein] for protein in fold_of.index[fold_of != fold]]
    held_out = [graphs[protein] for protein in fold_of.index[fold_of == fold]]
    torch.manual_seed(setup["seed"])
    model = _new_network(inside, inputs, layers, setup, dev)
    history = []

    def record(epoch: int, loss: float) -> None:
        scores, rows = predict(model, held_out, dev)
        watched = baseline.dataset_scores(df.iloc[rows].assign(predicted=scores))
        history.append(
            {"epoch": epoch, "training_loss": loss, "held_out_spearman": watched["spearman"].median()}
        )

    train(model, inside, setup, dev, each_epoch=record)
    return pd.DataFrame(history).set_index("epoch").round(4)


def probe(
    table, index: pd.DataFrame, residues: pd.DataFrame, column: str, groups: pd.Series, sample: int = 80_000
) -> float:
    """How much of ``column`` a straight-line read-out of the embedding recovers, on held-out proteins.

    A linear probe: for each fold, weights are fitted on the other proteins (ridge regression) and the
    read-out is compared with the true values of the held-out ones. The result is the correlation over all
    residues, 0 to 1. A high value means the embedding already contains that information, even though
    ESM-2 never saw a structure. ``groups`` gives each protein's group of relatives (index uniprot_id);
    ``sample`` residues are enough for 1,280 weights and keep it quick.
    """
    from sklearn.model_selection import GroupKFold

    data = index.merge(residues[["uniprot_id", "position", column]], on=["uniprot_id", "position"])
    data = data[data["uniprot_id"].isin(groups.index)].dropna(subset=[column])
    data = data.sample(min(sample, len(data)), random_state=0).sort_values("row")
    x = torch.from_numpy(np.array(table[data["row"].to_numpy()])).double()
    y = torch.tensor(data[column].to_numpy(np.float64))
    read_out = torch.zeros_like(y)
    splits = GroupKFold(5).split(data, groups=groups.loc[data["uniprot_id"]].to_numpy())
    for inside, held_out in splits:
        mean, centre = x[inside].mean(0), y[inside].mean()
        a = x[inside] - mean
        ridge = 0.01 * len(inside) * torch.eye(x.shape[1], dtype=torch.float64)  # keeps the weights small
        weights = torch.linalg.solve(a.T @ a + ridge, a.T @ (y[inside] - centre))
        read_out[held_out] = (x[held_out] - mean) @ weights + centre
    return round(float(np.corrcoef(read_out.numpy(), y.numpy())[0, 1]), 3)
