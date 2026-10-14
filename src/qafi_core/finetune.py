"""Fine-tuning ESM-2 with LoRA (chapter 08c): our labels change the language model itself.

In chapters 08a and 08b ESM-2 was frozen: its embeddings were computed once and a small network learned
to read them. Here a small ESM-2 (35 million parameters, so it fits a laptop) is trained together with
its head. LoRA keeps that cheap: the original weights stay as they are, and next to the attention and
feed-forward matrices sits a pair of thin matrices whose product is added to them. Only those are
trained, about 2% of the parameters.

The model reads crops, not whole proteins: 254 residues around the labelled stretch of one dataset, with
ESM's start and end tokens. All batches have the same shape, 8 crops of 256 tokens, and the labels sit in
an 8 x 256 x 20 grid with weight 0 where nothing was measured. Fixed shapes keep the Apple GPU from
caching a compiled copy for every new size (chapter 08b's incident).

Three ways to use the same small model, compared on the exam of chapter 07a:
  zero_shot   its own log-probabilities, no training
  frozen      its embeddings computed once, a head trained on them (chapter 08a's method)
  lora        the head and the LoRA matrices trained together (fine-tuning)

  crops            the crops of every dataset, with their labels on the grid
  cross_validate   every variant predicted by a model that saw neither its protein nor its relatives
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from qafi_core import baseline, validation
from qafi_core.features import AMINO_ACIDS
from qafi_core.neural import standardised

MODEL = "facebook/esm2_t12_35M_UR50D"
REVISION = "6fbf070e65b0b7291e7bbcd451118c216cff79d8"  # pinned Hugging Face commit
LENGTH = 256  # tokens per crop: a start token, 254 residues, an end token
CROPS_PER_BATCH = 8
SETTINGS = {
    "epochs": 4,
    "head_learning_rate": 1e-3,
    "lora_learning_rate": 3e-4,
    "lora_rank": 8,  # width of the thin matrices
    "lora_alpha": 16,  # how strongly their product is added
    "weight_decay": 0.01,
    "dropout": 0.1,
    "seed": 0,
}
RUNS = {"esm2_35m_zero_shot": "zero_shot", "esm2_35m_frozen": "frozen", "esm2_35m_lora": "lora"}
EPOCHS = {"frozen": 4, "lora": 5}  # chosen with learning_curve on fold 0


def parameters(how: str) -> dict:
    """What a run of ``RUNS`` should record about itself: the model, how it is used, every setting."""
    recorded = {
        "model": MODEL,
        "revision": REVISION,
        "use": how,
        "crop_length": LENGTH,
        "device": str(device()),
    }
    if how == "zero_shot":
        return recorded
    recorded |= {**SETTINGS, "epochs": EPOCHS[how], "crops_per_batch": CROPS_PER_BATCH}
    if how == "frozen":  # nothing of ESM-2 is trained, so the LoRA settings do not apply
        for name in ("lora_learning_rate", "lora_rank", "lora_alpha"):
            recorded.pop(name)
    return recorded


def device() -> torch.device:
    """An NVIDIA GPU, the Apple GPU, or the CPU. Crops have one fixed shape, so the Apple GPU is safe here."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load(model: str = MODEL, revision: str = REVISION):
    """The tokenizer and ESM-2 with its language-model head (needed for the zero-shot scores)."""
    from transformers import AutoTokenizer, EsmForMaskedLM

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    return tokenizer, EsmForMaskedLM.from_pretrained(model, revision=revision)


def windows(first: int, last: int, length: int, residues: int = LENGTH - 2) -> list[int]:
    """Start positions (1-based) of crops that cover residues first..last of a protein ``length`` long.

    A stretch that fits is centred in one crop, with as much context on both sides as the protein has.
    A longer one gets crops overlapping by half; each labelled residue later goes to the crop where it is
    most central.
    """
    width = last - first + 1
    if width <= residues:
        start = (first + last) // 2 - residues // 2
        return [max(1, min(start, length - residues + 1))]
    stride = residues // 2
    starts = list(range(max(1, first - stride // 2), last - residues + stride, stride))
    return [max(1, min(s, length - residues + 1)) for s in starts]


def crops(df: pd.DataFrame, sequences: pd.Series, tokenizer) -> list[dict]:
    """The crops of every dataset in ``df`` (fresh index), each with its tokens and its labels on the grid.

    A crop holds one dataset only, so a residue and amino acid have at most one label in it. Per crop:
    ``tokens``, ``mask`` (1 = a real token), ``target`` and ``weight`` (256 x 20: the standardised label
    and the dataset weight of each measured variant, 0 elsewhere), ``ref`` (the wild-type amino acid of
    each token, for the zero-shot score), ``protein``, and ``cells``: for each variant its row in ``df``,
    token and amino acid, to read predictions back.
    """
    target, weight = standardised(df).to_numpy(np.float32), baseline.weights(df).astype(np.float32)
    alt = df["alt_aa"].map(AMINO_ACIDS.index).to_numpy()
    out = []
    for dataset, rows in df.groupby("dataset_id", sort=True).indices.items():
        protein = df["uniprot_id"].iloc[rows[0]]
        sequence = sequences[protein]
        positions = df["uniprot_position"].to_numpy()[rows]
        starts = np.array(windows(positions.min(), positions.max(), len(sequence)))
        centre = starts + (LENGTH - 2) / 2
        best = np.abs(positions[:, None] - centre[None, :]).argmin(
            axis=1
        )  # the crop where it is most central
        for k, start in enumerate(starts):
            mine = rows[best == k]
            if not len(mine):
                continue
            piece = sequence[start - 1 : start - 1 + LENGTH - 2]
            encoded = tokenizer(piece, padding="max_length", max_length=LENGTH, return_tensors="pt")
            token = df["uniprot_position"].to_numpy()[mine] - start + 1  # token 0 is the start token
            grid_target = np.zeros((LENGTH, len(AMINO_ACIDS)), np.float32)
            grid_weight = np.zeros((LENGTH, len(AMINO_ACIDS)), np.float32)
            grid_target[token, alt[mine]] = target[mine]
            grid_weight[token, alt[mine]] = weight[mine]
            ref = np.full(LENGTH, -1)
            ref[1 : len(piece) + 1] = [AMINO_ACIDS.find(a) for a in piece]
            out.append(
                {
                    "dataset": dataset,
                    "protein": protein,
                    "tokens": encoded["input_ids"][0],
                    "mask": encoded["attention_mask"][0],
                    "target": torch.from_numpy(grid_target),
                    "weight": torch.from_numpy(grid_weight),
                    "ref": torch.from_numpy(ref),
                    "cells": np.stack([mine, token, alt[mine]], axis=1),
                }
            )
    return out


class Model(nn.Module):
    """ESM-2, frozen or with LoRA, and a head giving every token a score for each of the 20 amino acids."""

    def __init__(self, esm, hidden: int, dropout: float):
        super().__init__()
        self.esm = esm
        width = esm.config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 20),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.esm(input_ids=tokens, attention_mask=mask).last_hidden_state)


def _batches(items: list[dict], shuffle: bool, generator=None):
    """Batches of exactly ``CROPS_PER_BATCH`` crops; the last one is filled up with repeats of weight 0."""
    order = torch.randperm(len(items), generator=generator).tolist() if shuffle else list(range(len(items)))
    for i in range(0, len(order), CROPS_PER_BATCH):
        chosen = order[i : i + CROPS_PER_BATCH]
        filler = CROPS_PER_BATCH - len(chosen)
        group = [items[j] for j in chosen] + [items[chosen[0]]] * filler
        keys = [key for key in ("tokens", "mask", "target", "weight", "ref", "hidden") if key in group[0]]
        batch = {key: torch.stack([c[key] for c in group]) for key in keys}
        if filler:
            batch["weight"][len(chosen) :] = 0
        yield [items[j] for j in chosen], batch


def _score_grid(scores: torch.Tensor, real: list[dict], predicted: np.ndarray) -> None:
    """Copy each measured variant's score from the grid into ``predicted``."""
    grid = scores.detach().float().cpu().numpy()
    for b, crop in enumerate(real):
        rows, token, alt = crop["cells"].T
        predicted[rows] = grid[b, token, alt]


def zero_shot(items: list[dict], lm, dev) -> np.ndarray:
    """ESM-2's own score for every variant, log p(new amino acid) - log p(old one): 07e's score."""
    lm = lm.to(dev).eval()
    predicted = np.full(max(int(c["cells"][:, 0].max()) for c in items) + 1, np.nan, np.float32)
    amino = torch.tensor(_amino_token_ids(lm), device=dev)
    with torch.no_grad():
        for real, batch in _batches(items, shuffle=False):
            logits = lm(input_ids=batch["tokens"].to(dev), attention_mask=batch["mask"].to(dev)).logits
            logprobs = torch.log_softmax(logits[..., amino], dim=-1)
            ref = batch["ref"].to(dev).clamp(min=0)
            wild = logprobs.gather(2, ref[..., None])
            _score_grid(logprobs - wild, real, predicted)
    return predicted


def _amino_token_ids(lm) -> list[int]:
    """Token ids of the 20 amino acids in our order (ESM-2's vocabulary is the same for every size)."""
    vocab = ["<cls>", "<pad>", "<eos>", "<unk>", *"LAGVSERTIDPKQNFYMHWCXBUZO", ".", "-", "<null_1>", "<mask>"]
    return [vocab.index(a) for a in AMINO_ACIDS]


def _train(model: Model, items: list[dict], setup: dict, dev, lora: bool, each_epoch=None) -> list[float]:
    """The four steps of chapter 08a, on batches of crops; returns the mean loss of each epoch."""
    groups = [{"params": model.head.parameters(), "lr": setup["head_learning_rate"]}]
    if lora:
        trainable = [p for p in model.esm.parameters() if p.requires_grad]
        groups.append({"params": trainable, "lr": setup["lora_learning_rate"]})
    optimiser = torch.optim.AdamW(groups, weight_decay=setup["weight_decay"])
    generator = torch.Generator().manual_seed(setup["seed"])
    history = []
    for epoch in range(setup["epochs"]):
        model.train()
        if not lora:
            model.esm.eval()  # frozen: no dropout inside ESM either
        total, steps = 0.0, 0
        for _, batch in _batches(items, shuffle=True, generator=generator):
            batch = {key: value.to(dev) for key, value in batch.items()}
            if lora:
                scores = model(batch["tokens"], batch["mask"])
            else:  # frozen: ESM-2's output was computed once (``_embed``); only the head learns
                scores = model.head(batch["hidden"].float())
            weight = batch["weight"]
            loss = (weight * (scores - batch["target"]) ** 2).sum() / weight.sum().clamp(min=1e-6)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item()
            steps += 1
        history.append(total / steps)
        if each_epoch:
            each_epoch(epoch + 1, history[-1])
    return history


def _predict(model: Model, items: list[dict], dev, rows_total: int) -> np.ndarray:
    model.eval()
    predicted = np.full(rows_total, np.nan, np.float32)
    with torch.no_grad():
        for real, batch in _batches(items, shuffle=False):
            if "hidden" in batch:
                scores = model.head(batch["hidden"].to(dev).float())
            else:
                scores = model(batch["tokens"].to(dev), batch["mask"].to(dev))
            _score_grid(scores, real, predicted)
    return predicted


def _embed(items: list[dict], esm, dev) -> list[dict]:
    """The crops with ESM-2's output attached, computed once: a frozen model always gives the same."""
    esm = esm.to(dev).eval()
    out = []
    with torch.no_grad():
        for real, batch in _batches(items, shuffle=False):
            tokens, mask = batch["tokens"].to(dev), batch["mask"].to(dev)
            hidden = esm(input_ids=tokens, attention_mask=mask).last_hidden_state
            out += [{**crop, "hidden": h.half().cpu()} for crop, h in zip(real, hidden, strict=False)]
    return out


def _fresh(lm, setup: dict, lora: bool, dev) -> Model:
    """A new model around a copy of ESM-2: with LoRA matrices to train, or with every weight frozen."""
    import copy

    from peft import LoraConfig, get_peft_model

    esm = copy.deepcopy(lm.esm)
    if lora:
        config = LoraConfig(
            r=setup["lora_rank"],
            lora_alpha=setup["lora_alpha"],
            lora_dropout=0.05,
            target_modules=["query", "key", "value", "dense"],
        )
        esm = get_peft_model(esm, config)
    else:
        for parameter in esm.parameters():
            parameter.requires_grad = False
    return Model(esm, hidden=256, dropout=setup["dropout"]).to(dev)


def cross_validate(
    df: pd.DataFrame, items: list[dict], lm, how: str, settings: dict | None = None, n_folds: int = 5
) -> pd.DataFrame:
    """``df`` with ``fold`` and ``predicted`` added, like ``baseline.cross_validate``; same folds too.

    ``how`` is zero_shot, frozen or lora. Each fold starts again from the original ESM-2.
    """
    dev = device()
    setup = {**SETTINGS, "epochs": EPOCHS.get(how, SETTINGS["epochs"]), **(settings or {})}
    df = df.reset_index(drop=True)
    fold = validation.group_folds(df["similarity_group"], n_folds)
    if how == "zero_shot":
        return df.assign(fold=fold, predicted=zero_shot(items, lm, dev)[: len(df)])
    fold_of = fold.groupby(df["uniprot_id"]).first()
    if how == "frozen":
        items = _embed(items, lm.esm, dev)
    predicted = np.full(len(df), np.nan, np.float32)
    for k in range(n_folds):
        torch.manual_seed(setup["seed"] + k)
        inside = [c for c in items if fold_of[c["protein"]] != k]
        held_out = [c for c in items if fold_of[c["protein"]] == k]
        model = _fresh(lm, setup, how == "lora", dev)
        _train(model, inside, setup, dev, how == "lora")
        scores = _predict(model, held_out, dev, len(df))
        predicted[~np.isnan(scores)] = scores[~np.isnan(scores)]
        del model
        if dev.type == "mps":
            torch.mps.empty_cache()
    return df.assign(fold=fold, predicted=predicted)


def learning_curve(
    df: pd.DataFrame, items: list[dict], lm, how: str, settings: dict | None = None, fold: int = 0
) -> pd.DataFrame:
    """Loss on the training proteins and score on held-out proteins after each epoch (one fold held out)."""
    dev = device()
    setup = {**SETTINGS, **(settings or {})}
    df = df.reset_index(drop=True)
    fold_of = validation.group_folds(df["similarity_group"], 5).groupby(df["uniprot_id"]).first()
    inside = [c for c in items if fold_of[c["protein"]] != fold]
    held_out = [c for c in items if fold_of[c["protein"]] == fold]
    if how == "frozen":
        inside, held_out = _embed(inside, lm.esm, dev), _embed(held_out, lm.esm, dev)
    torch.manual_seed(setup["seed"])
    model = _fresh(lm, setup, how == "lora", dev)
    rows = []

    def record(epoch: int, loss: float) -> None:
        scores = _predict(model, held_out, dev, len(df))
        watched = df.assign(predicted=scores).dropna(subset=["predicted"])
        rows.append(
            {
                "epoch": epoch,
                "training_loss": loss,
                "held_out_spearman": baseline.dataset_scores(watched)["spearman"].median(),
            }
        )

    _train(model, inside, setup, dev, how == "lora", each_epoch=record)
    return pd.DataFrame(rows).set_index("epoch").round(4)
