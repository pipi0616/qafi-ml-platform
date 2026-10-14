"""Scoring the proteome (chapter 09c): every possible missense variant of every human protein.

For each protein, in chunks of 50:
  1. ESM-2 650M reads the sequence and gives every residue its vector (as in 07e, with fixed shapes)
  2. the released stability network reads vector + ten residue features: 20 scores per residue
  3. the 16 variant features are computed for the 19 substitutions of every residue, and the released
     trees score them
  4. the members become percentiles with the training distributions: stability and function scores
  5. the chunk is saved as parquet; a protein listed in proteins/ is done, so a stopped run resumes

The residue features come from gold.dim_residue_proteome, built by the same SQL as the training features.
The variant features are computed here in Python, from those residue features and the two lookup tables;
``check_features`` compares them with the SQL table on the training variants (they must be equal), since
two implementations of one feature are how training-serving skew happens.

Shapes on the Apple GPU stay fixed (padded lengths in ``BUCKETS``, full batches), because its backend keeps
a compiled copy for every new shape (chapter 08b's incident).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qafi_core import release
from qafi_core.features import AMINO_ACIDS, FEATURES

BUCKETS = (128, 256, 384, 512, 768, 1024)  # padded window lengths in tokens; the window itself is <= 1022
TOKENS_PER_BATCH = 4096
HEAD_ROWS = 8192  # residues per pass of the stability network
OUTPUT = ["uniprot_id", "position", "ref_aa", "alt_aa", "zero_shot", "stability_score", "function_score"]


def _bucket(tokens: int) -> int:
    return next(b for b in BUCKETS if b >= tokens)


def _language_model():
    """07e's ESM-2 code (windows, loading), which lives with the Databricks code."""
    try:
        from lakehouse import language_model
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parents[2] / "databricks" / "src"))
        from lakehouse import language_model
    return language_model


def embed(sequences: list[str], model, tokenizer, tokens_per_batch: int = TOKENS_PER_BATCH) -> list:
    """Every residue's ESM-2 vector (length x 1280) for each sequence, read in fixed-shape batches.

    Long sequences are read in half-overlapping windows of 1,022 residues, each residue taken from the
    window where it is most central (``lakehouse.language_model``). Each window is padded to the next
    length in ``BUCKETS`` and batches are always full (the last one filled with repeats), so the GPU only
    ever sees a few shapes.
    """
    import torch

    lm = _language_model()
    windows, most_central = lm.windows, lm.most_central
    device = next(model.parameters()).device
    pieces = [(i, start, end) for i, s in enumerate(sequences) for start, end in windows(len(s))]
    read = {}
    by_bucket: dict[int, list] = {}
    for piece in pieces:
        by_bucket.setdefault(_bucket(piece[2] - piece[1] + 2), []).append(piece)
    for bucket, group in sorted(by_bucket.items()):
        size = max(1, tokens_per_batch // bucket)
        for first in range(0, len(group), size):
            batch = group[first : first + size]
            filled = batch + [batch[0]] * (size - len(batch))
            texts = [sequences[i][start:end] for i, start, end in filled]
            encoded = tokenizer(texts, return_tensors="pt", padding="max_length", max_length=bucket).to(
                device
            )
            with torch.inference_mode():
                hidden = model.esm(**encoded).last_hidden_state.float().cpu().numpy()
            for row, (i, start, end) in enumerate(batch):
                read[i, start] = hidden[row, 1 : end - start + 1]
    out = []
    for i, sequence in enumerate(sequences):
        spans = windows(len(sequence))
        choice = most_central(len(sequence), spans)
        vectors = np.empty((len(sequence), model.config.hidden_size), dtype=np.float32)
        for k, (start, _) in enumerate(spans):
            take = np.flatnonzero(choice == k)
            vectors[take] = read[i, start][take - start]
        out.append(vectors)
    return out


def stability(head, vectors: np.ndarray, residue_features: np.ndarray) -> np.ndarray:
    """The released network's 20 outputs for each residue (residues x 20), in fixed-size passes.

    A residue without structure values (a protein AlphaFold did not model) gets the training mean for
    them, which the network sees as 0 after scaling: it then relies on the ESM-2 vector alone.
    """
    import torch

    device = next(head.parameters()).device
    mean = head.mean.cpu().numpy()
    features = np.where(np.isnan(residue_features), mean, residue_features).astype(np.float32)
    out = np.empty((len(vectors), len(AMINO_ACIDS)), dtype=np.float32)
    head.eval()
    with torch.inference_mode():
        for first in range(0, len(vectors), HEAD_ROWS):
            n = min(HEAD_ROWS, len(vectors) - first)
            v = np.zeros((HEAD_ROWS, vectors.shape[1]), np.float32)
            f = np.zeros((HEAD_ROWS, features.shape[1]), np.float32)
            v[:n], f[:n] = vectors[first : first + n], features[first : first + n]
            scores = head(torch.from_numpy(v).to(device), torch.from_numpy(f).to(device))
            out[first : first + n] = scores[:n].cpu().numpy()
    return out


def variant_features(
    residues: pd.DataFrame, amino_acid: pd.DataFrame, blosum62: pd.DataFrame
) -> pd.DataFrame:
    """The 19 substitutions of every standard residue, with the 16 features of gold.fct_variant_features.

    ``residues`` has the columns of gold.dim_residue (position, aa, logprobs and the residue features).
    The formulas are those of fct_variant_features.sql; ``check_features`` holds them to it.
    """
    residues = residues[residues["aa"].isin(list(AMINO_ACIDS))].reset_index(drop=True)
    n = len(residues)
    ref = residues["aa"].map(AMINO_ACIDS.index).to_numpy()
    logprobs = np.stack(residues["logprobs"].to_numpy()).astype(np.float32)  # n x 20
    alt = np.tile(np.arange(20), n).reshape(n, 20)
    keep = alt != ref[:, None]
    row = np.repeat(np.arange(n), 19)
    alt = alt[keep]
    props = amino_acid.set_index("aa").loc[list(AMINO_ACIDS)]
    hydro, volume, charge = (props[c].to_numpy(np.float32) for c in ("hydrophobicity", "volume", "charge"))
    order = list(AMINO_ACIDS)
    blosum = (
        blosum62.pivot(index="ref_aa", columns="alt_aa", values="score")
        .loc[order, order]
        .to_numpy(np.float32)
    )
    r = ref[row]
    wild = logprobs[np.arange(n), ref][row]
    rsa = residues["rsa"].to_numpy(np.float32)[row]
    out = pd.DataFrame(
        {
            "uniprot_position": residues["position"].to_numpy()[row],
            "ref_aa": np.array(list(AMINO_ACIDS))[r],
            "alt_aa": np.array(list(AMINO_ACIDS))[alt],
            "blosum62": blosum[r, alt],
            "d_hydrophobicity": hydro[alt] - hydro[r],
            "d_volume": volume[alt] - volume[r],
            "d_charge": charge[alt] - charge[r],
            "esm_delta": logprobs[row, alt] - wild,
            "esm_wt_logprob": wild,
        }
    )
    for name in release.RESIDUE_FEATURES:
        if name != "esm_wt_logprob":
            out[name] = residues[name].to_numpy(np.float32)[row]
    out["d_volume_buried"] = out["d_volume"] * (1 - np.minimum(rsa, 1))
    out["residue"] = row  # which residue of ``residues`` each variant belongs to
    return out


def check_features(residues: pd.DataFrame, variants: pd.DataFrame, amino_acid, blosum62) -> pd.Series:
    """Largest difference per feature between ``variant_features`` and the SQL feature table (``variants``,
    rows of gold.fct_variant_features). All must be zero up to rounding: one feature, one value."""
    ours = []
    for protein, r in residues.groupby("uniprot_id"):
        ours.append(variant_features(r, amino_acid, blosum62).assign(uniprot_id=protein))
    ours = pd.concat(ours, ignore_index=True)
    key = ["uniprot_id", "uniprot_position", "ref_aa", "alt_aa"]
    both = variants.drop_duplicates(key).merge(ours, on=key, suffixes=("_sql", ""))
    if len(both) != variants.drop_duplicates(key).shape[0]:
        raise ValueError("some labelled variants have no computed features")
    return pd.Series(
        {f: float(np.nanmax(np.abs(both[f] - both[f"{f}_sql"].astype(np.float32)))) for f in FEATURES}
    )


def score_proteins(
    sequences: dict, residues: pd.DataFrame, models: dict, amino_acid, blosum62
) -> pd.DataFrame:
    """The released scores of every missense variant of ``sequences`` (uniprot_id -> sequence).

    ``models`` holds esm and tokenizer (ESM-2 650M), head and trees (the released models) and tables (their
    percentile tables).
    """
    model, tokenizer, head, trees = models["esm"], models["tokenizer"], models["head"], models["trees"]
    ids = list(sequences)
    vectors = embed([sequences[i] for i in ids], model, tokenizer)
    parts = []
    for protein, protein_vectors in zip(ids, vectors, strict=True):
        r = residues[residues["uniprot_id"] == protein].sort_values("position").reset_index(drop=True)
        r = r[r["position"] <= len(protein_vectors)]
        standard = r["aa"].isin(list(AMINO_ACIDS)).to_numpy()
        r = r[standard].reset_index(drop=True)
        v = protein_vectors[r["position"].to_numpy() - 1]
        net = stability(head, v, r[release.RESIDUE_FEATURES].to_numpy(np.float32))
        variants = variant_features(r, amino_acid, blosum62)
        alt = variants["alt_aa"].map(AMINO_ACIDS.index).to_numpy()
        variants["stability"] = net[variants["residue"].to_numpy(), alt]
        variants["trees"] = trees.predict(variants[FEATURES].astype(np.float32))
        scored = release.scores(variants, models["tables"])
        parts.append(
            scored.assign(uniprot_id=protein).rename(
                columns={"uniprot_position": "position", "esm_delta": "zero_shot"}
            )[OUTPUT]
        )
    out = pd.concat(parts, ignore_index=True)
    return out.astype({c: np.float32 for c in ("zero_shot", "stability_score", "function_score")})


def done(folder: Path) -> set[str]:
    """Proteins already scored in ``folder``."""
    return {
        p
        for part in Path(folder).glob("proteins/part-*.parquet")
        for p in pd.read_parquet(part)["uniprot_id"]
    }


def run(
    sequences: pd.DataFrame,
    residues_path: Path,
    folder: Path,
    models: dict,
    amino_acid,
    blosum62,
    chunk_size: int = 50,
    keep_going=lambda: True,
    log=print,
) -> int:
    """Score ``sequences`` (uniprot_id, sequence) chunk by chunk into ``folder``; returns proteins scored.

    Each chunk writes ``scores/part-N.parquet`` and then ``proteins/part-N.parquet``; a chunk whose protein
    file is missing was cut off, and its scores are deleted and redone. ``keep_going`` is asked before
    every chunk (memory pressure), so the run can stop and resume later.
    """
    import pyarrow.dataset as ds

    folder = Path(folder)
    for kind in ("scores", "proteins"):
        (folder / kind).mkdir(parents=True, exist_ok=True)
    finished = {p.name for p in (folder / "proteins").glob("part-*.parquet")}
    for orphan in (folder / "scores").glob("part-*.parquet"):
        if orphan.name not in finished:
            orphan.unlink()
    todo = sequences[~sequences["uniprot_id"].isin(done(folder))].sort_values("uniprot_id")
    part = max((int(n[5:10]) for n in finished), default=-1) + 1
    data = ds.dataset(residues_path)
    started, residues_done, total = time.time(), 0, int(todo["sequence"].str.len().sum())
    for first in range(0, len(todo), chunk_size):
        if not keep_going():
            return first
        chunk = todo.iloc[first : first + chunk_size]
        residues = data.to_table(filter=ds.field("uniprot_id").isin(chunk["uniprot_id"].tolist())).to_pandas()
        sequences_of = dict(zip(chunk["uniprot_id"], chunk["sequence"], strict=True))
        scores = score_proteins(sequences_of, residues, models, amino_acid, blosum62)
        name = f"part-{part:05d}.parquet"
        scores.to_parquet(folder / "scores" / name, index=False)
        counts = chunk["uniprot_id"].map(scores["uniprot_id"].value_counts()).fillna(0).astype(int)
        chunk[["uniprot_id"]].assign(variants=counts).to_parquet(folder / "proteins" / name, index=False)
        part, residues_done = part + 1, residues_done + int(chunk["sequence"].str.len().sum())
        rate = residues_done / (time.time() - started)
        hours = (total - residues_done) / rate / 3600
        log(f"{first + len(chunk):,} of {len(todo):,} proteins, {rate:.0f} residues/s, {hours:.1f} h left")
    return len(todo)
