"""ESM-2 protein language model -> per-residue amino-acid log-probabilities and embeddings (notebook 07e).

ESM-2 (Lin et al. 2023) learned to fill in hidden amino acids from 65 million protein sequences. Reading a
protein once, it gives at every position the probability of each of the 20 amino acids ("wild-type
marginals", Meier et al. 2021). A substitution's zero-shot score is log p(mutant) - log p(wild type) at its
position: below 0, the model finds the mutant less likely than the wild type, which goes with damaging
effects. The model's last layer also gives each residue a vector (an embedding) that notebook 08 learns from.

ESM-2 reads at most 1,022 residues at a time, so longer proteins are read in overlapping windows; each residue
takes its values from the window in which it lies furthest from the ends, as notebook 07d does with AlphaFold
fragments.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "facebook/esm2_t33_650M_UR50D"  # 650 million parameters, the size most used for variant effects
REVISION = "08e4846e537177426273712802403f7ba8261b6c"  # pinned Hugging Face commit, so results can be redone
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # the order of the 20 log-probabilities
WINDOW = 1022  # residues per read: ESM-2 was trained on 1,024 tokens, a start and an end token included


def windows(length: int, size: int = WINDOW) -> list[tuple[int, int]]:
    """Start and end (0-based, end excluded) of the windows that cover a sequence, each overlapping the next
    by at least half."""
    if length <= size:
        return [(0, length)]
    starts = [*range(0, length - size, size // 2), length - size]
    return [(start, start + size) for start in starts]


def most_central(length: int, spans: list[tuple[int, int]]) -> np.ndarray:
    """For each residue, the window in which it lies furthest from the window's ends (earlier on a tie)."""
    positions = np.arange(length)
    best, choice = np.full(length, -1), np.zeros(length, dtype=int)
    for k, (start, end) in enumerate(spans):
        from_end = np.where(
            (positions >= start) & (positions < end), np.minimum(positions - start, end - 1 - positions), -1
        )
        better = from_end > best
        best[better], choice[better] = from_end[better], k
    return choice


def load(path: str = MODEL, revision: str | None = REVISION):
    """ESM-2 and its tokenizer from a folder or the Hugging Face Hub, on the fastest device there is: an
    NVIDIA GPU, a Mac's GPU, or else the CPU.

    Always in full (32-bit) precision. Half precision was only 12% faster on a Mac's GPU but moved
    log-probabilities by up to 0.2; in full precision the Mac's GPU and the CPU agree to 0.0001, so the
    features come out the same whichever machine computes them.
    """
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(path, revision=revision)
    model = EsmForMaskedLM.from_pretrained(path, revision=revision, dtype=torch.float32).to(device).eval()
    return model, tokenizer


def read_proteins(
    sequences: list[str],
    model,
    tokenizer,
    embeddings: bool = False,
    tokens_per_batch: int = 16384,
    window: int = WINDOW,
) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """For each sequence: the log-probabilities of the 20 amino acids at every position (length x 20) and,
    if asked, the last layer's vector of every residue (length x model width).

    The windows of all sequences are read longest first, in batches of about ``tokens_per_batch`` tokens, so
    little padding is computed.
    """
    import torch

    amino_acids = tokenizer.convert_tokens_to_ids(list(AMINO_ACIDS))
    device = next(model.parameters()).device
    pieces = sorted(
        ((i, start, end) for i, s in enumerate(sequences) for start, end in windows(len(s), window)),
        key=lambda piece: piece[2] - piece[1],
        reverse=True,
    )
    read = {}  # (sequence, window start) -> (log-probabilities, vectors)
    while pieces:
        width = pieces[0][2] - pieces[0][1] + 2  # the longest piece left, with its start and end tokens
        size = max(1, tokens_per_batch // width)
        batch, pieces = pieces[:size], pieces[size:]
        texts = [sequences[i][start:end] for i, start, end in batch]
        encoded = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            hidden = model.esm(**encoded).last_hidden_state
            logits = model.lm_head(hidden)[..., amino_acids].float()
            logprobs = torch.log_softmax(logits, dim=-1).cpu().numpy()
            vectors = hidden.float().cpu().numpy() if embeddings else None
        for row, (i, start, end) in enumerate(batch):
            residues = slice(1, end - start + 1)  # token 0 is the start token
            read[i, start] = logprobs[row, residues], vectors[row, residues] if embeddings else None

    results = []
    for i, sequence in enumerate(sequences):
        spans = windows(len(sequence), window)
        choice = most_central(len(sequence), spans)
        logprobs = np.empty((len(sequence), len(AMINO_ACIDS)), dtype=np.float32)
        width = model.config.hidden_size
        vectors = np.empty((len(sequence), width), dtype=np.float32) if embeddings else None
        for k, (start, _) in enumerate(spans):
            take = np.flatnonzero(choice == k)
            logprobs[take] = read[i, start][0][take - start]
            if embeddings:
                vectors[take] = read[i, start][1][take - start]
        results.append((logprobs, vectors))
    return results


def residue_frame(uniprot_ids, sequences, values, column: str = "logprobs") -> pd.DataFrame:
    """One row per residue: protein, position (from 1), amino acid and its values (a list of numbers)."""
    lengths = [len(s) for s in sequences]
    return pd.DataFrame(
        {
            "uniprot_id": np.repeat(np.asarray(uniprot_ids), lengths),
            "position": np.concatenate([np.arange(1, n + 1) for n in lengths]).astype("int32"),
            "aa": list("".join(sequences)),
            column: list(np.concatenate(values)),
        }
    )


def protein_frame(uniprot_ids, sequences, version: str, embedded) -> pd.DataFrame:
    """One row per protein read: which sequence (its MD5, as Spark's md5() writes it) and model version."""
    return pd.DataFrame(
        {
            "uniprot_id": list(uniprot_ids),
            "sequence_md5": [hashlib.md5(s.encode()).hexdigest() for s in sequences],
            "model_version": version,
            "residues": np.array([len(s) for s in sequences], dtype="int32"),
            "windows": np.array([len(windows(len(s))) for s in sequences], dtype="int32"),
            "embedded": list(embedded),
        }
    )


def free_memory() -> None:
    """Give back the GPU memory PyTorch keeps for reuse: on a laptop, other programs share it."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def table_ddl(catalog: str) -> list[str]:
    """The three silver tables, created by whichever task writes to them first."""
    return [
        f"""CREATE TABLE IF NOT EXISTS {catalog}.silver.residue_language_model
            (uniprot_id string, position int, aa string, logprobs array<float>)
            CLUSTER BY (uniprot_id)
            COMMENT 'ESM-2 log-probabilities of {AMINO_ACIDS} at every residue (notebook 07e)'""",
        f"""CREATE TABLE IF NOT EXISTS {catalog}.silver.residue_embedding
            (uniprot_id string, position int, embedding array<float>)
            CLUSTER BY (uniprot_id)
            COMMENT 'ESM-2 last-layer vector of every residue of the labelled proteins (notebook 07e)'""",
        f"""CREATE TABLE IF NOT EXISTS {catalog}.silver.language_model_proteins
            (uniprot_id string, sequence_md5 string, model_version string, residues int, windows int,
             embedded boolean, read_at timestamp)
            COMMENT 'Proteins read by ESM-2: which sequence and model version (notebook 07e)'""",
    ]


def proteins_read(folder: str | Path) -> set[str]:
    """The proteins already saved in ``folder`` by read_to_parquet (none if it is empty or missing)."""
    parts = Path(folder).glob("proteins/part-*.parquet")
    return {uniprot_id for part in parts for uniprot_id in pd.read_parquet(part)["uniprot_id"]}


def read_to_parquet(
    proteins: pd.DataFrame,
    model,
    tokenizer,
    folder: str | Path,
    version: str = REVISION,
    embeddings: bool = False,
    chunk_size: int = 200,
    tokens_per_batch: int = 4096,
    rest: float = 0.0,
    keep_going=lambda: True,
    log=print,
) -> int:
    """Read ``proteins`` (columns uniprot_id, sequence) chunk by chunk, saving each chunk as parquet files
    with the columns of the silver tables: ``residues/`` (log-probabilities), ``embeddings/`` (every
    residue's vector, if ``embeddings``) and ``proteins/``. Returns how many proteins were read.

    A chunk's protein file is written last, so a protein listed there is done: a rerun skips it and goes on
    where the last run stopped, after deleting the files of a chunk that was cut off half-way. After each
    chunk the GPU memory is given back, and with ``rest`` the run pauses for that fraction of the chunk's
    time, leaving the GPU to other programs. ``keep_going`` is asked before each chunk; when it says no, the
    run stops, to be resumed later.
    """
    folder = Path(folder)
    kinds = ["residues", "embeddings", "proteins"] if embeddings else ["residues", "proteins"]
    for kind in kinds:
        (folder / kind).mkdir(parents=True, exist_ok=True)
    finished = sorted(p.name for p in (folder / "proteins").glob("part-*.parquet"))
    for kind in kinds[:-1]:
        for orphan in (folder / kind).glob("part-*.parquet"):
            if orphan.name not in finished:
                orphan.unlink()
    todo = proteins[~proteins["uniprot_id"].isin(proteins_read(folder))]
    part = max((int(name[5:10]) for name in finished), default=-1) + 1
    total, residues, started = sum(len(s) for s in todo["sequence"]), 0, time.time()
    for first in range(0, len(todo), chunk_size):
        if not keep_going():
            return first
        chunk_started = time.time()
        chunk = todo.iloc[first : first + chunk_size]
        ids, sequences = chunk["uniprot_id"].tolist(), chunk["sequence"].tolist()
        results = read_proteins(sequences, model, tokenizer, embeddings, tokens_per_batch)
        name = f"part-{part:05d}.parquet"
        residue_frame(ids, sequences, [r[0] for r in results]).to_parquet(folder / "residues" / name)
        if embeddings:
            vectors = residue_frame(ids, sequences, [r[1] for r in results], "embedding")
            vectors.drop(columns="aa").to_parquet(folder / "embeddings" / name)
        protein_frame(ids, sequences, version, [embeddings] * len(ids)).assign(
            read_at=pd.Timestamp.now(tz="UTC")
        ).to_parquet(folder / "proteins" / name)
        part, residues = part + 1, residues + sum(len(s) for s in sequences)
        free_memory()
        time.sleep(rest * (time.time() - chunk_started))
        rate = residues / (time.time() - started)
        log(
            f"{first + len(chunk):,} of {len(todo):,} proteins, {rate:.0f} residues/s, "
            f"about {(total - residues) / rate / 3600:.1f} h left"
        )
    return len(todo)


if __name__ == "__main__":  # a backfill on a Mac's GPU, gentle with the other programs
    import argparse
    import gc
    import subprocess

    from lakehouse.uniprot import read_fasta

    parser = argparse.ArgumentParser(description="Read proteins with ESM-2 here, saving parquet files.")
    parser.add_argument("fasta", help="UniProt reference proteome FASTA")
    parser.add_argument("folder", help="where the parquet files go")
    parser.add_argument("--only", help="a file of UniProt accessions, one per line: read just these proteins")
    parser.add_argument("--embeddings", action="store_true", help="also save every residue's vector")
    args = parser.parse_args()

    def calm(highest: int) -> bool:
        """Memory pressure at most ``highest`` (macOS: 1 normal, 2 warning, 4 critical). Asked twice, 30 s
        apart, because loading the model causes a short peak of its own."""
        for attempt in range(2):
            command = ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]
            if int(subprocess.run(command, capture_output=True).stdout or 1) <= highest:
                return True
            if attempt == 0:
                time.sleep(30)
        return False

    def log(message: str) -> None:
        print(f"{time.strftime('%H:%M')} {message}", flush=True)

    proteome = pd.DataFrame(read_fasta(args.fasta)).rename(columns={"accession": "uniprot_id"})
    if args.only:
        proteome = proteome[proteome["uniprot_id"].isin(Path(args.only).read_text().split())]
    proteome = proteome.sort_values("uniprot_id")
    folder = Path(args.folder)
    while True:  # other programs come first: when memory gets critical, unload the model and wait
        while not calm(1):  # start only when memory pressure is normal
            time.sleep(300)
        model, tokenizer = load()
        read_to_parquet(  # small batches, GPU memory given back, a rest after each chunk
            proteome, model, tokenizer, folder, embeddings=args.embeddings, chunk_size=50,
            tokens_per_batch=2048, rest=0.5, keep_going=lambda: calm(2), log=log,  # stop if it gets critical
        )  # fmt: skip
        del model
        gc.collect()
        free_memory()
        left = len(set(proteome["uniprot_id"]) - proteins_read(folder))
        if left == 0:
            log("all proteins read")
            break
        log(f"memory is critical: model unloaded, {left:,} proteins left; next try in 5 minutes or later")
        time.sleep(300)
