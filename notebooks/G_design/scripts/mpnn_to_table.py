"""Turn ProteinMPNN's per-position log-probabilities into a table keyed by UniProt position.

AlphaFold models run 1:1 with the UniProt sequence, so residue i of the file is UniProt position i+1.
That is checked here against the project's own sequences rather than assumed: if the amino acid the
model saw differs from the one UniProt has, the protein is dropped and reported.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
OUT = Path("data/mpnn_probs")

seqs = pd.read_parquet("data/models/proteome_sequences.parquet").set_index("uniprot_id").sequence

rows, logps, bad = [], [], []
for f in sorted((OUT / "unconditional_probs_only").glob("*.npz")):
    acc = f.name.split("-")[1]
    d = np.load(f)
    log_p = d["log_p"][0]  # (L, 21)
    seen = "".join(ALPHABET[i] for i in d["S"])
    truth = seqs.get(acc)
    if truth is None or len(truth) != len(seen):
        bad.append((acc, f"length {len(seen)} vs {0 if truth is None else len(truth)}"))
        continue
    same = sum(a == b for a, b in zip(seen, truth, strict=True)) / len(truth)
    if same < 0.99:
        bad.append((acc, f"sequence agreement {same:.2%}"))
        continue

    p = np.exp(log_p[:, :20])
    p = p / p.sum(1, keepdims=True)
    entropy = -(p * np.log(p + 1e-12)).sum(1)
    rows.append(
        pd.DataFrame(
            {
                "uniprot_id": acc,
                "uniprot_position": np.arange(1, len(seen) + 1),
                "mpnn_entropy": entropy,
                "mpnn_wt_logp": log_p[np.arange(len(seen)), d["S"]],
            }
        )
    )
    logps.append(log_p[:, :20].astype(np.float32))

table = pd.concat(rows, ignore_index=True)
table.to_parquet(OUT / "mpnn_positions.parquet", index=False)
np.save(OUT / "mpnn_logp.npy", np.concatenate(logps))
index = table[["uniprot_id", "uniprot_position"]]
index.to_parquet(OUT / "mpnn_logp_index.parquet", index=False)

print(f"kept {table.uniprot_id.nunique()} proteins, {len(table):,} positions")
print(f"dropped {len(bad)}:")
for acc, why in bad[:10]:
    print(f"  {acc}  {why}")
print(
    f"\nentropy: min {table.mpnn_entropy.min():.2f}  median {table.mpnn_entropy.median():.2f} "
    f"max {table.mpnn_entropy.max():.2f}  (ln 20 = {np.log(20):.2f} is uniform)"
)
