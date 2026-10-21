"""The mega-scale domains homologous to this project's locked test proteins (14a's decontamination).

14a computes this inline and does not keep it. It is the same answer every time, so it is saved here
for 15b and anything later that needs the clean training set.
"""

from pathlib import Path

import pandas as pd

from qafi_core import stability, validation

WORK = Path("data/stability")
splits = stability.load_splits(WORK / "mega_splits.pkl")
muts = stability.megascale(WORK / "Tsuboyama2023_Dataset2_Dataset3_20230416.csv", splits)

split_table = pd.read_csv("dbt/seeds/protein_split.csv")
sequences = pd.read_parquet("data/models/proteome_sequences.parquet").set_index("uniprot_id").sequence
locked = {
    u: sequences[u]
    for u in split_table.loc[split_table.split == "test", "uniprot_id"]
    if u in sequences.index
}
domains = muts.groupby(["domain", "split"]).seq.first().reset_index()
print(f"{len(domains)} domains x {len(locked)} locked test proteins")

hits = pd.DataFrame(
    [
        {"domain": d.domain, "domain_split": d.split, "test_protein": u}
        for d in domains.itertuples()
        for u, s in locked.items()
        if validation.related(d.seq, s)
    ]
)
hits.to_parquet(WORK / "related_to_locked.parquet", index=False)
print(f"homologous domains: {hits.domain.nunique()} / {len(domains)}")
print(hits.groupby("domain_split").domain.nunique().to_string())
print(f"wrote {WORK / 'related_to_locked.parquet'}")
