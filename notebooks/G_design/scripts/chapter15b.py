"""15b: does adding our two energies to the generator help, variant by variant?

ProteinMPNN can score a single substitution without being asked to design anything: its per-position
distribution gives log P(mutant) - log P(wild type). That is the generator used alone, and it is the
baseline the whole phase has to beat, because the generator saw these proteins' structures during its
own pretraining and ours did not see the proteins at all.

The combination is parameter-free -- each score is turned into ranks within an assay and averaged -- so
nothing is fitted on the test measurements.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from qafi_core import energy, stability

MODELS = Path("data/models")
WORK = Path("data/stability")
MPNN = Path("data/mpnn_probs")
AA = "ACDEFGHIKLMNPQRSTVWY"

# --- training data, decontaminated exactly as 14c has it -------------------------------------------
splits = stability.load_splits(WORK / "mega_splits.pkl")
mega = stability.add_features(
    stability.megascale(WORK / "Tsuboyama2023_Dataset2_Dataset3_20230416.csv", splits),
    pd.read_parquet(WORK / "structure.parquet"),
    pd.read_parquet(WORK / "residue_index.parquet"),
    np.load(WORK / "logprobs.npy"),
)
related = set(
    pd.read_parquet(WORK / "related_to_locked.parquet").query("domain_split in ['train', 'val']").domain
)
mega_train = mega[mega.split.isin(["train", "val"]) & ~mega.domain.isin(related)]
print(
    f"stability rows {len(mega_train):,} on {mega_train.domain.nunique()} domains "
    f"(dropped {len(related)} homologous to the locked test proteins)"
)

labels = pd.read_parquet(MODELS / "train_variants.parquet")
titles = pd.read_parquet(MODELS / "dataset_titles.parquet").set_index("dataset_id").title.to_dict()
rows, vectors = energy.stack(
    mega_train,
    np.load(WORK / "vectors.npy", mmap_mode="r"),
    labels,
    pd.read_parquet(MODELS / "train_embeddings.parquet"),
    np.asarray(np.load(MODELS / "train_embeddings.npy", mmap_mode="r")),
    titles,
)
print(f"training rows {len(rows):,}: " + ", ".join(f"{k} {v:,}" for k, v in rows.kind.value_counts().items()))

models = [energy.fit(rows, vectors, energy.FEATURES, seed=s) for s in (0, 1, 2)]
print(f"trained {len(models)} seeds")

# --- predict the two energies on the locked test variants ------------------------------------------
test = pd.read_parquet(MODELS / "test_variants.parquet")
index = pd.read_parquet(MODELS / "test_embeddings.parquet").set_index(["uniprot_id", "position"]).row
test["row"] = index.reindex(pd.MultiIndex.from_arrays([test.uniprot_id, test.uniprot_position])).values
test = test.dropna(subset=["row"]).copy()
test["row"] = test.row.astype(int)
test["mut_aa"] = test.alt_aa
test["esm_wt"] = test.esm_wt_logprob
emb = np.asarray(np.load(MODELS / "test_embeddings.npy", mmap_mode="r"))

fold, func = [], []
for m, scaling, _ in models:
    f, u = energy.energies(m, scaling, test, emb, energy.FEATURES)
    fold.append(f)
    func.append(u)
test["ours_fold"] = np.mean(fold, axis=0)
test["ours_func"] = np.mean(func, axis=0)
print(f"predicted two energies for {len(test):,} test variants")

# --- the generator used alone ----------------------------------------------------------------------
logp = np.load(MPNN / "mpnn_logp.npy")
mpnn_index = pd.read_parquet(MPNN / "mpnn_logp_index.parquet")
mpnn_index["row"] = np.arange(len(mpnn_index))
where = mpnn_index.set_index(["uniprot_id", "uniprot_position"]).row
test["mpnn_row"] = where.reindex(pd.MultiIndex.from_arrays([test.uniprot_id, test.uniprot_position])).values
have = test.dropna(subset=["mpnn_row"]).copy()
have["mpnn_row"] = have.mpnn_row.astype(int)
ai = {a: i for i, a in enumerate(AA)}
have["mpnn"] = logp[have.mpnn_row, have.alt_aa.map(ai)] - logp[have.mpnn_row, have.ref_aa.map(ai)]
print(f"generator scores for {len(have):,} variants on {have.uniprot_id.nunique()} proteins")

# --- the ablation ----------------------------------------------------------------------------------
have["ours_both"] = have.ours_fold + have.ours_func


def ranks(s):
    return stats.rankdata(s) / len(s)


blends = {}
for name, cols in {
    "生成器 + 我们两个能量": ["mpnn", "ours_both"],
    "生成器 + 我们功能能": ["mpnn", "ours_func"],
}.items():
    blends[name] = cols

SCORES = {
    "只用生成器 (ProteinMPNN)": lambda g: g.mpnn,
    "只用我们 两个能量": lambda g: g.ours_both,
    "只用我们 折叠能": lambda g: g.ours_fold,
    "只用我们 功能能": lambda g: g.ours_func,
    "ESM-2 (零样本)": lambda g: g.esm_delta,
    "生成器 + 我们两个能量": lambda g: (ranks(g.mpnn) + ranks(g.ours_both)) / 2,
    "生成器 + ESM-2": lambda g: (ranks(g.mpnn) + ranks(g.esm_delta)) / 2,
}

big = have.groupby("dataset_id").filter(lambda g: len(g) >= 200 and g.label.nunique() > 10)
print(
    f"\nassays scored: {big.dataset_id.nunique()} "
    f"({big.uniprot_id.nunique()} proteins, {len(big):,} variants)\n"
)

out, per_assay = [], {}
for name, fn in SCORES.items():
    rhos = [stats.spearmanr(g.label, fn(g)).statistic for _, g in big.groupby("dataset_id")]
    per_assay[name] = np.array(rhos)
    out.append({"打分方式": name, "中位 Spearman": round(float(np.median(rhos)), 3)})
print(pd.DataFrame(out).sort_values("中位 Spearman", ascending=False).to_string(index=False))

print("\n--- 消融：加上我们的能量，比只用生成器好吗（配对检验）---")
base = per_assay["只用生成器 (ProteinMPNN)"]
for name in ("生成器 + 我们两个能量", "只用我们 两个能量", "生成器 + ESM-2"):
    a = per_assay[name]
    w = stats.wilcoxon(a, base)
    print(
        f"  {name:<22} 中位 {np.median(a):.3f} vs {np.median(base):.3f}  "
        f"更好 {int((a > base).sum())}/{len(a)}  p={w.pvalue:.1e}"
    )

big.to_parquet(MODELS / "variant_ablation.parquet", index=False)
print(f"\nwrote {MODELS / 'variant_ablation.parquet'}")
