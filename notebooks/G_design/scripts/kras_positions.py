"""15c, the functional axis: on KRAS, which ranking finds the positions binding needs?

The main benchmark is dominated by abundance assays, which see folding, so a folding term wins there
almost by construction. KRAS is the one held-out protein where binding is measured as well -- six
partners, thousands of substitutions -- so it is where a functional ranking can actually be tested.

Two questions, in order of sharpness:
  1. which ranking orders positions by measured binding damage?
  2. which ranking finds the positions that lose binding while folding is fine -- the ones a design
     must hold fixed even though nothing about the structure says so?
"""

import pandas as pd
from scipy import stats

BINDING = [
    "binding_RAF1",
    "binding_PIK3CG",
    "binding_RALGDS",
    "binding_SOS1",
    "binding_DARPinK27",
    "binding_DARPinK55",
]

k = pd.read_parquet("data/kras/kras_predictions.parquet")
mp = pd.read_parquet("data/mpnn_probs/mpnn_positions.parquet")
mp = mp[mp.uniprot_id == "P01116"].rename(columns={"uniprot_position": "position"})
print(f"KRAS substitutions {len(k):,} | MPNN positions {len(mp):,}")

k["binding"] = k[BINDING].mean(axis=1)
pos = (
    k.groupby("position")
    .agg(
        meas_fold=("folding", "mean"),
        meas_bind=("binding", "mean"),
        n=("folding", "size"),
        ours_fold=("predicted_fold", "mean"),
        ours_func=("predicted_func", "mean"),
        rsa=("rsa", "first"),
        esm_delta=("esm_delta", "mean"),
        esm_entropy=("esm_entropy", "first"),
    )
    .reset_index()
    .merge(mp[["position", "mpnn_entropy"]], on="position", how="left")
)
pos = pos.dropna(subset=["meas_fold", "meas_bind", "mpnn_entropy"])
print(f"positions with everything: {len(pos)}\n")

# every ranking oriented so HIGH = "important, freeze it"
RANKINGS = {
    "我们 功能能": -pos.ours_func,
    "我们 折叠能": -pos.ours_fold,
    "生成器 (MPNN 熵)": -pos.mpnn_entropy,
    "保守性 (ESM 熵)": -pos.esm_entropy,
    "ESM-2 替换代价": -pos.esm_delta,
    "埋藏程度 (RSA)": -pos.rsa,
}

print("=" * 72)
print("1  排序 vs 实测损伤（Spearman；KRAS 的实测是 ddG，高=损伤大，所以直接取正；越高越好）")
print("=" * 72)
rows = []
for name, r in RANKINGS.items():
    rows.append(
        {
            "排序方法": name,
            "对实测折叠": round(stats.spearmanr(pos.meas_fold, r).statistic, 3),
            "对实测结合": round(stats.spearmanr(pos.meas_bind, r).statistic, 3),
        }
    )
t = pd.DataFrame(rows)
t["结合 − 折叠"] = (t["对实测结合"] - t["对实测折叠"]).round(3)
print(t.sort_values("对实测结合", ascending=False).to_string(index=False))

print()
print("=" * 72)
print("2  真正的功能位置：结合坏了但折叠没坏")
print("=" * 72)
# residual binding damage after regressing out folding damage: what folding cannot explain
slope, intercept, *_ = stats.linregress(pos.meas_fold, pos.meas_bind)
pos["binding_beyond_folding"] = pos.meas_bind - (intercept + slope * pos.meas_fold)
cut = pos.binding_beyond_folding.quantile(0.80)
pos["is_functional"] = pos.binding_beyond_folding >= cut
n_func = int(pos.is_functional.sum())
n_func = int(pos.is_functional.sum())
print(f"结合损伤超出折叠解释的最严重 20% 定为功能位置：{n_func} / {len(pos)}")
print()

rows = []
for name, r in RANKINGS.items():
    auc = stats.mannwhitneyu(r[pos.is_functional], r[~pos.is_functional], alternative="greater")
    n1, n2 = int(pos.is_functional.sum()), int((~pos.is_functional).sum())
    rows.append(
        {
            "排序方法": name,
            "AUROC": round(auc.statistic / (n1 * n2), 3),
            "p": f"{auc.pvalue:.1e}",
            "对残差的 Spearman": round(stats.spearmanr(pos.binding_beyond_folding, r).statistic, 3),
        }
    )
print(pd.DataFrame(rows).sort_values("AUROC", ascending=False).to_string(index=False))

print()
print("=" * 72)
print("3  固定 25% 的位置，放开的那些结合实测损伤有多小（越低越好）")
print("=" * 72)
rows = []
for name, r in RANKINGS.items():
    n = int(round(len(pos) * 0.25))
    free = pos.loc[r.nsmallest(len(pos) - n).index]
    rows.append(
        {
            "排序方法": name,
            "放开位置的实测结合": round(free.meas_bind.mean(), 4),
            "放开位置的实测折叠": round(free.meas_fold.mean(), 4),
        }
    )
print(pd.DataFrame(rows).sort_values("放开位置的实测结合").to_string(index=False))

pos.to_parquet("data/kras/kras_position_benchmark.parquet", index=False)
print(f"\nwrote data/kras/kras_position_benchmark.parquet ({len(pos)} positions)")
