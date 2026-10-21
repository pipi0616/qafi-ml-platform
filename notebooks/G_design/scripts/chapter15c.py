"""15c: which ranking picks the positions a design must hold fixed?

Sumida et al. 2024 froze positions by two rules -- distance to the ligand, then a percentile of MSA
conservation -- and had to find the percentile empirically per protein. Nobody ranks positions by a
per-position functional cost. This asks, on measured data, which ranking identifies the positions where
mutations actually destroy the phenotype.

Truth: for each assay and position, the mean measured label over its substitutions. Labels run around 1
for wild-type-like and 0 for dead, so a low mean marks a position that must not be touched.

No wet lab is needed because the question is not "does this design work" but "is this the right set of
positions to protect", and that is answered by measurements that already exist.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MODELS = Path("data/models")
SITES = Path("data/sites")
MIN_POSITIONS = 20  # an assay needs enough positions for a ranking to mean anything

variants = pd.read_parquet(
    MODELS / "test_variants.parquet",
    columns=[
        "dataset_id",
        "data_kind",
        "uniprot_id",
        "uniprot_position",
        "label",
        "esm_entropy",
        "esm_delta",
        "rsa",
        "plddt",
    ],
)
sites = pd.read_parquet(SITES / "positions.parquet")
print(f"variants {len(variants):,} | site map {len(sites):,} positions")

# --- truth and the rankings that do not need a model run ------------------------------------------
per_position = (
    variants.groupby(["dataset_id", "data_kind", "uniprot_id", "uniprot_position"])
    .agg(
        measured=("label", "mean"),
        n=("label", "size"),
        esm_entropy=("esm_entropy", "first"),
        esm_delta=("esm_delta", "mean"),
        rsa=("rsa", "first"),
        plddt=("plddt", "first"),
    )
    .reset_index()
)
per_position = per_position.merge(
    sites.rename(columns={"position": "uniprot_position"}),
    on=["uniprot_id", "uniprot_position"],
    how="left",
)
print(
    f"positions with an assay: {len(per_position):,}; with a site-map entry: "
    f"{per_position.fold_cost.notna().sum():,}"
)

d = per_position.dropna(subset=["fold_cost", "func_cost"]).copy()

# Every ranking is oriented so that HIGH = "important, freeze it".
d["rank_func"] = -d.func_cost  # energies are positive when favourable, so cost is negative
d["rank_fold"] = -d.fold_cost
d["rank_both"] = -(d.fold_cost + d.func_cost)
d["rank_conservation"] = -d.esm_entropy  # low entropy = conserved
d["rank_esm_delta"] = -d.esm_delta  # how bad ESM-2 thinks substitutions here are
d["rank_burial"] = -d.rsa  # buried positions first

RANKINGS = {
    "我们 功能能": "rank_func",
    "我们 折叠能": "rank_fold",
    "我们 两个相加": "rank_both",
    "保守性 (ESM 熵)": "rank_conservation",
    "ESM-2 替换代价": "rank_esm_delta",
    "埋藏程度 (RSA)": "rank_burial",
}

mp = Path("data/mpnn_probs/mpnn_positions.parquet")
if mp.exists():
    probs = pd.read_parquet(mp)
    d = d.merge(probs, on=["uniprot_id", "uniprot_position"], how="left")
    d["rank_mpnn"] = -d.mpnn_entropy
    RANKINGS["生成器自己 (MPNN 熵)"] = "rank_mpnn"
    print(f"ProteinMPNN entropies joined for {d.rank_mpnn.notna().sum():,} positions")

big = d.groupby("dataset_id").filter(lambda g: len(g) >= MIN_POSITIONS)
print(
    f"\nassays with >= {MIN_POSITIONS} positions: {big.dataset_id.nunique()} "
    f"({big.uniprot_id.nunique()} proteins, {len(big):,} positions)\n"
)

# --- 1. does the ranking order positions by measured damage? ---------------------------------------
print("=" * 78)
print("1  每组实验内部：排序 vs 实测损伤（Spearman，越高越好）")
print("=" * 78)
rows = []
for name, col in RANKINGS.items():
    rhos = [
        -stats.spearmanr(g.measured, g[col]).statistic  # high rank should mean low measured
        for _, g in big.groupby("dataset_id")
        if g[col].notna().all() and g.measured.nunique() > 2
    ]
    rows.append(
        {
            "排序方法": name,
            "中位 Spearman": round(float(np.median(rhos)), 3),
            "赢过一半实验": sum(r > 0 for r in rhos),
            "实验数": len(rhos),
        }
    )
print(pd.DataFrame(rows).sort_values("中位 Spearman", ascending=False).to_string(index=False))

# --- 2. Sumida's question: freeze a budget, how safe is what is left free? -------------------------
print()
print("=" * 78)
print("2  Sumida 的问法：固定 N% 的位置，放开的那些实测有多安全（越高越好）")
print("=" * 78)
rows = []
for budget in (0.25, 0.50):
    row = {"固定比例": f"{budget:.0%}"}
    for name, col in RANKINGS.items():
        kept = []
        for _, g in big.groupby("dataset_id"):
            if g[col].isna().any():
                continue
            n = max(1, int(round(len(g) * budget)))
            free = g.nsmallest(len(g) - n, col)  # the ones NOT frozen
            kept.append(free.measured.mean())
        row[name] = round(float(np.median(kept)), 3)
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))
print("\n（数字是放开位置上的实测读数中位数；1 附近=正常，0=失活。越高说明放开得越对。）")

big.to_parquet("data/sites/ranking_benchmark.parquet", index=False)
print(f"\nwrote data/sites/ranking_benchmark.parquet ({len(big):,} rows)")
