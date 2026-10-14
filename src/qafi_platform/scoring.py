"""Running the proteome scoring and publishing it (chapter 09c).

python -m qafi_platform.scoring RESIDUES SEQUENCES FOLDER
    scores every protein of SEQUENCES on this Mac with the champion models of the registry, gently: at
    the lowest priority, it unloads the models when memory pressure gets critical and tries again later,
    and resumes where it stopped (``qafi_core.proteome.run``)
upload(w, folder, release)        the parquet parts into the volume ml.files (resumable)
load(w, release, versions)        the volume into the table ml.proteome_scores
clinvar(w)                        ClinVar variants with our scores and the published ones, for the AUROC
auroc(df, columns)                how well each score separates pathogenic from benign variants
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qafi_core.release import PUBLISHED
from qafi_platform.lakehouse import sql

SCORES = {
    "function_score": "function (ours)",
    "stability_score": "stability (ours)",
    "zero_shot": "ESM-2 zero-shot",
}


def memory_pressure() -> int:
    """macOS memory pressure: 1 normal, 2 warning, 4 critical."""
    out = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True)
    return int(out.stdout or 1)


def calm(highest: int, wait: float = 30) -> bool:
    """Memory pressure at most ``highest``, asked twice ``wait`` seconds apart (a model load spikes it)."""
    for attempt in range(2):
        if memory_pressure() <= highest:
            return True
        if attempt == 0:
            time.sleep(wait)
    return False


def upload(w, folder: Path, release: str, catalog: str = "qafi_dev", attempts: int = 30) -> str:
    """Copy the scored parquet parts into the volume, one file at a time; returns the volume folder.

    Resumable: a part already there with the same size is skipped, a failed transfer is tried again (for up
    to half an hour, a patchy connection), and files there that are not in ``folder`` are removed.
    """
    from databricks.sdk.errors import NotFound

    target = f"/Volumes/{catalog}/ml/files/proteome/{release}/scores"
    try:
        there = {Path(e.path).name: e.file_size for e in w.files.list_directory_contents(target)}
    except NotFound:
        there = {}
    parts = sorted((Path(folder) / "scores").glob("part-*.parquet"))
    for name in set(there) - {part.name for part in parts}:
        w.files.delete(f"{target}/{name}")
    for part in parts:
        if there.get(part.name) == part.stat().st_size:
            continue
        for attempt in range(attempts):
            try:
                with part.open("rb") as contents:
                    w.files.upload(f"{target}/{part.name}", contents, overwrite=True)
                break
            except OSError:  # includes a lost connection
                if attempt == attempts - 1:
                    raise
                time.sleep(min(2**attempt, 60))
    return target


def load(w, release: str, versions: dict[str, str], catalog: str = "qafi_dev") -> pd.DataFrame:
    """Replace ``ml.proteome_scores`` with the uploaded release; returns its size per score column."""
    source = f"/Volumes/{catalog}/ml/files/proteome/{release}/scores"
    sql(
        w,
        f"""CREATE OR REPLACE TABLE {catalog}.ml.proteome_scores
            CLUSTER BY (uniprot_id)
            COMMENT 'Released scores of every missense variant of the human reference proteome (notebook 09c)'
            AS SELECT uniprot_id, CAST(position AS INT) AS position, ref_aa, alt_aa,
                      CAST(zero_shot AS FLOAT) AS zero_shot,
                      CAST(stability_score AS FLOAT) AS stability_score,
                      CAST(function_score AS FLOAT) AS function_score,
                      '{release}' AS release,
                      '{versions["stability"]}' AS stability_head_version,
                      '{versions["trees"]}' AS function_trees_version
               FROM read_files('{source}', format => 'parquet')""",
        timeout=3600,
    )
    return sql(
        w,
        f"""SELECT count(*) AS variants, count(DISTINCT uniprot_id) AS proteins,
                   count_if(function_score IS NULL) AS without_function_score
            FROM {catalog}.ml.proteome_scores""",
    )


def clinvar(w, path: Path, catalog: str = "qafi_dev") -> pd.DataFrame:
    """ClinVar variants of proteins outside our data, with our scores and the published ones (downloaded)."""
    from qafi_platform.lakehouse import download

    published = ", ".join(f"v.{column}" for column in PUBLISHED)
    download(
        w,
        f"""SELECT c.uniprot_id, c.position, c.ref_aa, c.alt_aa, c.pathogenic,
                   s.function_score, s.stability_score, s.zero_shot, {published}
            FROM {catalog}.gold.fct_clinvar_missense c
            JOIN {catalog}.ml.proteome_scores s USING (uniprot_id, position, ref_aa, alt_aa)
            LEFT JOIN {catalog}.silver.variant_scores v USING (uniprot_id, position, ref_aa, alt_aa)
            WHERE NOT c.in_our_data""",
        path,
    )
    return pd.read_parquet(path)


def auroc(df: pd.DataFrame) -> pd.DataFrame:
    """Area under the ROC curve of every score for pathogenic against benign, on two sets of variants.

    Our scores and the published ones are oriented so that low = damaging (as our labels); the AUROC is
    the chance that a random pathogenic variant scores lower than a random benign one. ``own coverage`` uses
    every variant a score has; ``shared`` only the variants every score has, the fair comparison.
    """
    from sklearn.metrics import roc_auc_score

    columns = {**SCORES, **{column: name for column, (name, _) in PUBLISHED.items()}}
    signs = {**dict.fromkeys(SCORES, 1), **{column: sign for column, (_, sign) in PUBLISHED.items()}}
    present = [c for c in columns if c in df and df[c].notna().any()]
    shared = df.dropna(subset=present)

    def area(part: pd.DataFrame, column: str) -> float:  # undefined unless both classes are there
        if part["pathogenic"].nunique() < 2:
            return np.nan
        return roc_auc_score(part["pathogenic"], -signs[column] * part[column])

    rows = []
    for column in present:
        have = df[df[column].notna()]
        rows.append(
            {
                "score": columns[column],
                "variants (own coverage)": len(have),
                "AUROC (own coverage)": area(have, column),
                "AUROC (shared variants)": area(shared, column),
            }
        )
    table = pd.DataFrame(rows).set_index("score").round(3)
    table.attrs["shared_variants"] = len(shared)
    return table.sort_values("AUROC (shared variants)", ascending=False)


if __name__ == "__main__":  # the long run on this Mac
    import argparse
    import gc
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")  # PyTorch and XGBoost in one process (chapter 07f)

    from qafi_core import proteome
    from qafi_platform import registry

    parser = argparse.ArgumentParser(description="Score the proteome here with the champion models.")
    parser.add_argument("residues", help="parquet of gold.dim_residue_proteome")
    parser.add_argument("sequences", help="parquet with uniprot_id and sequence")
    parser.add_argument("folder", help="where the parquet parts go")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N proteins (a trial)")
    args = parser.parse_args()

    def log(message: str) -> None:
        print(f"{time.strftime('%H:%M')} {message}", flush=True)

    root = Path(__file__).parents[2]
    amino_acid = pd.read_csv(root / "dbt/seeds/amino_acid.csv")
    blosum62 = pd.read_csv(root / "dbt/seeds/blosum62.csv")
    sequences = pd.read_parquet(args.sequences).sort_values("uniprot_id")
    if args.limit:
        sequences = sequences.head(args.limit)
    folder = Path(args.folder)

    def left() -> int:
        return len(set(sequences["uniprot_id"]) - proteome.done(folder))

    if left():  # the models are fetched only when there is something to score
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        head = registry.load(registry.full_name("stability"), flavor="pytorch")
        trees = registry.load(registry.full_name("trees"), flavor="xgboost")
        tables = registry.distributions(registry.full_name("trees"))
        log(f"{left():,} proteins to score with the champion models, versions {registry.champions()}")
    while left():  # other programs come first
        if not calm(2):  # a Mac with an editor open often sits at 'warning': only 'critical' waits
            log("waiting until memory pressure is below critical (close heavy programs to start sooner)")
            while not calm(2):
                time.sleep(300)
        esm, tokenizer = proteome._language_model().load()
        head = head.to(next(esm.parameters()).device)
        models = {"esm": esm, "tokenizer": tokenizer, "head": head, "trees": trees, "tables": tables}
        proteome.run(sequences, Path(args.residues), folder, models, amino_acid, blosum62,
                     keep_going=lambda: calm(2), log=log)  # fmt: skip
        del esm, models
        gc.collect()
        proteome._language_model().free_memory()
        if left():
            log(f"memory is critical: model unloaded, {left():,} proteins left; next try in 5 minutes")
            time.sleep(300)
    log("all proteins scored")
