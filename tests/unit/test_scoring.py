import numpy as np
import pandas as pd

from qafi_platform import scoring


def test_auroc_orients_every_score_and_compares_them_on_shared_variants():
    rng = np.random.default_rng(0)
    pathogenic = rng.integers(0, 2, 400).astype(bool)
    df = pd.DataFrame({"pathogenic": pathogenic})
    df["function_score"] = np.where(pathogenic, 0.2, 0.8) + rng.normal(scale=0.3, size=400)  # low = damaging
    df["stability_score"] = rng.uniform(size=400)  # no signal
    df["zero_shot"] = np.nan
    df["am_pathogenicity"] = np.where(pathogenic, 0.9, 0.1)  # high = pathogenic: perfect once flipped
    df.loc[:99, "am_pathogenicity"] = np.nan  # AlphaMissense misses some variants
    table = scoring.auroc(df)
    assert table.loc["AlphaMissense", "AUROC (own coverage)"] == 1.0
    assert table.loc["function (ours)", "AUROC (own coverage)"] > 0.8
    assert 0.4 < table.loc["stability (ours)", "AUROC (own coverage)"] < 0.6
    assert "ESM-2 zero-shot" not in table.index  # no values at all
    assert (
        table.attrs["shared_variants"] == 300 and table.loc["AlphaMissense", "variants (own coverage)"] == 300
    )


def test_upload_resumes_retries_and_removes_files_that_are_not_ours(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS

    (tmp_path / "scores").mkdir()
    for name, size in [("part-00000.parquet", 3), ("part-00001.parquet", 5)]:
        (tmp_path / "scores" / name).write_bytes(b"x" * size)
    target = "/Volumes/qafi_dev/ml/files/proteome/r1/scores"
    there = [  # part 0 arrived earlier; part 1 was cut off; part 9 is left from something else
        NS(path=f"{target}/part-00000.parquet", file_size=3),
        NS(path=f"{target}/part-00001.parquet", file_size=2),
        NS(path=f"{target}/part-00009.parquet", file_size=7),
    ]
    uploaded, deleted, failures = [], [], [1]

    def upload(path, contents, overwrite):
        if failures:  # the first try breaks off
            failures.pop()
            raise ConnectionError("broken pipe")
        uploaded.append((path, len(contents.read())))

    w = NS(files=NS(list_directory_contents=lambda d: iter(there), upload=upload, delete=deleted.append))
    monkeypatch.setattr(scoring.time, "sleep", lambda s: None)
    assert scoring.upload(w, tmp_path, "r1") == target
    assert uploaded == [(f"{target}/part-00001.parquet", 5)]
    assert deleted == [f"{target}/part-00009.parquet"]
