"""Golden tests: numbers published in the QAFI paper, re-derived from compact copies of
the benchmark data in tests/golden/data/.

They are the platform's safety net. Any refactor of the science code, the pipelines or
the metrics must keep these passing; if one fails, the behaviour changed.
"""

from pathlib import Path

import pandas as pd
import pytest

from qafi_platform.metrics import auroc, median_per_protein_pearson

DATA = Path(__file__).parent / "data"


@pytest.mark.golden
def test_beltran_median_pearson_qafisplit3():
    # QAFI paper, Figure 5 / README: median per-protein Pearson r of qafisplit3 against
    # Beltran et al. 2025 scaled fitness, 393 proteins. Published value: 0.480.
    df = pd.read_parquet(DATA / "beltran_qafisplit3.parquet")
    assert df["uniprot"].nunique() == 393
    assert median_per_protein_pearson(df, "scaled_fitness", "qafisplit3") == pytest.approx(0.480, abs=0.0005)


@pytest.mark.golden
def test_clinvar_auroc_qafisplit3():
    # QAFI paper, ClinVar benchmark (lollipop figure): one pooled AUROC per predictor over
    # the 11,690 ClinVar variants scored by every predictor, label ClinSigSimple
    # (1 = pathogenic / likely pathogenic). Published values: qafisplit3 0.914, qafi2 0.906.
    df = pd.read_parquet(DATA / "clinvar_qafisplit3.parquet")
    assert len(df) == 11_690
    assert auroc(df["ClinSigSimple"], df["qafisplit3"]) == pytest.approx(0.914, abs=0.0005)
    assert auroc(df["ClinSigSimple"], df["qafi2"]) == pytest.approx(0.906, abs=0.0005)
