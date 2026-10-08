import numpy as np
import pandas as pd

from qafi_core import labels


def test_new_labels_are_compared_variant_by_variant():
    rng = np.random.default_rng(0)
    paper_scores = rng.normal(size=60)
    paper = pd.DataFrame(
        {
            "uniprot": "P1",
            "protein": "PROT",
            "pos": range(1, 61),
            "first": "A",
            "second": "G",
            "score_log_normalized": paper_scores,
        }
    )
    new = pd.DataFrame(
        {
            "dataset_id": ["same assay"] * 60 + ["too small"] * 10,
            "uniprot_id": "P1",
            "uniprot_position": list(range(1, 61)) + list(range(1, 11)),
            "ref_aa": "A",
            "alt_aa": "G",
            "label": list(0.5 * paper_scores + 0.2) + list(rng.normal(size=10)),  # same ranking, other scale
        }
    )
    [row] = labels.compare_with_paper(new, paper).to_dict("records")  # the 10-variant dataset is left out
    assert row["dataset_id"] == "same assay" and row["variants"] == 60
    assert row["pearson"] > 0.999
