import numpy as np
import pandas as pd
import pytest
import torch

from qafi_core import baseline, neural
from qafi_core.features import FEATURES

FAST = {"epochs": 6, "batch_size": 256, "hidden": 64, "learning_rate": 3e-3}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Made-up proteins: every residue has a hidden number, spread over its embedding along one direction.

    A variant's label is that number times a factor of the new amino acid, on a scale that differs between
    datasets, as real assays do.
    """
    rng = np.random.default_rng(0)
    folder = tmp_path_factory.mktemp("embeddings")
    direction = rng.normal(size=neural.EMBEDDING_SIZE)
    effect = {"C": 1.0, "D": -1.0, "E": 0.5}
    residues, variants = [], []
    for p in range(12):
        size = 60
        hidden = rng.normal(size=size)
        noise = 0.1 * rng.normal(size=(size, neural.EMBEDDING_SIZE))
        embedding = list((np.outer(hidden, direction) + noise).astype(np.float32))
        residues.append(
            pd.DataFrame({"uniprot_id": f"P{p}", "position": np.arange(size) + 1, "embedding": embedding})
        )
        for alt, factor in effect.items():
            block = pd.DataFrame(rng.normal(size=(size, len(FEATURES))), columns=FEATURES)
            block["label"] = hidden * factor * (1 + p % 3) + 5 * (p % 3)  # each dataset on its own scale
            block["dataset_id"], block["data_kind"] = f"d{p}", ["domainome", "mavedb", "proteingym"][p % 3]
            block["uniprot_id"], block["similarity_group"] = f"P{p}", f"G{p // 2}"
            block["uniprot_position"], block["ref_aa"], block["alt_aa"] = np.arange(size) + 1, "A", alt
            variants.append(block)
    for k in range(3):  # several files, as the real folder has
        pd.concat(residues[k * 4 : k * 4 + 4]).to_parquet(folder / f"part-{k}.parquet", index=False)
    return folder, pd.concat(variants, ignore_index=True)


def test_the_embedding_table_holds_the_wanted_proteins_and_is_written_once(world, tmp_path):
    folder, _ = world
    table, index = neural.embedding_table(folder, ["P1", "P7"], tmp_path / "table.npy")
    assert table.shape == (120, neural.EMBEDDING_SIZE) and table.dtype == np.float16
    assert set(index["uniprot_id"]) == {"P1", "P7"} and index["row"].tolist() == list(range(120))
    original = pd.concat(pd.read_parquet(f) for f in sorted(folder.glob("*.parquet")))
    wanted = original[(original.uniprot_id == "P7") & (original.position == 5)].embedding.iloc[0]
    row = index[(index.uniprot_id == "P7") & (index.position == 5)].row.item()
    np.testing.assert_allclose(table[row], wanted, rtol=1e-3, atol=1e-3)  # 16-bit numbers keep three digits
    again, _ = neural.embedding_table(tmp_path / "gone", [], tmp_path / "table.npy")  # not read a second time
    assert again.shape == table.shape


def test_labels_are_standardised_within_each_dataset(world):
    _, variants = world
    z = neural.standardised(variants)
    by_dataset = z.groupby(variants["dataset_id"])
    assert by_dataset.mean().abs().max() < 1e-6 and (by_dataset.std() - 1).abs().max() < 1e-3


def test_the_head_gives_twenty_scores_and_scales_features_as_in_training():
    mean, std = torch.full((16,), 2.0), torch.full((16,), 4.0)
    head = neural.Head(mean, std, hidden=32).eval()
    out = head(torch.randn(5, neural.EMBEDDING_SIZE), torch.randn(5, 16))
    assert out.shape == (5, 20)
    saved = {name: value for name, value in head.state_dict().items() if name in ("mean", "std")}
    assert (
        saved["mean"].tolist() == [2.0] * 16 and saved["std"].tolist() == [4.0] * 16
    )  # travel with the model
    assert neural.Head(hidden=32)(torch.randn(3, neural.EMBEDDING_SIZE)).shape == (3, 20)  # embedding only


def test_training_lowers_the_loss_and_generalises_to_held_out_proteins(world, tmp_path):
    folder, variants = world
    table, index = neural.embedding_table(folder, variants.uniprot_id.unique(), tmp_path / "table.npy")
    cv = neural.cross_validate(variants, table, index, standardise=True, settings=FAST, n_folds=3)
    assert cv["predicted"].notna().all() and len(cv) == len(variants)
    assert (cv.groupby("similarity_group")["fold"].nunique() == 1).all()  # relatives share a fold
    assert baseline.dataset_scores(cv)["spearman"].median() > 0.5  # learnt from other proteins
    curve = neural.learning_curve(variants, table, index, standardise=True, settings=FAST)
    assert curve["training_loss"].iloc[-1] < curve["training_loss"].iloc[0]
    assert list(curve.columns) == ["training_loss", "held_out_spearman"] and len(curve) == 6


def test_features_can_be_added_and_the_final_model_is_returned_on_the_cpu(world, tmp_path):
    folder, variants = world
    table, index = neural.embedding_table(folder, variants.uniprot_id.unique(), tmp_path / "table.npy")
    model = neural.fit(variants, table, index, features=FEATURES, settings={**FAST, "epochs": 1})
    assert next(model.parameters()).device.type == "cpu" and len(model.mean) == 16
    assert model.body[0].in_features == neural.EMBEDDING_SIZE + 16


def test_a_variant_without_embedding_is_an_error_not_a_silent_gap(world, tmp_path):
    folder, variants = world
    table, index = neural.embedding_table(folder, ["P0"], tmp_path / "table.npy")
    with pytest.raises(ValueError, match="have no embedding"):
        neural.cross_validate(variants, table, index, settings=FAST)


def test_predictions_are_combined_by_their_ranks_within_each_dataset():
    df = pd.DataFrame(
        {
            "dataset_id": ["a"] * 3 + ["b"] * 3,
            "x": [1.0, 2.0, 3.0, -50.0, -40.0, -30.0],  # different scales
            "y": [0.3, 0.2, 0.1, 7.0, 8.0, 9.0],
        }
    )
    combined = baseline.rank_average(df, ["x", "y"])
    assert combined.tolist() == pytest.approx([2 / 3, 2 / 3, 2 / 3, 1 / 3, 2 / 3, 1.0])
