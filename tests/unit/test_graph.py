import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch

from qafi_core import baseline, graph, neural
from qafi_core.features import FEATURES

FAST = {"epochs": 40, "proteins_per_batch": 4, "learning_rate": 3e-3, "hidden": 32, "dropout": 0.0}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Made-up proteins, each a chain whose residues touch the two before and the two after them.

    A variant's label is the average ``rsa`` of the residue's NEIGHBOURS: a residue's own features say
    nothing about it, so only a network that looks along the edges can learn it. The embedding carries
    ``plddt`` along one direction, for the probe.
    """
    rng = np.random.default_rng(0)
    direction = rng.normal(size=neural.EMBEDDING_SIZE)
    residues, variants, embeddings = [], [], []
    for p in range(12):
        size = 30
        rsa, plddt = rng.uniform(size=size), rng.uniform(50, 100, size=size)
        around = [[j + 1 for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < size] for i in range(size)]
        block = pd.DataFrame({"uniprot_id": f"P{p}", "position": np.arange(size) + 1, "aa": "A", "rsa": rsa})
        block["plddt"], block["neighbours"] = plddt, around
        for name in ("contacts_8", "contacts_12", "hse_up", "hse_down"):
            block[name] = rng.integers(0, 20, size=size)
        residues.append(block)
        noise = 0.05 * rng.normal(size=(size, neural.EMBEDDING_SIZE))
        embeddings.append(
            block[["uniprot_id", "position"]].assign(embedding=list(np.outer(plddt / 100, direction) + noise))
        )
        v = pd.DataFrame(rng.normal(size=(size, len(FEATURES))), columns=FEATURES)
        v["label"] = [rsa[[j - 1 for j in n]].mean() for n in around]
        v["dataset_id"], v["data_kind"] = f"d{p}", ["domainome", "mavedb", "proteingym"][p % 3]
        v["uniprot_id"], v["similarity_group"] = f"P{p}", f"G{p // 2}"
        v["uniprot_position"], v["ref_aa"], v["alt_aa"] = np.arange(size) + 1, "A", "C"
        variants.append(v)
    folder = tmp_path_factory.mktemp("graph")
    pd.concat(embeddings).to_parquet(folder / "embeddings.parquet", index=False)
    df = pd.concat(variants, ignore_index=True)
    table, index = neural.embedding_table(
        folder / "embeddings.parquet", df.uniprot_id.unique(), folder / "table.npy"
    )
    residues = pd.concat(residues, ignore_index=True)
    return df, residues, table, index, graph.protein_graphs(df, residues, table, index)


def test_a_protein_becomes_nodes_edges_and_its_variants(world):
    df, _, _, _, graphs = world
    g = graphs["P3"]
    assert len(graphs) == 12 and g.num_nodes == 30 and g.embedding.shape == (30, neural.EMBEDDING_SIZE)
    edges = set(zip(*g.edge_index.tolist(), strict=True))  # (sender, receiver), nodes counted from 0
    assert len(edges) == 30 * 4 - 6 and {(1, 0), (2, 0), (0, 1), (28, 29)} <= edges and (3, 0) not in edges
    assert g.variant_node.tolist() == list(range(30))
    assert df.iloc[g.variant_row.numpy()]["uniprot_id"].eq("P3").all()  # each variant knows its row


def test_residues_too_far_to_be_heard_are_dropped_without_changing_any_prediction(world):
    df, residues, table, index, _ = world
    few = df[df["uniprot_position"] <= 3].reset_index(drop=True)  # only the first three residues are labelled
    near = graph.protein_graphs(few, residues, table, index, steps=2)
    whole = graph.protein_graphs(few, residues, table, index, steps=30)
    assert (
        near["P0"].num_nodes == 7 and whole["P0"].num_nodes == 30
    )  # 3 labelled, 2 one step away, 2 two steps
    for inputs in ("structure", "esm"):
        torch.manual_seed(0)
        setup = {**graph.INPUTS[inputs], "hidden": 16}
        model = graph._new_network(list(whole.values()), inputs, 2, setup, "cpu")
        far_scores, far_rows = graph.predict(model, list(whole.values()), "cpu")
        near_scores, near_rows = graph.predict(model, list(near.values()), "cpu")
        assert (far_rows == near_rows).all()
        np.testing.assert_allclose(
            near_scores, far_scores, atol=1e-5
        )  # two layers reach two steps, no further


def test_joining_proteins_into_a_batch_keeps_variants_on_their_own_nodes(world):
    *_, graphs = world
    batch = Batch.from_data_list([graphs["P0"], graphs["P1"]])
    assert batch.num_nodes == 60
    assert batch.variant_node[30:].tolist() == list(range(30, 60))  # shifted like the second protein's edges
    assert batch.edge_index[:, graphs["P0"].edge_index.shape[1] :].min() == 30  # no edge between the two


def test_only_a_network_with_graph_layers_can_learn_from_neighbours(world):
    df, _, _, _, graphs = world
    score = {}
    for layers in (0, 2):
        cv = graph.cross_validate(df, graphs, inputs="structure", layers=layers, settings=FAST, n_folds=3)
        assert cv["predicted"].notna().all() and (cv.groupby("similarity_group")["fold"].nunique() == 1).all()
        score[layers] = baseline.dataset_scores(cv)["spearman"].median()
    assert score[2] > 0.6 > 0.3 > score[0]


def test_the_embedding_input_and_the_learning_curve_run(world):
    df, _, _, _, graphs = world
    quick = {**FAST, "epochs": 2}
    cv = graph.cross_validate(df, graphs, inputs="esm", layers=1, settings=quick, n_folds=3)
    assert cv["predicted"].notna().all()
    curve = graph.learning_curve(df, graphs, inputs="structure", layers=1, settings=quick)
    assert list(curve.columns) == ["training_loss", "held_out_spearman"] and len(curve) == 2


def test_graphs_of_another_table_are_refused(world):
    df, _, _, _, graphs = world
    with pytest.raises(ValueError, match="another table"):
        graph.cross_validate(df.head(100), graphs, settings=FAST)


def test_the_probe_finds_what_the_embedding_holds_and_nothing_else(world):
    df, residues, table, index, _ = world
    groups = df.groupby("uniprot_id")["similarity_group"].first()
    assert graph.probe(table, index, residues, "plddt", groups) > 0.9  # written into the embedding
    assert abs(graph.probe(table, index, residues, "rsa", groups)) < 0.3  # not in it


def test_a_run_records_its_inputs_and_settings():
    recorded = graph.parameters(graph.RUNS["gnn_structure_2_layers"])
    assert recorded["inputs"] == "structure" and recorded["graph_layers"] == 2 and recorded["epochs"] == 30
    assert recorded["variant_features"] == "blosum62,d_hydrophobicity,d_volume,d_charge,d_volume_buried"
    assert graph.parameters(graph.RUNS["gnn_esm_0_layers"])["model"] == "network without graph layers"
