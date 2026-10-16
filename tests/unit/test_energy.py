import numpy as np
import pytest
import torch

from qafi_core import energy


def test_folded_share_follows_boltzmann():
    assert energy.fraction_folded(0.0) == pytest.approx(0.5)  # equal free energy: half and half
    assert energy.fraction_folded(2.0) == pytest.approx(0.966, abs=1e-3)  # a typical small domain
    assert energy.fraction_folded(-1.0) == pytest.approx(0.159, abs=1e-3)  # after a -3 kcal/mol mutation
    # 1.4 kcal/mol at 30 C is about a factor of ten in the folded/unfolded ratio
    ratio = lambda g: energy.fraction_folded(g) / (1 - energy.fraction_folded(g))  # noqa: E731
    assert ratio(1.4) / ratio(0.0) == pytest.approx(10.3, abs=0.1)


def test_a_protein_that_cannot_fold_cannot_function():
    unfolded, folded = -10.0, 10.0
    assert energy.fraction_functional(unfolded, 1.0) == pytest.approx(0.0, abs=1e-6)
    # coupled equilibria: a very favourable function energy pulls some of an unstable protein into the fold
    assert energy.fraction_functional(unfolded, 5.0) > energy.fraction_functional(unfolded, 1.0)
    # fully folded: only the functional energy matters, a two-state sigmoid again
    assert energy.fraction_functional(folded, 1.0) == pytest.approx(energy.fraction_folded(1.0), abs=1e-6)
    # more stable or more functional always means more functional molecules
    g = np.linspace(-3, 3, 13)
    assert (np.diff(energy.fraction_functional(g, 1.0)) > 0).all()
    assert (np.diff(energy.fraction_functional(1.0, g)) > 0).all()


def test_numpy_and_torch_give_the_same_shares():
    g_fold, g_func = np.array([-2.0, 0.5, 3.0]), np.array([1.0, -0.5, 2.0])
    as_numpy = energy.fraction_functional(g_fold, g_func)
    as_torch = energy.fraction_functional(torch.tensor(g_fold), torch.tensor(g_func)).numpy()
    np.testing.assert_allclose(as_numpy, as_torch, rtol=1e-6)


def test_the_readout_layer_routes_each_kind_of_assay():
    torch.manual_seed(0)
    model = energy.build(width=8, n_features=2, n_proteins=1, n_datasets=3, hidden=16, dropout=0.0)
    n = 6
    vectors, features = torch.randn(n, 8), torch.randn(n, 2)
    alt, protein = torch.randint(0, 20, (n,)), torch.zeros(n, dtype=torch.long)
    dataset = torch.tensor([0, 0, 1, 1, 2, 2])
    kind = torch.tensor([0, 0, 1, 1, 2, 2])  # ddg, abundance, activity

    out = model(vectors, features, alt, protein, dataset, kind)
    ddg_fold, ddg_func = model.energies(vectors, features, alt)

    # ddg rows read the folding energy itself, in kcal/mol
    torch.testing.assert_close(out[:2], ddg_fold[:2])
    # abundance rows: offset + scale x folded share (offset 0, scale 1 at the start)
    torch.testing.assert_close(out[2:4], energy.fraction_folded(2.0 + ddg_fold[2:4]))
    torch.testing.assert_close(out[4:], energy.fraction_functional(2.0 + ddg_fold[4:], 1.0 + ddg_func[4:]))


def test_each_assay_gets_the_readout_its_measurement_implies():
    kind = energy.readout_kind
    assert kind("domainome", "GRB2 SH3 domain domainome 1.0") == energy.ABUNDANCE
    assert kind("proteingym", None) == energy.STABILITY_SCORE
    # the paired assays 14c checks the decomposition on
    assert kind("mavedb", "PTEN VAMP-seq") == energy.ABUNDANCE
    assert kind("mavedb", "PTEN lipid phosphatase scores") == energy.ACTIVITY
    assert kind("mavedb", "CYP2C9 abundance scores as measured by VAMP-seq") == energy.ABUNDANCE
    assert kind("mavedb", "CYP2C9 activity scores as measured by Click-seq") == energy.ACTIVITY
    assert kind("mavedb", "VKOR protein stability") == energy.ABUNDANCE
    assert kind("mavedb", "VKOR carboxylation activity") == energy.ACTIVITY
    assert kind("mavedb", "GCK variant abundance measured by DHFR-PCA") == energy.ABUNDANCE
    assert kind("mavedb", "LDLR missense variant effects on LDLR cell-surface abundance.") == energy.ABUNDANCE
    assert kind("mavedb", "CCR5 surface expression") == energy.ABUNDANCE
    assert kind("mavedb", "CCR5 HIV binding") == energy.ACTIVITY


def test_a_stability_score_is_an_affine_map_of_the_folding_energy():
    torch.manual_seed(0)
    model = energy.build(width=4, n_features=1, n_proteins=1, n_datasets=1, hidden=8, dropout=0.0)
    with torch.no_grad():
        model.scale.fill_(2.0)
        model.offset.fill_(0.5)
    v, f, alt = torch.randn(3, 4), torch.randn(3, 1), torch.tensor([0, 5, 9])
    zeros = torch.zeros(3, dtype=torch.long)

    out = model(v, f, alt, zeros, zeros, torch.full((3,), energy.KIND_CODE[energy.STABILITY_SCORE]))

    ddg_fold, _ = model.energies(v, f, alt)
    torch.testing.assert_close(out, 0.5 + 2.0 * ddg_fold)


def simulated(n_proteins=6, length=30, seed=0):
    """Measurements generated from known energies: a kcal/mol set, then abundance and activity per protein.

    The residue vector carries both energies plainly (first two dimensions), as an ideal language model
    would; the test asks whether the readout layer can tell which energy each kind of assay sees.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows, vectors = [], []
    for p in range(n_proteins):
        for pos in range(length):
            fold, func = rng.normal(-1.0, 1.2), rng.normal(-0.5, 1.0) if pos % 3 == 0 else 0.0
            vectors.append([fold, func, *rng.normal(size=6)])
            row = len(vectors) - 1
            truth = {"true_fold": fold, "true_func": func}
            f = energy.fraction_folded(2.0 + fold)
            a = energy.fraction_functional(2.0 + fold, 1.0 + func)
            for alt in "AG":
                base = {"protein": f"P{p}", "row": row, "mut_aa": alt, "f0": 0.0, **truth}
                if p == 0:
                    rows.append({**base, "dataset": "kcal", "kind": energy.DDG, "label": fold})
                rows.append({**base, "dataset": f"abundance_{p}", "kind": energy.ABUNDANCE, "label": f})
                rows.append({**base, "dataset": f"activity_{p}", "kind": energy.ACTIVITY, "label": a})
    return pd.DataFrame(rows), np.array(vectors, dtype=np.float16)


def test_the_two_energies_are_recovered_from_abundance_and_activity():
    rows, vectors = simulated()
    settings = {"hidden": 32, "dropout": 0.0, "epochs": 300, "batch": 256, "lr": 3e-3, "kcal_weight": 0.3}

    model, scaling, _ = energy.fit(rows, vectors, ["f0"], settings)
    one = rows.drop_duplicates("row")
    fold, func = energy.energies(model, scaling, one, vectors, ["f0"])

    # measured over three seeds: folding 0.90-0.94, function 0.83-0.90, leakage -0.08 to +0.28
    corr = lambda a, b: np.corrcoef(a, b)[0, 1]  # noqa: E731
    assert corr(fold, one.true_fold) > 0.85
    assert corr(func, one.true_func) > 0.75
    # and they are told apart: the folding energy does not become the functional one
    assert abs(corr(func, one.true_fold)) < 0.4


def test_both_sources_land_in_one_table_with_one_vector_matrix():
    import pandas as pd

    feats = {f: [0.1, 0.2] for f in energy.FEATURES}
    stability_rows = pd.DataFrame(
        {"domain": ["d1.pdb", "d1.pdb"], "mut_aa": ["A", "G"], "ddg": [-1.0, 0.5], "row": [0, 1], **feats}
    )
    label_feats = {("esm_wt_logprob" if f == "esm_wt" else f): [0.3, 0.4, 0.5] for f in energy.FEATURES}
    label_rows = pd.DataFrame(
        {
            "dataset_id": ["urn:a", "urn:a", "urn:b"],
            "data_kind": ["mavedb", "mavedb", "domainome"],
            "uniprot_id": ["P1", "P1", "P2"],
            "uniprot_position": [2, 3, 1],
            "alt_aa": ["L", "K", "W"],
            "label": [0.9, 0.1, 0.5],
            **label_feats,
        }
    )
    label_index = pd.DataFrame(
        {"uniprot_id": ["P1", "P1", "P1", "P2"], "position": [1, 2, 3, 1], "row": [0, 1, 2, 3]}
    )
    stability_vectors, label_vectors = np.zeros((2, 4)), np.ones((4, 4))

    rows, vectors = energy.stack(
        stability_rows,
        stability_vectors,
        label_rows,
        label_index,
        label_vectors,
        {"urn:a": "PTEN lipid phosphatase scores"},
    )

    assert vectors.shape == (6, 4)
    assert list(rows.kind) == [energy.DDG, energy.DDG, energy.ACTIVITY, energy.ACTIVITY, energy.ABUNDANCE]
    assert list(rows.row) == [0, 1, 3, 4, 5]  # label rows shifted past the two stability vectors
    assert (vectors[rows.row[2:]] == 1).all() and (vectors[rows.row[:2]] == 0).all()
    assert rows.esm_wt.tolist()[2:] == [0.3, 0.4, 0.5]
