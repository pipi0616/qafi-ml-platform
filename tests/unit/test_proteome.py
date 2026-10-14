import numpy as np
import pandas as pd
import pytest
import torch

from qafi_core import neural, proteome, release
from qafi_core.features import AMINO_ACIDS, FEATURES, amino_acid_table, blosum62_table

VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", *"LAGVSERTIDPKQNFYMHWCXBUZO", ".", "-", "<null_1>", "<mask>"]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A tiny ESM-2 with random weights and its tokenizer (07e's test model)."""
    from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer

    vocab = tmp_path_factory.mktemp("esm") / "vocab.txt"
    vocab.write_text("\n".join(VOCAB))
    torch.manual_seed(0)
    config = EsmConfig(
        vocab_size=len(VOCAB), hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=1026, position_embedding_type="rotary",
        pad_token_id=1, mask_token_id=32, token_dropout=True, emb_layer_norm_before=False,
    )  # fmt: skip
    return EsmForMaskedLM(config).eval(), EsmTokenizer(str(vocab))


def _residues(sequence: str, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(sequence)
    logits = rng.normal(size=(n, 20))
    logprobs = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    frame = pd.DataFrame({"uniprot_id": "P1", "position": np.arange(1, n + 1), "aa": list(sequence)})
    frame["logprobs"] = list(logprobs.astype(np.float32))
    frame["esm_wt_logprob"] = [
        lp[AMINO_ACIDS.index(a)] if a in AMINO_ACIDS else np.nan
        for lp, a in zip(logprobs, sequence, strict=True)
    ]  # noqa: E501
    for name in release.RESIDUE_FEATURES[1:]:
        frame[name] = rng.uniform(size=n)
    return frame


def test_every_standard_residue_gets_its_19_substitutions_with_the_sql_formulas():
    residues = _residues("MKRXW")  # X is not an amino acid: no variants there
    out = proteome.variant_features(residues, amino_acid_table(), blosum62_table())
    assert len(out) == 4 * 19 and set(FEATURES) <= set(out.columns)
    k_to_e = out[(out.uniprot_position == 2) & (out.alt_aa == "E")].iloc[0]
    assert k_to_e["ref_aa"] == "K" and k_to_e["blosum62"] == 1 and k_to_e["d_charge"] == -2  # +1 -> -1
    assert k_to_e["d_volume"] == pytest.approx(138.4 - 168.6, abs=1e-4)
    lp = residues.loc[1, "logprobs"]
    assert k_to_e["esm_delta"] == pytest.approx(
        lp[AMINO_ACIDS.index("E")] - lp[AMINO_ACIDS.index("K")], abs=1e-6
    )
    assert k_to_e["d_volume_buried"] == pytest.approx(
        k_to_e["d_volume"] * (1 - residues.loc[1, "rsa"]), abs=1e-4
    )
    assert not (out.ref_aa == out.alt_aa).any() and 4 not in set(out.uniprot_position)  # X: no variants


def test_reading_in_fixed_shapes_gives_the_same_vectors_as_one_by_one(tiny):
    model, tokenizer = tiny
    sequences = ["MKTAYIAKQR", "GAVLIPFWMSTCYNQDEKRH" * 15, "MEEPQ"]  # 300 residues: another bucket
    together = proteome.embed(sequences, model, tokenizer, tokens_per_batch=512)
    alone = [proteome.embed([s], model, tokenizer, tokens_per_batch=512)[0] for s in sequences]
    assert [v.shape for v in together] == [(10, 16), (300, 16), (5, 16)]
    for a, b in zip(together, alone, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-4)  # padding and batch-mates change nothing


def test_the_stability_network_treats_missing_structure_as_the_training_mean():
    head = neural.Head(mean=torch.full((10,), 0.5), std=torch.ones(10), hidden=32).eval()
    vectors = np.random.default_rng(0).normal(size=(3, 1280)).astype(np.float32)
    features = np.full((3, 10), 0.5, np.float32)
    with_mean = proteome.stability(head, vectors, features)
    features[:, 2:] = np.nan  # no structure values
    assert proteome.stability(head, vectors, features) == pytest.approx(with_mean)
    assert with_mean.shape == (3, 20)


def test_a_stopped_run_resumes_where_it_stopped(tmp_path, tiny, monkeypatch):
    model, tokenizer = tiny
    sequences = pd.DataFrame(
        {"uniprot_id": ["P1", "P2", "P3"], "sequence": ["MKTAYIAKQR", "MEEPQSDPSV", "GAVLIP"]}
    )
    residues = pd.concat(
        [
            _residues(s).assign(uniprot_id=p)
            for p, s in zip(sequences.uniprot_id, sequences.sequence, strict=True)
        ]
    )  # noqa: E501
    residues.to_parquet(tmp_path / "residues.parquet", index=False)
    monkeypatch.setattr(neural, "EMBEDDING_SIZE", 16)  # the tiny ESM's vectors
    head = neural.Head(mean=torch.zeros(10), std=torch.ones(10), hidden=32).eval()

    class Trees:
        def predict(self, frame):
            return frame["esm_delta"].to_numpy()

    tables = {name: list(np.linspace(-10, 10, 1001)) for name in release.FUNCTION_MEMBERS}
    models = {"esm": model, "tokenizer": tokenizer, "head": head, "trees": Trees(), "tables": tables}
    answers = iter([True, False])  # the first chunk runs, then memory "gets critical"
    args = (
        sequences,
        tmp_path / "residues.parquet",
        tmp_path / "out",
        models,
        amino_acid_table(),
        blosum62_table(),
    )
    assert proteome.run(*args, chunk_size=2, keep_going=lambda: next(answers), log=lambda m: None) == 2
    assert proteome.done(tmp_path / "out") == {"P1", "P2"}
    assert proteome.run(*args, chunk_size=2, log=lambda m: None) == 1  # only P3 is left
    scores = pd.concat(pd.read_parquet(p) for p in (tmp_path / "out" / "scores").glob("*.parquet"))
    assert len(scores) == 19 * (10 + 10 + 6) and list(scores.columns) == proteome.OUTPUT
    assert scores["function_score"].between(0, 1).all()
