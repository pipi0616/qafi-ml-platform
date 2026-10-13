import hashlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import torch
from lakehouse import language_model as lm

# ESM-2's vocabulary, in its order: special tokens, the amino acids (commonest first), rare letters, gaps
VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", *"LAGVSERTIDPKQNFYMHWCXBUZO", ".", "-", "<null_1>", "<mask>"]
SEQUENCES = ["MKTAYIAKQRQISFVKSHFSRQ", "MEEPQSDPSV", "GAVLIPFWMSTCYNQDEKRH" * 2]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A tiny ESM-2 with random weights: the real model's code and vocabulary, nothing to download."""
    from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer

    vocab = tmp_path_factory.mktemp("esm") / "vocab.txt"
    vocab.write_text("\n".join(VOCAB))
    torch.manual_seed(0)
    config = EsmConfig(
        vocab_size=len(VOCAB),
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=1026,
        position_embedding_type="rotary",
        pad_token_id=1,
        mask_token_id=32,
        token_dropout=True,
        emb_layer_norm_before=False,
    )
    return EsmForMaskedLM(config).eval(), EsmTokenizer(str(vocab))


def test_windows_cover_a_long_sequence_and_overlap_by_half():
    assert lm.windows(500) == [(0, 500)]
    spans = lm.windows(3000)
    assert spans[0][0] == 0 and spans[-1][1] == 3000
    assert all(end - start == lm.WINDOW for start, end in spans)
    assert all(b[0] - a[0] <= lm.WINDOW // 2 for a, b in zip(spans, spans[1:], strict=False))


def test_each_residue_comes_from_the_window_where_it_is_most_central():
    spans = lm.windows(25, size=10)  # (0, 10), (5, 15), (10, 20), (15, 25)
    choice = lm.most_central(25, spans)
    assert all(spans[k][0] <= position < spans[k][1] for position, k in enumerate(choice))
    assert choice[7] == 0  # 2 from the end of the first window, 2 from the start of the second: a tie
    assert choice[8] == 1  # 1 from the end of the first window, 3 from the start of the second


def test_one_row_of_20_probabilities_and_one_vector_per_residue(tiny):
    ((logprobs, vectors),) = lm.read_proteins(SEQUENCES[:1], *tiny, embeddings=True)
    assert logprobs.shape == (len(SEQUENCES[0]), 20) and vectors.shape == (len(SEQUENCES[0]), 16)
    assert np.allclose(np.exp(logprobs).sum(axis=1), 1, atol=1e-5)


def test_same_as_the_standard_masked_language_model_output(tiny):
    model, tokenizer = tiny
    with torch.inference_mode():
        logits = model(**tokenizer(SEQUENCES[:1], return_tensors="pt")).logits[0, 1:-1]
    amino_acids = tokenizer.convert_tokens_to_ids(list(lm.AMINO_ACIDS))
    ((logprobs, _),) = lm.read_proteins(SEQUENCES[:1], model, tokenizer)
    assert np.allclose(logprobs, torch.log_softmax(logits[:, amino_acids], dim=-1).numpy(), atol=1e-6)


def test_batching_and_padding_do_not_change_the_result(tiny):
    together = lm.read_proteins(SEQUENCES, *tiny, embeddings=True, tokens_per_batch=30)  # several batches
    for sequence, (logprobs, vectors) in zip(SEQUENCES, together, strict=True):
        ((alone, alone_vectors),) = lm.read_proteins([sequence], *tiny, embeddings=True)
        assert np.allclose(logprobs, alone, atol=1e-5) and np.allclose(vectors, alone_vectors, atol=1e-5)


def test_a_long_sequence_is_stitched_from_its_windows(tiny):
    sequence = SEQUENCES[2]  # 40 residues, read in windows of 10
    ((logprobs, _),) = lm.read_proteins([sequence], *tiny, window=10)
    spans = lm.windows(len(sequence), 10)
    for position, k in enumerate(lm.most_central(len(sequence), spans)):
        start, end = spans[k]
        ((piece, _),) = lm.read_proteins([sequence[start:end]], *tiny, window=10)
        assert np.allclose(logprobs[position], piece[position - start], atol=1e-5)


def test_rare_letters_are_read(tiny):
    ((logprobs, _),) = lm.read_proteins(["MAUXGK"], *tiny)  # selenocysteine and an unknown residue
    assert logprobs.shape == (6, 20) and np.isfinite(logprobs).all()


def test_saved_files_have_the_silver_tables_columns(tiny, tmp_path):
    proteins = pd.DataFrame({"uniprot_id": ["P1", "P2", "P3"], "sequence": SEQUENCES})
    assert lm.read_to_parquet(proteins, *tiny, tmp_path, chunk_size=2, log=lambda m: None) == 3
    residues = pq.read_table(tmp_path / "residues")
    assert residues.num_rows == sum(len(s) for s in SEQUENCES)
    assert str(residues.schema.field("position").type) == "int32"
    assert str(residues.schema.field("logprobs").type) == "list<element: float>"
    read = pd.read_parquet(tmp_path / "proteins").sort_values("uniprot_id")
    assert read["sequence_md5"].tolist() == [hashlib.md5(s.encode()).hexdigest() for s in SEQUENCES]
    assert read.columns.tolist() == [
        "uniprot_id", "sequence_md5", "model_version", "residues", "windows", "embedded", "read_at"
    ]  # fmt: skip


def test_an_interrupted_run_goes_on_where_it_stopped(tiny, tmp_path):
    proteins = pd.DataFrame({"uniprot_id": ["P1", "P2", "P3"], "sequence": SEQUENCES})
    lm.read_to_parquet(proteins.head(2), *tiny, tmp_path, chunk_size=1, log=lambda m: None)
    (tmp_path / "proteins" / "part-00001.parquet").unlink()  # cut off after the residues of P2 were saved
    assert lm.read_to_parquet(proteins, *tiny, tmp_path, chunk_size=1, log=lambda m: None) == 2  # P2 and P3
    residues = pd.read_parquet(tmp_path / "residues")
    assert residues.groupby("uniprot_id").size().to_dict() == {"P1": 22, "P2": 10, "P3": 40}  # no duplicates


def test_a_run_can_be_told_to_stop_and_resumed_later(tiny, tmp_path):
    proteins = pd.DataFrame({"uniprot_id": ["P1", "P2", "P3"], "sequence": SEQUENCES})
    answers = iter([True, False])  # yes before the first chunk, no before the second
    assert lm.read_to_parquet(proteins, *tiny, tmp_path, chunk_size=1, keep_going=lambda: next(answers)) == 1
    assert lm.read_to_parquet(proteins, *tiny, tmp_path, chunk_size=1, log=lambda m: None) == 2
    assert sorted(pd.read_parquet(tmp_path / "proteins")["uniprot_id"]) == ["P1", "P2", "P3"]


def test_vectors_are_saved_when_asked(tiny, tmp_path):
    proteins = pd.DataFrame({"uniprot_id": ["P1", "P2"], "sequence": SEQUENCES[:2]})
    lm.read_to_parquet(proteins, *tiny, tmp_path, embeddings=True, log=lambda m: None)
    vectors = pd.read_parquet(tmp_path / "embeddings")
    assert vectors.columns.tolist() == ["uniprot_id", "position", "embedding"]
    assert len(vectors) == len(SEQUENCES[0]) + len(SEQUENCES[1]) and len(vectors["embedding"].iloc[0]) == 16
    assert pd.read_parquet(tmp_path / "proteins")["embedded"].all()


def test_proteins_read_from_an_empty_or_missing_folder_is_nothing(tmp_path):
    assert lm.proteins_read(tmp_path / "missing") == set()
    (tmp_path / "proteins").mkdir()
    assert lm.proteins_read(tmp_path) == set()
