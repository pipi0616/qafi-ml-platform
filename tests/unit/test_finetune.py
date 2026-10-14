import numpy as np
import pandas as pd
import pytest
import torch

from qafi_core import baseline, finetune
from qafi_core.features import AMINO_ACIDS, FEATURES

VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", *"LAGVSERTIDPKQNFYMHWCXBUZO", ".", "-", "<null_1>", "<mask>"]
FAST = {"epochs": 2, "head_learning_rate": 3e-3, "lora_learning_rate": 3e-3}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A tiny ESM-2 with random weights (07e's test model) and its tokenizer: nothing to download."""
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


@pytest.fixture(scope="module")
def world():
    """Ten proteins of 300 residues; each dataset measured a stretch of them, one dataset a long stretch."""
    rng = np.random.default_rng(0)
    sequences, rows = {}, []
    for p in range(10):
        sequences[f"P{p}"] = "".join(rng.choice(list(AMINO_ACIDS), 300))
        first, last = (20, 290) if p == 0 else (100, 160)  # P0's stretch needs two crops
        for position in range(first, last + 1):
            for alt in "CW":
                ref = sequences[f"P{p}"][position - 1]
                kind = ["domainome", "mavedb"][p % 2]
                where = {
                    "uniprot_id": f"P{p}",
                    "similarity_group": f"G{p // 2}",
                    "uniprot_position": position,
                }
                variant = {"alt_aa": alt, "ref_aa": ref, "label": rng.normal()}
                rows.append({"dataset_id": f"d{p}", "data_kind": kind, **where, **variant})
    df = pd.DataFrame(rows)
    for name in FEATURES:
        df[name] = 0.0
    return df, pd.Series(sequences)


def test_a_short_stretch_is_centred_in_one_crop_and_a_long_one_gets_overlapping_crops():
    assert finetune.windows(100, 160, 300) == [3]  # 254 residues centred on 130: residues 3 to 256
    assert finetune.windows(1, 30, 300) == [1] and finetune.windows(280, 300, 300) == [47]
    starts = finetune.windows(20, 290, 300)
    assert len(starts) > 1 and starts[0] <= 20 and starts[-1] + 253 >= 290


def test_every_variant_lands_on_exactly_one_grid_cell(world, tiny):
    df, sequences = world
    _, tokenizer = tiny
    items = finetune.crops(df, sequences, tokenizer)
    cells = np.concatenate([c["cells"] for c in items])
    assert sorted(cells[:, 0].tolist()) == list(range(len(df)))  # each row once
    assert len([c for c in items if c["dataset"] == "d0"]) >= 2  # the long stretch needed several crops
    crop = next(c for c in items if c["dataset"] == "d3")
    assert crop["tokens"].shape == (finetune.LENGTH,) and crop["tokens"][0] == VOCAB.index("<cls>")
    row, token, alt = crop["cells"][0]
    assert crop["weight"][token, alt] > 0 and crop["target"][token, alt] == pytest.approx(
        float(finetune.standardised(df)[row])
    )
    assert int((crop["weight"] > 0).sum()) == len(crop["cells"])  # nothing else carries weight
    letter = AMINO_ACIDS[int(crop["ref"][token])]
    assert letter == df.loc[row, "ref_aa"]  # the token really is that residue


def test_batches_always_have_the_same_shape(world, tiny):
    df, sequences = world
    items = finetune.crops(df, sequences, tiny[1])
    batches = list(finetune._batches(items, shuffle=False))
    assert all(b["tokens"].shape == (finetune.CROPS_PER_BATCH, finetune.LENGTH) for _, b in batches)
    real, last = batches[-1]
    assert last["weight"][len(real) :].sum() == 0  # the filler crops teach nothing


def test_zero_shot_is_the_log_probability_difference_of_the_two_amino_acids(world, tiny):
    df, sequences = world
    model, tokenizer = tiny
    items = finetune.crops(df, sequences, tokenizer)
    scores = finetune.zero_shot(items, model, torch.device("cpu"))
    crop = items[0]
    row, token, alt = crop["cells"][0]
    with torch.no_grad():
        logits = model(input_ids=crop["tokens"][None], attention_mask=crop["mask"][None]).logits[0, token]
    ids = [VOCAB.index(a) for a in AMINO_ACIDS]
    logp = torch.log_softmax(logits[ids], dim=-1)
    assert scores[row] == pytest.approx(float(logp[alt] - logp[int(crop["ref"][token])]), abs=1e-5)


def test_frozen_and_lora_training_predict_every_variant_and_never_change_the_original(world, tiny):
    df, sequences = world
    model, tokenizer = tiny
    before = {name: value.clone() for name, value in model.state_dict().items()}
    items = finetune.crops(df, sequences, tokenizer)
    for how in ("frozen", "lora"):
        cv = finetune.cross_validate(df, items, model, how, settings=FAST)
        assert cv["predicted"].notna().all() and (cv.groupby("similarity_group")["fold"].nunique() == 1).all()
        assert len(baseline.dataset_scores(cv)) == df["dataset_id"].nunique()
    assert all(torch.equal(before[name], value.cpu()) for name, value in model.state_dict().items())


def test_lora_trains_a_small_share_of_the_weights(tiny):
    model, _ = tiny
    lora = finetune._fresh(model, finetune.SETTINGS, lora=True, dev=torch.device("cpu"))
    frozen = finetune._fresh(model, finetune.SETTINGS, lora=False, dev=torch.device("cpu"))
    trainable = lambda m: sum(p.numel() for p in m.esm.parameters() if p.requires_grad)  # noqa: E731
    assert trainable(frozen) == 0 and 0 < trainable(lora) < sum(p.numel() for p in lora.esm.parameters())
    assert any("lora" in name for name, _ in lora.esm.named_parameters())


def test_each_run_records_what_it_used():
    assert finetune.parameters("zero_shot")["revision"] == finetune.REVISION
    assert "epochs" not in finetune.parameters("zero_shot")
    frozen, lora = finetune.parameters("frozen"), finetune.parameters("lora")
    assert frozen["epochs"] == 4 and "lora_rank" not in frozen
    assert lora["epochs"] == 5 and lora["lora_rank"] == 8 and lora["crop_length"] == 256
