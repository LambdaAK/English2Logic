"""Unit tests for Phase 4: Model and training."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from model.transformer import Seq2SeqTransformer
from model.train import (
    LogicDataset,
    collate_batch,
    exact_match_accuracy,
    run_overfit_test,
    train_epoch,
)
from model.tokenizer import get_english_vocab, get_logic_vocab
from model.tokens import PAD, SOS, EOS


def test_transformer_forward():
    """Forward pass runs, output shape correct."""
    model = Seq2SeqTransformer.create_default()
    eng_vocab = get_english_vocab()
    logic_vocab = get_logic_vocab()
    pad_idx = eng_vocab[PAD]

    batch_size = 4
    src_len = 10
    tgt_len = 12
    src = torch.randint(1, len(eng_vocab), (batch_size, src_len))
    src[:, -3:] = pad_idx  # Some padding
    tgt = torch.randint(1, len(logic_vocab), (batch_size, tgt_len))
    tgt[:, -2:] = pad_idx

    src_key_padding_mask = src == pad_idx
    tgt_key_padding_mask = tgt == pad_idx

    logits = model(src, tgt, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
    assert logits.shape == (batch_size, tgt_len, len(logic_vocab))


def test_dataset_and_collate():
    """Dataset and collate produce valid batches."""
    examples = [
        {"input": "if A then B", "target": "IMPLIES(A,B)"},
        {"input": "A and B", "target": "AND(A,B)"},
    ]
    dataset = LogicDataset(examples)
    eng_vocab = get_english_vocab()
    logic_vocab = get_logic_vocab()

    batch = [dataset[i] for i in range(len(dataset))]
    src, src_mask, tgt_in, tgt_out, tgt_mask = collate_batch(
        batch,
        pad_idx=logic_vocab[PAD],
        sos_idx=logic_vocab[SOS],
        eos_idx=logic_vocab[EOS],
    )
    assert src.shape[0] == 2
    assert tgt_in.shape[0] == 2
    assert tgt_in[0, 0] == logic_vocab[SOS]
    assert tgt_out.shape == tgt_in.shape


@pytest.mark.slow
def test_overfit_test():
    """Model can overfit on small data (>95% accuracy)."""
    data_path = Path("data/train.json")
    if not data_path.exists():
        pytest.skip("data/train.json not found")
    acc = run_overfit_test(
        data_path=data_path,
        n_examples=200,
        n_epochs=150,
        batch_size=16,
        lr=5e-4,
        seed=42,
    )
    assert acc >= 0.95, f"Overfit test failed: {acc:.2%}"
