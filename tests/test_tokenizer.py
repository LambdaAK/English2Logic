"""Unit tests for Phase 3: Tokenization."""

import json
from pathlib import Path

import pytest

from logic.parser import parse
from logic.ast import serialize
from model.tokenizer import (
    check_english_oov,
    check_logic_oov,
    detokenize_logic,
    tokenize_english,
    tokenize_logic,
    get_english_vocab,
    get_logic_vocab,
    english_tokens_to_ids,
    logic_tokens_to_ids,
    ids_to_english_tokens,
    ids_to_logic_tokens,
)
from model.tokens import ENGLISH_TOKENS, LOGIC_TOKENS


# --- Token lists ---


def test_english_tokens_explicit():
    """English tokens are a concrete, explicit list."""
    assert "<pad>" in ENGLISH_TOKENS
    assert "<sos>" in ENGLISH_TOKENS
    assert "<eos>" in ENGLISH_TOKENS
    assert "<unk>" in ENGLISH_TOKENS
    assert "a" in ENGLISH_TOKENS
    assert "if" in ENGLISH_TOKENS
    assert "then" in ENGLISH_TOKENS
    assert len(ENGLISH_TOKENS) > 40


def test_logic_tokens_explicit():
    """Logic tokens are a concrete, explicit list."""
    assert "IMPLIES" in LOGIC_TOKENS
    assert "AND" in LOGIC_TOKENS
    assert "NOT" in LOGIC_TOKENS
    assert "A" in LOGIC_TOKENS
    assert "(" in LOGIC_TOKENS
    assert ")" in LOGIC_TOKENS
    assert "," in LOGIC_TOKENS


# --- English tokenizer ---


def test_tokenize_english():
    tokens = tokenize_english("If A then B")
    assert tokens == ["if", "a", "then", "b"]


def test_tokenize_english_lowercase():
    tokens = tokenize_english("A AND B")
    assert tokens == ["a", "and", "b"]


def test_tokenize_english_no_oov():
    """Tokenize 50 examples, confirm no OOV tokens."""
    data_path = Path("data/train.json")
    if not data_path.exists():
        pytest.skip("data/train.json not found")
    with open(data_path) as f:
        data = json.load(f)
    for i, ex in enumerate(data[:50]):
        tokens = tokenize_english(ex["input"])
        oov = check_english_oov(tokens)
        assert not oov, f"Example {i} has OOV: {oov}"


def test_english_ids_roundtrip():
    text = "if A then B"
    tokens = tokenize_english(text)
    ids = english_tokens_to_ids(tokens)
    back = ids_to_english_tokens(ids)
    assert back == tokens


# --- Logic tokenizer ---


def test_tokenize_logic_simple():
    tokens = tokenize_logic("IMPLIES(A,B)")
    assert tokens == ["IMPLIES", "(", "A", ",", "B", ")"]


def test_tokenize_logic_nested():
    tokens = tokenize_logic("IMPLIES(AND(A,NOT(B)),C)")
    expected = ["IMPLIES", "(", "AND", "(", "A", ",", "NOT", "(", "B", ")", ")", ",", "C", ")"]
    assert tokens == expected


def test_logic_roundtrip_parse():
    """Tokenize, detokenize, parse - round-trip."""
    formula_str = "IMPLIES(AND(A,NOT(B)),C)"
    tokens = tokenize_logic(formula_str)
    back = detokenize_logic(tokens)
    assert back == formula_str
    parsed = parse(back)
    assert serialize(parsed) == formula_str


def test_logic_no_oov():
    """Tokenize 50 examples, confirm no OOV tokens."""
    data_path = Path("data/train.json")
    if not data_path.exists():
        pytest.skip("data/train.json not found")
    with open(data_path) as f:
        data = json.load(f)
    for i, ex in enumerate(data[:50]):
        tokens = tokenize_logic(ex["target"])
        oov = check_logic_oov(tokens)
        assert not oov, f"Example {i} has OOV: {oov}"


def test_logic_ids_roundtrip():
    formula_str = "OR(A,B)"
    tokens = tokenize_logic(formula_str)
    ids = logic_tokens_to_ids(tokens)
    back = ids_to_logic_tokens(ids)
    assert back == tokens
    assert detokenize_logic(back) == formula_str


def test_logic_full_roundtrip_50():
    """50 examples: tokenize -> detokenize -> parse -> serialize."""
    data_path = Path("data/train.json")
    if not data_path.exists():
        pytest.skip("data/train.json not found")
    with open(data_path) as f:
        data = json.load(f)
    for i, ex in enumerate(data[:50]):
        target = ex["target"]
        tokens = tokenize_logic(target)
        back = detokenize_logic(tokens)
        parsed = parse(back)
        assert serialize(parsed) == target, f"Example {i} failed"
