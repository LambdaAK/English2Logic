"""Unit tests for Phase 2: Dataset generation."""

import json
import tempfile
from pathlib import Path

import pytest

from dataset.dataset_builder import build_dataset, generate_examples
from dataset.english_realizer import realize, realize_with_seed
from dataset.english_vocabulary import all_tokens_in_vocabulary, ENGLISH_TOKENS
from dataset.formula_generator import generate_formula
from logic.parser import parse
from logic.ast import serialize


# --- Formula Generator ---


def test_generate_formula_depth1():
    rng = __import__("random").Random(42)
    for _ in range(20):
        f = generate_formula(max_depth=1, rng=rng)
        assert f.__class__.__name__ == "Var"


def test_generate_formula_variety():
    """Generate 100 formulas, parse all, confirm valid."""
    rng = __import__("random").Random(123)
    for _ in range(100):
        formula = generate_formula(max_depth=4, rng=rng)
        s = serialize(formula)
        parsed = parse(s)
        assert serialize(parsed) == s


# --- English Realizer ---


def test_realize_var():
    from logic.ast import Var
    assert realize(Var("A"), __import__("random").Random(0)) in ["A", "A is true", "A holds"]


def test_realize_reproducible():
    from logic.ast import Implies, Var
    formula = Implies(Var("A"), Var("B"))
    r1 = realize_with_seed(formula, 42)
    r2 = realize_with_seed(formula, 42)
    assert r1 == r2


def test_realize_nested():
    from logic.ast import And, Implies, Not, Var
    formula = Implies(And(Var("A"), Not(Var("B"))), Var("C"))
    english = realize(formula, __import__("random").Random(999))
    assert "A" in english or "a" in english
    assert "B" in english or "b" in english
    assert "C" in english or "c" in english


# --- Vocabulary ---


def test_vocabulary_covers_realizer_output():
    """Generate 1000 examples, confirm every token in vocabulary."""
    examples = list(generate_examples(1000, seed=456))
    texts = [ex["input"] for ex in examples]
    assert all_tokens_in_vocabulary(texts), "Some tokens not in vocabulary"


# --- Dataset Builder ---


def test_dataset_builder_output_format():
    examples = list(generate_examples(10, seed=1))
    for ex in examples:
        assert "input" in ex
        assert "target" in ex
        assert isinstance(ex["input"], str)
        assert isinstance(ex["target"], str)


def test_dataset_builder_parse_all():
    """All targets must parse to valid formulas."""
    examples = list(generate_examples(100, seed=2))
    for ex in examples:
        parsed = parse(ex["target"])
        assert serialize(parsed) == ex["target"]


def test_dataset_builder_saves_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        build_dataset(
            Path(tmpdir),
            train_size=100,
            val_size=20,
            test_size=20,
            seed=3,
        )
        total = 0
        for name in ("train.json", "val.json", "test.json"):
            with open(Path(tmpdir) / name) as f:
                data = json.load(f)
            total += len(data)
            for ex in data:
                assert "input" in ex and "target" in ex
        assert total >= 100


def test_dataset_builder_deduplicates():
    """build_dataset produces no duplicate (input, target) pairs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        build_dataset(
            Path(tmpdir),
            train_size=200,
            val_size=50,
            test_size=50,
            seed=7,
        )
        for name in ("train.json", "val.json", "test.json"):
            with open(Path(tmpdir) / name) as f:
                data = json.load(f)
            seen = set()
            for ex in data:
                key = (ex["input"], ex["target"])
                assert key not in seen, f"Duplicate in {name}: {key}"
                seen.add(key)
