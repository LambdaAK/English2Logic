"""Tokenizers for English input and logic output."""

from __future__ import annotations

import re
from typing import Optional

from model.tokens import (
    ENGLISH_TOKEN_SET,
    ENGLISH_TOKENS,
    LOGIC_OPERATORS,
    LOGIC_TOKEN_SET,
    LOGIC_TOKENS,
    LOGIC_VARIABLES,
    UNK,
)


# =============================================================================
# English Tokenizer
# =============================================================================


def tokenize_english(text: str) -> list[str]:
    """Tokenize English text: lowercase, remove punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return tokens


def english_tokens_to_ids(
    tokens: list[str],
    vocab: Optional[dict[str, int]] = None,
) -> list[int]:
    """Convert tokens to IDs. OOV tokens map to UNK."""
    if vocab is None:
        vocab = get_english_vocab()
    return [vocab.get(t, vocab[UNK]) for t in tokens]


def ids_to_english_tokens(ids: list[int], vocab: Optional[dict[int, str]] = None) -> list[str]:
    """Convert IDs back to tokens."""
    if vocab is None:
        vocab = get_english_id_to_token()
    return [vocab.get(i, UNK) for i in ids]


def get_english_vocab() -> dict[str, int]:
    """Token -> ID mapping. Stable order: special first, then content alphabetically."""
    return {t: i for i, t in enumerate(ENGLISH_TOKENS)}


def get_english_id_to_token() -> dict[int, str]:
    """ID -> token mapping."""
    return {i: t for i, t in enumerate(ENGLISH_TOKENS)}


def english_vocab_size() -> int:
    return len(ENGLISH_TOKENS)


def check_english_oov(tokens: list[str]) -> set[str]:
    """Return set of tokens that are OOV."""
    return set(tokens) - ENGLISH_TOKEN_SET


# =============================================================================
# Logic Tokenizer
# =============================================================================


def tokenize_logic(formula_str: str) -> list[str]:
    """
    Tokenize canonical logic format: IMPLIES(AND(A,NOT(B)),C)

    Returns tokens: ["IMPLIES", "(", "AND", "(", "A", ",", "NOT", "(", "B", ")", ")", ",", "C", ")"]
    """
    tokens = []
    i = 0
    s = formula_str.strip()

    while i < len(s):
        # Skip whitespace
        if s[i].isspace():
            i += 1
            continue

        # Check for operator (longest first)
        matched = False
        for op in LOGIC_OPERATORS:
            if s[i : i + len(op)] == op and (
                i + len(op) >= len(s) or s[i + len(op)] in "(),"
            ):
                tokens.append(op)
                i += len(op)
                matched = True
                break
        if matched:
            continue

        # Check for variable
        if s[i] in LOGIC_VARIABLES and (i + 1 >= len(s) or s[i + 1] in "(),"):
            tokens.append(s[i])
            i += 1
            continue

        # Check for syntax
        if s[i] == "(":
            tokens.append("(")
            i += 1
            continue
        if s[i] == ")":
            tokens.append(")")
            i += 1
            continue
        if s[i] == ",":
            tokens.append(",")
            i += 1
            continue

        # Unknown character - skip or raise
        i += 1

    return tokens


def detokenize_logic(tokens: list[str]) -> str:
    """Convert logic tokens back to canonical string (no spaces)."""
    return "".join(tokens)


def logic_tokens_to_ids(
    tokens: list[str],
    vocab: Optional[dict[str, int]] = None,
) -> list[int]:
    """Convert tokens to IDs. OOV tokens map to UNK."""
    if vocab is None:
        vocab = get_logic_vocab()
    return [vocab.get(t, vocab[UNK]) for t in tokens]


def ids_to_logic_tokens(ids: list[int], vocab: Optional[dict[int, str]] = None) -> list[str]:
    """Convert IDs back to tokens."""
    if vocab is None:
        vocab = get_logic_id_to_token()
    return [vocab.get(i, UNK) for i in ids]


def get_logic_vocab() -> dict[str, int]:
    """Token -> ID mapping."""
    return {t: i for i, t in enumerate(LOGIC_TOKENS)}


def get_logic_id_to_token() -> dict[int, str]:
    """ID -> token mapping."""
    return {i: t for i, t in enumerate(LOGIC_TOKENS)}


def logic_vocab_size() -> int:
    return len(LOGIC_TOKENS)


def check_logic_oov(tokens: list[str]) -> set[str]:
    """Return set of tokens that are OOV."""
    return set(tokens) - LOGIC_TOKEN_SET
