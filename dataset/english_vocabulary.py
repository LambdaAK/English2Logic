"""English vocabulary for the propositional logic domain.

Tokens are derived from the realizer patterns. Tokenization uses
lowercase and splits on whitespace (punctuation removed).
"""

# Special tokens
PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"

# All tokens that can appear in realizer output (after lowercasing)
# Derived from: Var, NOT, AND, OR, IMPLIES, IFF patterns
ENGLISH_TOKENS = frozenset({
    # Variables (lowercase)
    "a", "b", "c", "d", "e",
    # Var patterns
    "is", "true", "holds", "hold",
    # NOT patterns
    "not", "false", "it", "the", "case", "that", "does",
    # AND patterns
    "and", "both", "together", "with",
    # OR patterns
    "or", "either", "at", "least", "one", "of",
    # IMPLIES patterns
    "if", "then", "implies", "whenever", "follows", "from", "provided",
    # IFF patterns
    "iff", "only", "when", "equivalent", "exactly", "to",
})

# Full vocabulary including special tokens
VOCABULARY = frozenset(ENGLISH_TOKENS) | {PAD, SOS, EOS, UNK}


def tokenize_for_vocab(text: str) -> set[str]:
    """Tokenize text (lowercase, split on whitespace, strip punctuation)."""
    import re
    text = text.lower()
    # Remove punctuation, split on whitespace
    tokens = re.sub(r"[^\w\s]", " ", text).split()
    return set(tokens)


def extract_vocabulary_from_texts(texts: list[str]) -> set[str]:
    """Extract all unique tokens from a list of texts."""
    vocab = set()
    for text in texts:
        vocab.update(tokenize_for_vocab(text))
    return vocab


def all_tokens_in_vocabulary(texts: list[str]) -> bool:
    """Check that every token in texts appears in VOCABULARY (excluding special)."""
    extracted = extract_vocabulary_from_texts(texts)
    return extracted <= ENGLISH_TOKENS
