"""
Explicit, concrete lists of allowed tokens for the propositional logic domain.

These are the ONLY tokens the model may see. Any token not in these lists
is out-of-vocabulary (OOV) and must be mapped to <unk>.
"""

# =============================================================================
# SPECIAL TOKENS (all vocabularies)
# =============================================================================

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"

SPECIAL_TOKENS = (PAD, SOS, EOS, UNK)


# =============================================================================
# ENGLISH TOKENS
# =============================================================================
# Complete list of allowed English tokens (after lowercasing, punctuation removed).
# Derived from the realizer paraphrase patterns. No other tokens are permitted.
# =============================================================================

ENGLISH_CONTENT_TOKENS = (
    # Variables (lowercase, as they appear after tokenization)
    "a",
    "b",
    "c",
    "d",
    "e",
    # Var patterns: is, true, holds, hold
    "is",
    "true",
    "holds",
    "hold",
    # NOT patterns: not, false, it, the, case, that, does
    "not",
    "false",
    "it",
    "the",
    "case",
    "that",
    "does",
    # AND patterns: and, both, together, with
    "and",
    "both",
    "together",
    "with",
    # OR patterns: or, either, at, least, one, of
    "or",
    "either",
    "at",
    "least",
    "one",
    "of",
    # IMPLIES patterns: if, then, implies, whenever, follows, from, provided
    "if",
    "then",
    "implies",
    "whenever",
    "follows",
    "from",
    "provided",
    # IFF patterns: iff, only, when, equivalent, exactly, to
    "iff",
    "only",
    "when",
    "equivalent",
    "exactly",
    "to",
)

# Full English vocabulary: special tokens first, then content (sorted for reproducibility)
ENGLISH_TOKENS = SPECIAL_TOKENS + tuple(sorted(ENGLISH_CONTENT_TOKENS))

# Set for OOV checking
ENGLISH_TOKEN_SET = frozenset(ENGLISH_TOKENS)


# =============================================================================
# LOGIC TOKENS
# =============================================================================
# Complete list of allowed logic tokens for the canonical format.
# Format: IMPLIES(AND(A,NOT(B)),C)
# No other tokens are permitted.
# =============================================================================

# Operators (order matters for tokenizer: longest first to avoid "I" matching "IFF")
LOGIC_OPERATORS = (
    "IMPLIES",
    "IFF",
    "AND",
    "OR",
    "NOT",
)

# Variables
LOGIC_VARIABLES = ("A", "B", "C", "D", "E")

# Syntax
LOGIC_LEFT_PAREN = "("
LOGIC_RIGHT_PAREN = ")"
LOGIC_COMMA = ","

LOGIC_SYNTAX_TOKENS = (LOGIC_LEFT_PAREN, LOGIC_RIGHT_PAREN, LOGIC_COMMA)

# Full logic vocabulary: special tokens first, then operators, variables, syntax
LOGIC_CONTENT_TOKENS = (
    LOGIC_OPERATORS
    + LOGIC_VARIABLES
    + LOGIC_SYNTAX_TOKENS
)

LOGIC_TOKENS = SPECIAL_TOKENS + LOGIC_CONTENT_TOKENS

# Set for OOV checking
LOGIC_TOKEN_SET = frozenset(LOGIC_TOKENS)


def print_all_tokens() -> None:
    """Print the complete explicit token lists (for reference)."""
    print("=" * 60)
    print("ENGLISH TOKENS (explicit allowed list)")
    print("=" * 60)
    print("Special:", list(SPECIAL_TOKENS))
    print("Content:", list(sorted(ENGLISH_CONTENT_TOKENS)))
    print("Total:", len(ENGLISH_TOKENS))
    print()
    print("=" * 60)
    print("LOGIC TOKENS (explicit allowed list)")
    print("=" * 60)
    print("Special:", list(SPECIAL_TOKENS))
    print("Operators:", list(LOGIC_OPERATORS))
    print("Variables:", list(LOGIC_VARIABLES))
    print("Syntax:", list(LOGIC_SYNTAX_TOKENS))
    print("Total:", len(LOGIC_TOKENS))


if __name__ == "__main__":
    print_all_tokens()
