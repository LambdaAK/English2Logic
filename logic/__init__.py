"""Propositional logic module: AST, parser, and truth tables."""

from logic.ast import (
    And,
    Formula,
    Iff,
    Implies,
    Not,
    Or,
    Var,
    VARIABLES,
    serialize,
)
from logic.parser import parse
from logic.truth_table import evaluate, logically_equivalent, truth_table

__all__ = [
    "And",
    "Formula",
    "Iff",
    "Implies",
    "Not",
    "Or",
    "Var",
    "VARIABLES",
    "evaluate",
    "logically_equivalent",
    "parse",
    "serialize",
    "truth_table",
]
