"""AST nodes for propositional logic formulas."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

# Allowed variables
VARIABLES = frozenset({"A", "B", "C", "D", "E"})


@dataclass(frozen=True)
class Var:
    """Atomic proposition (variable)."""
    name: str

    def __post_init__(self) -> None:
        if self.name not in VARIABLES:
            raise ValueError(f"Invalid variable: {self.name}. Must be one of {VARIABLES}")

    def serialize(self) -> str:
        return self.name


@dataclass(frozen=True)
class Not:
    """Negation: NOT(x)."""
    operand: "Formula"

    def serialize(self) -> str:
        return f"NOT({self.operand.serialize()})"


@dataclass(frozen=True)
class And:
    """Conjunction: AND(x, y)."""
    left: "Formula"
    right: "Formula"

    def serialize(self) -> str:
        return f"AND({self.left.serialize()},{self.right.serialize()})"


@dataclass(frozen=True)
class Or:
    """Disjunction: OR(x, y)."""
    left: "Formula"
    right: "Formula"

    def serialize(self) -> str:
        return f"OR({self.left.serialize()},{self.right.serialize()})"


@dataclass(frozen=True)
class Implies:
    """Implication: IMPLIES(x, y)."""
    antecedent: "Formula"
    consequent: "Formula"

    def serialize(self) -> str:
        return f"IMPLIES({self.antecedent.serialize()},{self.consequent.serialize()})"


@dataclass(frozen=True)
class Iff:
    """Biconditional: IFF(x, y)."""
    left: "Formula"
    right: "Formula"

    def serialize(self) -> str:
        return f"IFF({self.left.serialize()},{self.right.serialize()})"


Formula = Union[Var, Not, And, Or, Implies, Iff]


def serialize(formula: Formula) -> str:
    """Serialize a formula to canonical string format."""
    return formula.serialize()
