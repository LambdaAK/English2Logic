"""English realization of propositional logic formulas via templates."""

from __future__ import annotations

import random
from typing import Optional

from logic.ast import And, Formula, Iff, Implies, Not, Or, Var

VAR_PATTERNS = [
    "{var}",
    "{var} is true",
    "{var} holds",
]

NOT_PATTERNS = [
    "not {0}",
    "{0} is false",
    "it is not the case that {0}",
    "{0} does not hold",
]

AND_PATTERNS = [
    "{0} and {1}",
    "both {0} and {1}",
    "{0} together with {1}",
]

OR_PATTERNS = [
    "{0} or {1}",
    "either {0} or {1}",
    "at least one of {0} or {1}",
]

IMPLIES_PATTERNS = [
    "if {0} then {1}",
    "{1} if {0}",
    "{0} implies {1}",
    "whenever {0}, {1}",
    "{1} follows from {0}",
    "provided that {0}, {1}",
]

IFF_PATTERNS = [
    "{0} iff {1}",
    "{0} if and only if {1}",
    "{0} exactly when {1}",
    "{0} is equivalent to {1}",
]


def realize(
    formula: Formula,
    rng: Optional[random.Random] = None,
) -> str:
    """Convert a formula to a natural language string using random templates."""
    rng = rng or random.Random()
    return _realize(formula, rng)


def realize_with_seed(formula: Formula, seed: int) -> str:
    """Realize with a fixed seed for reproducibility."""
    return realize(formula, random.Random(seed))


def _realize(formula: Formula, rng: random.Random) -> str:
    if isinstance(formula, Var):
        return rng.choice(VAR_PATTERNS).format(var=formula.name)
    elif isinstance(formula, Not):
        sub = _realize(formula.operand, rng)
        return rng.choice(NOT_PATTERNS).format(sub)
    elif isinstance(formula, And):
        left = _realize(formula.left, rng)
        right = _realize(formula.right, rng)
        return rng.choice(AND_PATTERNS).format(left, right)
    elif isinstance(formula, Or):
        left = _realize(formula.left, rng)
        right = _realize(formula.right, rng)
        return rng.choice(OR_PATTERNS).format(left, right)
    elif isinstance(formula, Implies):
        ant = _realize(formula.antecedent, rng)
        cons = _realize(formula.consequent, rng)
        return rng.choice(IMPLIES_PATTERNS).format(ant, cons)
    elif isinstance(formula, Iff):
        left = _realize(formula.left, rng)
        right = _realize(formula.right, rng)
        return rng.choice(IFF_PATTERNS).format(left, right)
    else:
        raise TypeError(f"Unknown formula type: {type(formula)}")
