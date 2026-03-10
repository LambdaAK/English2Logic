"""Truth table evaluation for propositional logic formulas."""

from __future__ import annotations

import itertools
from typing import Dict

from logic.ast import And, Formula, Iff, Implies, Not, Or, Var

VARIABLES_ORDERED = ("A", "B", "C", "D", "E")


def evaluate(formula: Formula, assignment: Dict[str, bool]) -> bool:
    """Evaluate a formula under a truth assignment.

    assignment maps variable names (A, B, C, D, E) to True/False.
    """
    if isinstance(formula, Var):
        return assignment.get(formula.name, False)
    elif isinstance(formula, Not):
        return not evaluate(formula.operand, assignment)
    elif isinstance(formula, And):
        return evaluate(formula.left, assignment) and evaluate(formula.right, assignment)
    elif isinstance(formula, Or):
        return evaluate(formula.left, assignment) or evaluate(formula.right, assignment)
    elif isinstance(formula, Implies):
        return (not evaluate(formula.antecedent, assignment)) or evaluate(
            formula.consequent, assignment
        )
    elif isinstance(formula, Iff):
        return evaluate(formula.left, assignment) == evaluate(formula.right, assignment)
    else:
        raise TypeError(f"Unknown formula type: {type(formula)}")


def all_assignments() -> list[Dict[str, bool]]:
    """Generate all 2^5 truth assignments over A, B, C, D, E."""
    for values in itertools.product([False, True], repeat=5):
        yield dict(zip(VARIABLES_ORDERED, values))


def truth_table(formula: Formula) -> list[tuple[Dict[str, bool], bool]]:
    """Compute the full truth table for a formula.

    Returns a list of (assignment, result) pairs.
    """
    return [(a, evaluate(formula, a)) for a in all_assignments()]


def logically_equivalent(f1: Formula, f2: Formula) -> bool:
    """Check if two formulas are logically equivalent."""
    for assignment in all_assignments():
        if evaluate(f1, assignment) != evaluate(f2, assignment):
            return False
    return True
