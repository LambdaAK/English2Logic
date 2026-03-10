"""Random formula generator for propositional logic."""

from __future__ import annotations

import random
from typing import Optional

from logic.ast import And, Formula, Iff, Implies, Not, Or, Var

VARIABLES = list("ABCDE")
BINARY_OPS = [And, Or, Implies, Iff]
UNARY_OPS = [Not]


def generate_formula(
    max_depth: int = 4,
    min_depth: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Formula:
    """Generate a random formula with depth between min_depth and max_depth.

    Depth 1 = single variable.
    Depth 2 = NOT(var) or binary_op(var, var).
    Depth 3+ = nested operators.
    """
    rng = rng or random.Random()
    min_depth = min_depth if min_depth is not None else 1
    depth = rng.randint(min_depth, max_depth)
    return _generate_at_depth(depth, rng)


def _generate_at_depth(depth: int, rng: random.Random) -> Formula:
    if depth <= 1:
        return Var(rng.choice(VARIABLES))

    # Choose operator type: unary (Not) or binary
    if depth == 2:
        # At depth 2, we can do NOT(var) or op(var, var)
        if rng.random() < 0.3:
            return Not(_generate_at_depth(1, rng))
        else:
            op_class = rng.choice(BINARY_OPS)
            return op_class(
                _generate_at_depth(1, rng),
                _generate_at_depth(1, rng),
            )
    else:
        # Depth 3+: pick unary or binary, recurse
        if rng.random() < 0.2:
            return Not(_generate_at_depth(depth - 1, rng))
        else:
            op_class = rng.choice(BINARY_OPS)
            # Split remaining depth between left and right
            left_depth = rng.randint(1, depth - 1)
            right_depth = depth - 1
            return op_class(
                _generate_at_depth(left_depth, rng),
                _generate_at_depth(right_depth, rng),
            )
