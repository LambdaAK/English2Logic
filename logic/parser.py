"""Parser for canonical propositional logic format."""

from __future__ import annotations

from logic.ast import And, Formula, Iff, Implies, Not, Or, Var


def parse(s: str) -> Formula:
    """Parse a canonical formula string into an AST.

    Format: IMPLIES(AND(A,NOT(B)),C)
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty string")

    formula, remainder = _parse_formula(s)
    if remainder.strip():
        raise ValueError(f"Unexpected trailing input: {remainder!r}")
    return formula


def _parse_formula(s: str) -> tuple[Formula, str]:
    """Parse a formula from the start of s. Return (formula, remainder)."""
    s = s.lstrip()

    # Variable: single letter A-E
    if len(s) >= 1 and s[0] in "ABCDE" and (len(s) == 1 or s[1] in "(,)"):
        return Var(s[0]), s[1:].lstrip()

    # Operator: NOT, AND, OR, IMPLIES, IFF
    for op_name, op_class in [
        ("NOT", _parse_not),
        ("AND", _parse_binary),
        ("OR", _parse_binary),
        ("IMPLIES", _parse_binary),
        ("IFF", _parse_binary),
    ]:
        if s.startswith(op_name):
            remainder = s[len(op_name) :].lstrip()
            if not remainder.startswith("("):
                raise ValueError(f"Expected '(' after {op_name}, got {remainder[:20]!r}")
            if op_name == "NOT":
                return op_class(remainder[1:], Not)
            else:
                return op_class(remainder[1:], op_name)

    raise ValueError(f"Cannot parse formula from: {s[:50]!r}")


def _parse_not(s: str, _op_class: type) -> tuple[Formula, str]:
    """Parse NOT(operand). s is the content after '('."""
    operand, remainder = _parse_formula(s)
    remainder = remainder.lstrip()
    if not remainder.startswith(")"):
        raise ValueError(f"Expected ')' after NOT operand, got {remainder[:20]!r}")
    return Not(operand), remainder[1:].lstrip()


def _parse_binary(s: str, op_name: str) -> tuple[Formula, str]:
    """Parse OP(left, right). s is the content after '('."""
    # Find the comma that separates left and right at depth 0 (we're inside one paren)
    depth = 1
    i = 0
    comma_pos = -1
    while i < len(s) and depth > 0:
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
        elif s[i] == "," and depth == 1:
            comma_pos = i
            break
        i += 1

    if comma_pos < 0:
        raise ValueError(f"Expected ',' in {op_name} arguments, got: {s[:50]!r}")

    left_str = s[:comma_pos].strip()
    right_str = s[comma_pos + 1 :].strip()

    if not left_str or not right_str:
        raise ValueError(f"Empty argument in {op_name}")

    left, left_rem = _parse_formula(left_str)
    if left_rem:
        raise ValueError(f"Unexpected content in left arg of {op_name}: {left_rem!r}")

    # right_str may include trailing ')' from the outer OP( ); that's expected
    right, right_rem = _parse_formula(right_str)

    # Find closing paren - right_str may have trailing content for nested structure
    # Actually right_str is just the second argument. We need to get the remainder
    # after the closing paren. Let me re-read the string.
    # s = "AND(A,NOT(B)),C)" for IMPLIES(AND(A,NOT(B)),C)
    # So we parse AND(A,NOT(B)) - the left part. The right part is C.
    # After parsing right, we have ")" from the original OP( ... ). So we need to
    # consume the closing paren from the original string.
    # The issue: when we call _parse_formula(right_str), right_str could be "C)"
    # if we're not careful. Let me trace through.
    # For IMPLIES(AND(A,NOT(B)),C):
    # s = "AND(A,NOT(B)),C)"
    # comma_pos = 13 (the comma between NOT(B) and C)
    # left_str = "AND(A,NOT(B))"
    # right_str = "C)"
    # So right_str includes the closing paren. When we parse "C)", we get Var("C") and remainder ")"
    # So we're good - we don't need to consume the paren from the outer call.
    # But wait - the remainder after parsing the full IMPLIES should be empty or whatever comes after.
    # The caller of _parse_binary gets s = "AND(A,NOT(B)),C)" - the content after IMPLIES(
    # So we parse left = AND(A,NOT(B)), right = C. The closing ")" of IMPLIES is part of right_str.
    # When we parse "C)", we get Var("C") and ")". So we need to return the ")" to the caller?
    # No - the caller of _parse_binary is _parse_formula when it sees "IMPLIES". So the flow is:
    # parse("IMPLIES(AND(A,NOT(B)),C)")
    # _parse_formula sees IMPLIES, calls _parse_binary with s = "AND(A,NOT(B)),C)"
    # _parse_binary splits at comma, gets left_str="AND(A,NOT(B))", right_str="C)"
    # _parse_formula(left_str) -> And(Var(A), Not(Var(B))), ""
    # _parse_formula(right_str) -> Var(C), ")"  -- the ) is leftover
    # So we need to return the remainder including the ")" so the caller can consume it.
    # Actually the caller of _parse_binary is _parse_formula. After _parse_binary returns,
    # we need to pass the remainder back. So _parse_binary should return the part of s
    # that wasn't consumed. We consumed left_str and right_str. The part after right_str
    # in s is... right_str = s[comma_pos+1:].strip() = "C)". So we consumed "C)" when
    # parsing right. The remainder from parsing right is ")". So the remainder for the
    # whole IMPLIES is ")" - the closing paren. We need to consume that and return
    # whatever is after. So _parse_binary needs to:
    # 1. Parse left from left_str
    # 2. Parse right from right_str - but right_str might be "C)" or "OR(A,B))" etc.
    #    So we parse and get remainder. The remainder should be ")" for the closing of IMPLIES.
    # 3. Return the formula and the remainder (which would be "" if the full string was
    #    "IMPLIES(AND(A,NOT(B)),C)" with nothing after).

    # Actually the issue is: we're splitting s at the first comma at depth 1. So
    # s = "AND(A,NOT(B)),C)"
    # We get left_str = "AND(A,NOT(B))", right_str = "C)"
    # When we parse right_str = "C)", we get Var("C") and remainder ")"
    # So we need to return that remainder from _parse_binary. The remainder is
    # whatever remains after parsing the second argument. So we should use the
    # remainder from parsing right_str. Let me update the code.

    # Consume the closing ')' of OP(left, right)
    remainder = right_rem.lstrip()
    if not remainder.startswith(")"):
        raise ValueError(f"Expected ')' after {op_name} arguments, got {remainder[:20]!r}")
    remainder = remainder[1:].lstrip()

    ops = {
        "AND": And,
        "OR": Or,
        "IMPLIES": Implies,
        "IFF": Iff,
    }
    op_class = ops[op_name]
    formula = op_class(left, right)
    return formula, remainder
