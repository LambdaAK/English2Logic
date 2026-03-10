"""Unit tests for Phase 1: Logic foundation."""

import pytest

from logic.ast import And, Iff, Implies, Not, Or, Var, serialize
from logic.parser import parse
from logic.truth_table import evaluate, logically_equivalent, truth_table


# --- AST & Serialization ---


def test_var_serialization():
    assert Var("A").serialize() == "A"
    assert Var("E").serialize() == "E"


def test_var_invalid():
    with pytest.raises(ValueError, match="Invalid variable"):
        Var("X")


def test_not_serialization():
    assert Not(Var("A")).serialize() == "NOT(A)"


def test_and_serialization():
    assert And(Var("A"), Var("B")).serialize() == "AND(A,B)"


def test_or_serialization():
    assert Or(Var("A"), Var("B")).serialize() == "OR(A,B)"


def test_implies_serialization():
    assert Implies(Var("A"), Var("B")).serialize() == "IMPLIES(A,B)"


def test_iff_serialization():
    assert Iff(Var("A"), Var("B")).serialize() == "IFF(A,B)"


def test_nested_serialization():
    formula = Implies(And(Var("A"), Not(Var("B"))), Var("C"))
    assert serialize(formula) == "IMPLIES(AND(A,NOT(B)),C)"


# --- Parser ---


def test_parse_var():
    assert parse("A") == Var("A")
    assert parse("B") == Var("B")


def test_parse_not():
    assert parse("NOT(A)") == Not(Var("A"))


def test_parse_and():
    assert parse("AND(A,B)") == And(Var("A"), Var("B"))


def test_parse_or():
    assert parse("OR(A,B)") == Or(Var("A"), Var("B"))


def test_parse_implies():
    assert parse("IMPLIES(A,B)") == Implies(Var("A"), Var("B"))


def test_parse_iff():
    assert parse("IFF(A,B)") == Iff(Var("A"), Var("B"))


def test_parse_nested():
    result = parse("IMPLIES(AND(A,NOT(B)),C)")
    expected = Implies(And(Var("A"), Not(Var("B"))), Var("C"))
    assert result == expected


def test_parse_roundtrip():
    """Parse IMPLIES(AND(A,NOT(B)),C), serialize back, get identical string."""
    s = "IMPLIES(AND(A,NOT(B)),C)"
    formula = parse(s)
    assert serialize(formula) == s


def test_parse_complex():
    s = "IFF(NOT(A),OR(B,C))"
    formula = parse(s)
    assert serialize(formula) == s


def test_parse_deep():
    s = "IMPLIES(AND(A,NOT(B)),OR(C,D))"
    formula = parse(s)
    assert serialize(formula) == s


def test_parse_empty_raises():
    with pytest.raises(ValueError, match="Empty"):
        parse("")


def test_parse_invalid_raises():
    with pytest.raises(ValueError):
        parse("INVALID(A)")


# --- Truth Table ---


def test_evaluate_var():
    assert evaluate(Var("A"), {"A": True}) is True
    assert evaluate(Var("A"), {"A": False}) is False


def test_evaluate_not():
    assert evaluate(Not(Var("A")), {"A": True}) is False
    assert evaluate(Not(Var("A")), {"A": False}) is True


def test_evaluate_and():
    assert evaluate(And(Var("A"), Var("B")), {"A": True, "B": True}) is True
    assert evaluate(And(Var("A"), Var("B")), {"A": True, "B": False}) is False


def test_evaluate_or():
    assert evaluate(Or(Var("A"), Var("B")), {"A": False, "B": False}) is False
    assert evaluate(Or(Var("A"), Var("B")), {"A": True, "B": False}) is True


def test_evaluate_implies():
    # A -> B is false only when A=True, B=False
    assert evaluate(Implies(Var("A"), Var("B")), {"A": True, "B": False}) is False
    assert evaluate(Implies(Var("A"), Var("B")), {"A": False, "B": False}) is True


def test_truth_table_size():
    formula = And(Var("A"), Var("B"))
    table = truth_table(formula)
    assert len(table) == 32  # 2^5 assignments


def test_implies_equiv_or_not():
    """IMPLIES(A,B) is logically equivalent to OR(NOT(A),B)."""
    f1 = parse("IMPLIES(A,B)")
    f2 = parse("OR(NOT(A),B)")
    assert logically_equivalent(f1, f2)


def test_iff_equiv_both_directions():
    """IFF(A,B) is equivalent to AND(IMPLIES(A,B), IMPLIES(B,A))."""
    f1 = parse("IFF(A,B)")
    f2 = parse("AND(IMPLIES(A,B),IMPLIES(B,A))")
    assert logically_equivalent(f1, f2)
