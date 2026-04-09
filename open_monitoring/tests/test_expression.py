import pytest
from src.expression import Expression, Op


def expr(json: dict) -> Expression:
    return Expression.model_validate(json)


# --- JSON parsing ---

def test_parse_single_op():
    e = expr({
        "type": "expr", "logic": "and",
        "operands": [{"type": "op", "key": "bid", "op": "<", "value": 100}],
    })
    assert isinstance(e.operands[0], Op)
    assert e.operands[0].key == "bid"
    assert e.operands[0].op == "<"
    assert e.operands[0].value == 100.0


def test_parse_nested_expression():
    e = expr({
        "type": "expr", "logic": "and",
        "operands": [
            {"type": "op", "key": "a", "op": ">", "value": 1},
            {
                "type": "expr", "logic": "or",
                "operands": [
                    {"type": "op", "key": "b", "op": "<", "value": 2},
                    {"type": "op", "key": "c", "op": "==", "value": 3},
                ],
            },
        ],
    })
    assert e.logic == "and"
    assert isinstance(e.operands[1], Expression)
    assert e.operands[1].logic == "or"


# --- Op.format ---

def test_op_format():
    op = Op(key="bid", op="<", value=100.0)
    assert op.format() == "bid < 100.0"


# --- Expression.format: flat ---

def test_format_flat_and():
    e = expr({
        "type": "expr", "logic": "and",
        "operands": [
            {"type": "op", "key": "a", "op": ">", "value": 1},
            {"type": "op", "key": "b", "op": "<", "value": 2},
        ],
    })
    assert e.format() == "a > 1.0 && b < 2.0"


def test_format_flat_or():
    e = expr({
        "type": "expr", "logic": "or",
        "operands": [
            {"type": "op", "key": "a", "op": ">", "value": 1},
            {"type": "op", "key": "b", "op": "<", "value": 2},
        ],
    })
    assert e.format() == "a > 1.0 || b < 2.0"


# --- Expression.format: precedence grouping ---

def test_format_or_inside_and_gets_parens():
    # (a || b) && c  — or has lower precedence, must be wrapped
    e = expr({
        "type": "expr", "logic": "and",
        "operands": [
            {
                "type": "expr", "logic": "or",
                "operands": [
                    {"type": "op", "key": "a", "op": ">", "value": 1},
                    {"type": "op", "key": "b", "op": ">", "value": 2},
                ],
            },
            {"type": "op", "key": "c", "op": ">", "value": 3},
        ],
    })
    assert e.format() == "(a > 1.0 || b > 2.0) && c > 3.0"


def test_format_and_inside_or_no_parens():
    # a || b && c  — and already binds tighter, no parens needed
    e = expr({
        "type": "expr", "logic": "or",
        "operands": [
            {"type": "op", "key": "a", "op": ">", "value": 1},
            {
                "type": "expr", "logic": "and",
                "operands": [
                    {"type": "op", "key": "b", "op": ">", "value": 2},
                    {"type": "op", "key": "c", "op": ">", "value": 3},
                ],
            },
        ],
    })
    assert e.format() == "a > 1.0 || b > 2.0 && c > 3.0"


def test_format_deeply_nested():
    # (a || b) && (c || d)
    e = expr({
        "type": "expr",
        "logic": "and",
        "operands": [
            {
                "type": "expr", "logic": "or",
                "operands": [
                    {"type": "op", "key": "a", "op": ">", "value": 1},
                    {"type": "op", "key": "b", "op": ">", "value": 2},
                ],
            },
            {
                "type": "expr", "logic": "or",
                "operands": [
                    {"type": "op", "key": "c", "op": ">", "value": 3},
                    {"type": "op", "key": "d", "op": ">", "value": 4},
                ],
            },
        ],
    })
    assert e.format() == "(a > 1.0 || b > 2.0) && (c > 3.0 || d > 4.0)"
