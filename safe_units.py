"""
Safe evaluation of unit-bearing parameter strings coming from the UI.

The frontend sends physical parameters as strings ("20*ms", "0.1*mV", "4.5").
We must turn those into Brian2 Quantities WITHOUT calling eval() on user input.

Strategy: parse to a Python AST, then walk it with an interpreter that supports
only arithmetic and name lookups resolved against an explicit allowlist of
Brian2 units and constants. Anything else -- attribute access, calls,
subscripts, comprehensions -- is rejected before it can be evaluated.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from brian2.core.namespace import DEFAULT_CONSTANTS, DEFAULT_UNITS


class UnsafeExpression(ValueError):
    """Raised when a parameter string contains anything outside the allowlist."""


# Only these node types may appear anywhere in a parameter expression.
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}

#: `**` is a denial-of-service in five characters. `9**9**9` asks Python to
#: build an integer with hundreds of millions of digits and never returns, and
#: because validation runs inline in the web process rather than in the
#: simulation worker, that hangs the whole server with no timeout to rescue it.
#: No physical parameter needs an exponent anywhere near this.
_MAX_EXPONENT = 64

# Names a parameter string is allowed to reference: every Brian2 unit
# (volt, mV, ms, Hz, nS, pF, ...) plus pi/e/inf.
_ALLOWED_NAMES: dict[str, Any] = {**DEFAULT_UNITS, **DEFAULT_CONSTANTS}


def safe_eval_quantity(expr: str) -> Any:
    """Evaluate a unit-bearing expression string into a Brian2 Quantity.

    >>> safe_eval_quantity("20*ms")
    20. * msecond
    >>> safe_eval_quantity("4.5")
    4.5
    >>> safe_eval_quantity("__import__('os')")
    Traceback (most recent call last):
        ...
    spikenet.safe_units.UnsafeExpression: ...
    """
    if not isinstance(expr, str):
        # Already a number or Quantity -- nothing to parse.
        return expr

    expr = expr.strip()
    if not expr:
        raise UnsafeExpression("Empty parameter expression.")
    if len(expr) > 200:
        raise UnsafeExpression("Parameter expression is implausibly long.")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpression(f"Could not parse {expr!r}: {exc.msg}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise UnsafeExpression(
                f"{type(node).__name__} is not permitted in a parameter "
                f"expression (in {expr!r}). Use numbers, units and "
                f"+ - * / ** only."
            )

    return _eval_node(tree.body, expr)


def _eval_node(node: ast.AST, source: str) -> Any:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise UnsafeExpression(
                f"Only numeric literals are allowed (in {source!r})."
            )
        return node.value

    if isinstance(node, ast.Name):
        try:
            return _ALLOWED_NAMES[node.id]
        except KeyError:
            raise UnsafeExpression(
                f"Unknown unit or constant {node.id!r} in {source!r}."
            ) from None

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, source)
        right = _eval_node(node.right, source)
        if isinstance(node.op, ast.Pow):
            try:
                magnitude = abs(float(right))
            except (TypeError, ValueError):
                raise UnsafeExpression(
                    f"Exponent must be a plain number (in {source!r})."
                ) from None
            if not math.isfinite(magnitude) or magnitude > _MAX_EXPONENT:
                raise UnsafeExpression(
                    f"Exponents above {_MAX_EXPONENT} are not allowed "
                    f"(in {source!r})."
                )
        try:
            return _BINOPS[type(node.op)](left, right)
        except ZeroDivisionError:
            raise UnsafeExpression(f"Division by zero in {source!r}.") from None
        except OverflowError:
            raise UnsafeExpression(f"Result is too large in {source!r}.") from None

    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS[type(node.op)]
        return op(_eval_node(node.operand, source))

    raise UnsafeExpression(f"Unsupported expression in {source!r}.")
