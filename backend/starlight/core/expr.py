"""A tiny, safe expression language for rule tests and computed bindings.

Rules in the YAML file may say  test: "?ratio > 2"  or  bind: {?n: "floor((?d - 5) / 6)"}.
Expressions are parsed with Python's ast module and evaluated by walking a
whitelist of node types, so rule files can never run arbitrary code.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

_VAR = re.compile(r"\?([A-Za-z_]\w*)")

_FUNCS = {
    "abs": abs, "min": min, "max": max, "round": round, "len": len,
    "floor": math.floor, "ceil": math.ceil,
}
_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
}
_CMPOPS = {
    ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


class ExpressionError(ValueError):
    pass


def variables_in(source: str) -> list[str]:
    """Rule variables (with their '?') referenced by an expression."""
    return ["?" + name for name in _VAR.findall(source)]


@lru_cache(maxsize=512)
def _parse(source: str) -> ast.Expression:
    try:
        return ast.parse(_VAR.sub(r"__v_\1", source), mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"cannot parse expression {source!r}: {e.msg}") from None


def evaluate(source: str, bindings: Mapping[str, Any]) -> Any:
    """Evaluate an expression with rule variables taken from bindings."""
    env = {"__v_" + k[1:]: v for k, v in bindings.items() if k.startswith("?")}
    return _eval(_parse(source).body, env, source)


def _eval(node: ast.AST, env: dict[str, Any], source: str) -> Any:
    match node:
        case ast.Constant(value=v) if isinstance(v, (int, float, str, bool)) or v is None:
            return v
        case ast.Name(id=name):
            if name in env:
                return env[name]
            if name.startswith("__v_"):
                raise ExpressionError(f"unbound variable ?{name[4:]} in {source!r}")
            raise ExpressionError(f"unknown name {name!r} in {source!r}")
        case ast.Tuple(elts=elts) | ast.List(elts=elts):
            return tuple(_eval(e, env, source) for e in elts)
        case ast.UnaryOp(op=ast.USub(), operand=x):
            return -_eval(x, env, source)
        case ast.UnaryOp(op=ast.Not(), operand=x):
            return not _eval(x, env, source)
        case ast.BinOp(left=l, op=op, right=r) if type(op) in _BINOPS:
            return _BINOPS[type(op)](_eval(l, env, source), _eval(r, env, source))
        case ast.BoolOp(op=ast.And(), values=vals):
            return all(_eval(v, env, source) for v in vals)
        case ast.BoolOp(op=ast.Or(), values=vals):
            return any(_eval(v, env, source) for v in vals)
        case ast.Compare(left=left, ops=ops, comparators=comps):
            lhs = _eval(left, env, source)
            for op, comp in zip(ops, comps):
                rhs = _eval(comp, env, source)
                if type(op) not in _CMPOPS or not _CMPOPS[type(op)](lhs, rhs):
                    return False
                lhs = rhs
            return True
        case ast.IfExp(test=t, body=b, orelse=o):
            return _eval(b, env, source) if _eval(t, env, source) else _eval(o, env, source)
        case ast.Call(func=ast.Name(id=fname), args=args, keywords=[]) if fname in _FUNCS:
            return _FUNCS[fname](*(_eval(a, env, source) for a in args))
    raise ExpressionError(f"unsupported syntax {ast.dump(node)[:40]}... in {source!r}")
