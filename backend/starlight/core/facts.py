"""Facts: the symbol structures the inference engine reasons over.

A fact such as (MISSING-RESULT q2) is a predicate plus a tuple of arguments.
It also carries where it came from on the session timeline (span), how strongly
it holds (certainty, 0..1) and who produced it (an extractor module or a rule id).

Patterns are sequences like ["STAR-SEGMENT", "?a", "result", "?_"]:
  - "?name" is a variable; the same variable must match the same value everywhere.
  - "?_"    is an anonymous wildcard; it matches anything and binds nothing.
  - anything else is a literal compared with ==.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .timespan import TimeSpan

Bindings = dict[str, Any]
WILDCARD = "?_"


def is_var(term: Any) -> bool:
    return isinstance(term, str) and term.startswith("?") and len(term) > 1


def substitute(pattern: Sequence[Any], bindings: Mapping[str, Any]) -> tuple:
    """Replace bound variables in a pattern with their values."""
    return tuple(bindings.get(t, t) if is_var(t) and t != WILDCARD else t for t in pattern)


def unbound_vars(pattern: Sequence[Any], bindings: Mapping[str, Any]) -> list[str]:
    return [t for t in pattern if is_var(t) and t != WILDCARD and t not in bindings]


def match_terms(
    pattern: Sequence[Any], values: Sequence[Any], bindings: Mapping[str, Any] | None = None
) -> Bindings | None:
    """Match pattern terms against concrete values. Returns extended bindings or None."""
    if len(pattern) != len(values):
        return None
    result: Bindings = dict(bindings or {})
    for term, value in zip(pattern, values):
        if term == WILDCARD:
            continue
        if is_var(term):
            if term in result:
                if result[term] != value:
                    return None
            else:
                result[term] = value
        elif term != value:
            return None
    return result


@dataclass(frozen=True, slots=True)
class Fact:
    pred: str
    args: tuple = ()
    span: TimeSpan | None = field(default=None, compare=False)
    certainty: float = field(default=1.0, compare=False)
    source: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.args, tuple):
            object.__setattr__(self, "args", tuple(self.args))
        if not 0.0 <= self.certainty <= 1.0:
            raise ValueError(f"certainty must be in [0, 1], got {self.certainty}")

    @classmethod
    def of(cls, *terms: Any, span: TimeSpan | None = None, certainty: float = 1.0,
           source: str = "") -> Fact:
        """Fact.of("MISSING-RESULT", "q2") — convenience constructor."""
        return cls(terms[0], tuple(terms[1:]), span, certainty, source)

    @property
    def key(self) -> tuple:
        """Identity used by working memory: predicate and arguments only."""
        return (self.pred, *self.args)

    def matches(self, pattern: Sequence[Any], bindings: Mapping[str, Any] | None = None
                ) -> Bindings | None:
        """Bindings that make the pattern equal this fact, or None."""
        if not pattern or pattern[0] != self.pred:
            return None
        return match_terms(pattern[1:], self.args, bindings)

    def with_span(self, span: TimeSpan | None) -> Fact:
        return replace(self, span=span)

    def to_dict(self) -> dict:
        return {
            "pred": self.pred,
            "args": list(self.args),
            "span": self.span.to_dict() if self.span else None,
            "certainty": round(self.certainty, 3),
            "source": self.source,
            "text": str(self),
        }

    def __str__(self) -> str:
        return "(" + " ".join([self.pred, *(_fmt_arg(a) for a in self.args)]) + ")"


def _fmt_arg(arg: Any) -> str:
    if isinstance(arg, float):
        return f"{arg:.2f}".rstrip("0").rstrip(".")
    if isinstance(arg, str) and (" " in arg or arg == ""):
        return f'"{arg}"'
    return str(arg)
