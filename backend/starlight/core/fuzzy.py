"""Fuzzy sets for vague quantities such as answer duration and speaking pace."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MembershipFunction:
    """Trapezoid (a, b, c, d): 0 below a, rising to 1 at b, 1 until c, falling to 0 at d.

    a == b gives a left shoulder; c == d == inf gives a right shoulder.
    """

    name: str
    a: float
    b: float
    c: float
    d: float

    def __post_init__(self) -> None:
        if not self.a <= self.b <= self.c <= self.d:
            raise ValueError(f"{self.name}: need a <= b <= c <= d, got "
                             f"({self.a}, {self.b}, {self.c}, {self.d})")

    def degree(self, x: float) -> float:
        if x < self.a or x > self.d:
            return 0.0
        if self.b <= x <= self.c:
            return 1.0
        if x < self.b:
            return (x - self.a) / (self.b - self.a)
        if math.isinf(self.d):
            return 1.0
        return (self.d - x) / (self.d - self.c)


class FuzzyVariable:
    """A measured quantity (e.g. answer_duration in seconds) with its fuzzy sets."""

    def __init__(self, name: str, sets: Mapping[str, MembershipFunction], unit: str = "") -> None:
        self.name = name
        self.unit = unit
        self.sets = dict(sets)

    def fuzzify(self, x: float) -> dict[str, float]:
        """Membership degree of x in every set."""
        return {name: mf.degree(x) for name, mf in self.sets.items()}

    def degree(self, set_name: str, x: float) -> float:
        try:
            return self.sets[set_name].degree(x)
        except KeyError:
            raise KeyError(f"fuzzy variable {self.name!r} has no set {set_name!r}") from None

    @classmethod
    def from_dict(cls, name: str, d: Mapping) -> FuzzyVariable:
        """{"unit": "s", "sets": {"SHORT": [0, 0, 60, 90], ...}}"""
        sets = {
            set_name: MembershipFunction(set_name, *(float(v) for v in bounds))
            for set_name, bounds in d["sets"].items()
        }
        return cls(name, sets, d.get("unit", ""))

    def __repr__(self) -> str:
        return f"FuzzyVariable({self.name!r}, sets={list(self.sets)})"
