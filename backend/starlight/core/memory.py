"""Working memory: the current facts about one session."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from itertools import count
from typing import Any

from .facts import WILDCARD, Bindings, Fact, is_var


class WorkingMemory:
    """Facts indexed by predicate. Each fact key appears at most once.

    Every assertion gets an increasing timestamp, which the engine uses for
    recency-based conflict resolution.
    """

    def __init__(self, facts: Sequence[Fact] = ()) -> None:
        self._by_pred: dict[str, dict[tuple, Fact]] = {}
        self._stamp: dict[tuple, int] = {}
        self._by_arg0: dict[tuple, dict[tuple, Fact]] = {}
        self._clock = count(1)
        for fact in facts:
            self.assert_fact(fact)

    def assert_fact(self, fact: Fact) -> bool:
        """Add a fact. Returns True if it is new.

        If the fact already exists with a lower certainty, the higher certainty
        is kept (fuzzy OR), but it does not count as new.
        """
        bucket = self._by_pred.setdefault(fact.pred, {})
        existing = bucket.get(fact.key)
        if existing is None:
            bucket[fact.key] = fact
            self._stamp[fact.key] = next(self._clock)
            if fact.args:
                self._by_arg0.setdefault((fact.pred, fact.args[0]), {})[fact.key] = fact
            return True
        if fact.certainty > existing.certainty:
            bucket[fact.key] = fact
            if fact.args:
                self._by_arg0[(fact.pred, fact.args[0])][fact.key] = fact
        return False

    def assert_all(self, facts: Sequence[Fact]) -> int:
        return sum(self.assert_fact(f) for f in facts)

    def retract(self, pattern: Fact | Sequence[Any]) -> int:
        """Remove a fact, or every fact matching a pattern. Returns how many were removed."""
        targets = [pattern] if isinstance(pattern, Fact) else [f for f, _ in self.query(pattern)]
        removed = 0
        for fact in targets:
            bucket = self._by_pred.get(fact.pred, {})
            if bucket.pop(fact.key, None) is not None:
                self._stamp.pop(fact.key, None)
                if fact.args:
                    self._by_arg0.get((fact.pred, fact.args[0]), {}).pop(fact.key, None)
                removed += 1
        return removed

    def query(self, pattern: Sequence[Any], bindings: Mapping[str, Any] | None = None
              ) -> list[tuple[Fact, Bindings]]:
        """All facts matching the pattern, each with the bindings it produces."""
        results = []
        candidates = self._by_pred.get(pattern[0], {})
        if len(pattern) > 1:
            first = pattern[1]
            if is_var(first) and first != WILDCARD and bindings and first in bindings:
                first = bindings[first]
            if not is_var(first):
                candidates = self._by_arg0.get((pattern[0], first), {})
        for fact in candidates.values():
            b = fact.matches(pattern, bindings)
            if b is not None:
                results.append((fact, b))
        return results

    def get(self, *terms: Any) -> Fact | None:
        """Exact lookup: wm.get("PACE", "q1", 150)."""
        return self._by_pred.get(terms[0], {}).get(tuple(terms))

    def first(self, pattern: Sequence[Any]) -> Fact | None:
        found = self.query(pattern)
        return found[0][0] if found else None

    def timestamp(self, fact: Fact) -> int:
        return self._stamp.get(fact.key, 0)

    def facts(self, pred: str | None = None) -> list[Fact]:
        if pred is not None:
            return list(self._by_pred.get(pred, {}).values())
        return sorted((f for b in self._by_pred.values() for f in b.values()),
                      key=self.timestamp)

    def clear(self) -> None:
        self._by_pred.clear()
        self._stamp.clear()
        self._by_arg0.clear()

    def __contains__(self, fact: Fact) -> bool:
        return fact.key in self._by_pred.get(fact.pred, {})

    def __len__(self) -> int:
        return len(self._stamp)

    def __iter__(self) -> Iterator[Fact]:
        return iter(self.facts())
