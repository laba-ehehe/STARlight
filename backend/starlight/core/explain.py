"""Explanation facility (Lecture 7).

HOW     (retrospective): which rule produced a conclusion, from which facts, recursively,
        down to the facts the extractors observed in the recording.
WHY NOT (counterfactual): for an expected conclusion that was not reached, which
        condition of each rule that could have produced it failed, and on what.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .facts import Fact, is_var, match_terms

if TYPE_CHECKING:
    from .kb import KnowledgeBase
    from .memory import WorkingMemory
    from .rules import Match, Rule


@dataclass(frozen=True)
class Firing:
    seq: int
    rule_id: str
    rule_name: str
    bindings: Mapping[str, Any]
    matched: tuple[Fact, ...]
    strength: float
    produced: tuple[Fact, ...]
    retracted: tuple[Fact, ...] = ()

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "rule": self.rule_id,
            "name": self.rule_name,
            "strength": round(self.strength, 3),
            "matched": [f.to_dict() for f in self.matched],
            "produced": [f.to_dict() for f in self.produced],
            "retracted": [f.to_dict() for f in self.retracted],
        }


class ExplanationTrace:
    """Log of every rule firing, indexed by the facts each firing produced."""

    def __init__(self) -> None:
        self.firings: list[Firing] = []
        self._producer: dict[tuple, Firing] = {}

    def record(self, rule: Rule, match: Match, strength: float,
               produced: Sequence[Fact], retracted: Sequence[Fact] = ()) -> Firing:
        firing = Firing(len(self.firings) + 1, rule.id, rule.name, dict(match.bindings),
                        tuple(match.facts) + tuple(match.counted), strength,
                        tuple(produced), tuple(retracted))
        self.firings.append(firing)
        for fact in produced:
            self._producer.setdefault(fact.key, firing)
        return firing

    def producer(self, fact: Fact) -> Firing | None:
        return self._producer.get(fact.key)

    def how(self, fact: Fact, _depth: int = 0) -> dict:
        """Tree explaining how a fact was concluded."""
        firing = self._producer.get(fact.key)
        node: dict[str, Any] = {"fact": fact.to_dict()}
        if firing is None or _depth > 20:
            node["observed_by"] = fact.source or "input"
            return node
        node.update({
            "rule": firing.rule_id,
            "rule_name": firing.rule_name,
            "strength": round(firing.strength, 3),
            "because": [self.how(f, _depth + 1) for f in firing.matched],
        })
        return node

    def why_not(self, pattern: Sequence[Any], kb: KnowledgeBase, wm: WorkingMemory,
                threshold: float = 0.5) -> dict:
        """Why no fact matching the pattern was concluded.

        Example: why_not(["RATING", "q2", "organization", "strong"], kb, wm)
        """
        pattern = tuple(pattern)
        found = wm.query(pattern)
        if found:
            return {"pattern": _pstr(pattern), "holds": True, "how": self.how(found[0][0])}
        reasons = []
        for rule in kb.rules_concluding(pattern[0]):
            for head in rule.concludes(pattern[0]):
                init = unify_head(head, pattern)
                if init is None:
                    continue
                failure = rule.first_failure(wm, kb.fuzzy_vars, init, threshold)
                if failure is None:
                    # The rule could fire now; it simply has not been run yet.
                    reasons.append({"rule": rule.id, "rule_name": rule.name,
                                    "reason": "rule is satisfied but has not fired yet"})
                    continue
                entry = {"rule": rule.id, "rule_name": rule.name, **failure.to_dict()}
                if failure.blocking is not None:
                    entry["blocking_how"] = self.how(failure.blocking)
                reasons.append(entry)
        if not reasons:
            reasons.append({"rule": None,
                            "reason": f"no rule concludes {pattern[0]}; "
                                      "it can only be observed by an extractor"})
        return {"pattern": _pstr(pattern), "holds": False, "reasons": reasons}

    def to_list(self) -> list[dict]:
        return [f.to_dict() for f in self.firings]

    def __len__(self) -> int:
        return len(self.firings)


def unify_head(head: Sequence[Any], goal: Sequence[Any]) -> dict | None:
    """Bind a rule's conclusion pattern (head) to the constants of a goal pattern.

    Goal variables and wildcards are left open. Returns the rule-variable
    bindings, or None if a constant in the goal contradicts the head.
    """
    if len(head) != len(goal) or head[0] != goal[0]:
        return None
    bindings: dict[str, Any] = {}
    for h, g in zip(head[1:], goal[1:]):
        if is_var(g):
            continue
        if is_var(h):
            if match_terms([h], [g], bindings) is None:
                return None
            bindings[h] = g
        elif h != g:
            return None
    return bindings


def _pstr(pattern: Sequence[Any]) -> str:
    return "(" + " ".join(str(t) for t in pattern) + ")"
