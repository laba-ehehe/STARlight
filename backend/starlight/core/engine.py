"""The inference engine: forward chaining, backward chaining and conflict resolution.

Forward chaining (data-driven) is used for answer analysis: extractor facts go
into working memory and rules fire until nothing new can be concluded.

Backward chaining (goal-driven) is used for live follow-up selection: start
from (COMPLETE ?a), try its sub-goals in priority order, and report the first
one that cannot be proven, which decides the follow-up question.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from .explain import ExplanationTrace, unify_head
from .expr import evaluate
from .facts import WILDCARD, Bindings, Fact, is_var, substitute
from .kb import KnowledgeBase
from .memory import WorkingMemory
from .rules import Match, Rule

DEFAULT_THRESHOLD = 0.5


class InferenceError(RuntimeError):
    pass


@dataclass
class ProofFailure:
    """A node in the explanation of why a backward-chaining goal failed."""
    goal: str
    reason: str
    rule_id: str | None = None
    condition: str = ""
    blocking: Fact | None = None
    causes: list[ProofFailure] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "goal": self.goal, "reason": self.reason, "rule": self.rule_id,
            "condition": self.condition,
            "blocking": self.blocking.to_dict() if self.blocking else None,
            "causes": [c.to_dict() for c in self.causes],
        }


@dataclass
class BackwardResult:
    goal: tuple
    proven: bool
    bindings: Bindings = field(default_factory=dict)
    failed_subgoal: str | None = None     # e.g. "(RESULT-STATED q2)"
    blocking: Fact | None = None          # e.g. (MISSING-RESULT q2)
    followup: str | None = None           # e.g. "What was the outcome?"
    followup_rule: str | None = None      # e.g. "R7"
    failure: ProofFailure | None = None
    steps: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "goal": "(" + " ".join(map(str, self.goal)) + ")",
            "proven": self.proven,
            "failed_subgoal": self.failed_subgoal,
            "blocking": self.blocking.to_dict() if self.blocking else None,
            "followup": self.followup,
            "followup_rule": self.followup_rule,
            "failure": self.failure.to_dict() if self.failure else None,
            "steps": self.steps,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


class InferenceEngine:
    def __init__(self, kb: KnowledgeBase, wm: WorkingMemory | None = None,
                 trace: ExplanationTrace | None = None, *,
                 threshold: float = DEFAULT_THRESHOLD, max_cycles: int = 10_000,
                 max_depth: int = 12) -> None:
        self.kb = kb
        self.wm = wm if wm is not None else WorkingMemory()
        self.trace = trace if trace is not None else ExplanationTrace()
        self.threshold = threshold
        self.max_cycles = max_cycles
        self.max_depth = max_depth
        self._fired: set[tuple] = set()

    # ================================================================ forward chaining

    def forward_chain(self, groups: str | Iterable[str] | None = None) -> list[Fact]:
        """Match → resolve → act until quiescence. Returns the facts newly asserted.

        groups: rule group(s) to run; None runs every group except 'goal'.
        Refraction: each match (rule + matched facts + bindings) fires at most once
        over the engine's lifetime, so calling this again only fires new matches.

        The agenda is kept per rule. After a firing, only rules whose conditions
        mention a predicate that changed are matched again.
        """
        rules = self.kb.rules_in(groups)
        per_rule = {r.id: self._matches_for(r) for r in rules}
        new_facts: list[Fact] = []
        for _ in range(self.max_cycles):
            agenda = self.resolve_conflicts([m for ms in per_rule.values() for m in ms])
            if not agenda:
                return new_facts
            match, strength = agenda[0]
            self._fired.add(match.key)
            per_rule[match.rule.id] = [x for x in per_rule[match.rule.id]
                                       if x[0].key != match.key]
            changed = self._act(match, strength, new_facts)
            for rule in rules:
                if changed & _condition_preds(rule):
                    per_rule[rule.id] = self._matches_for(rule)
        raise InferenceError(f"forward chaining did not settle after {self.max_cycles} cycles")

    def resolve_conflicts(self, agenda: Sequence[tuple[Match, float]]
                          ) -> list[tuple[Match, float]]:
        """Order the agenda: priority, then specificity, then recency, then rule order."""
        order = {r.id: i for i, r in enumerate(self.kb.rules)}

        def key(item: tuple[Match, float]):
            m, _ = item
            recency = max((self.wm.timestamp(f) for f in m.facts), default=0)
            return (-m.rule.priority, -len(m.rule.conditions), -recency, order.get(m.rule.id, 0))

        return sorted(agenda, key=key)

    def reset(self) -> None:
        """Forget which matches have fired (working memory is left alone)."""
        self._fired.clear()

    def _matches_for(self, rule: Rule) -> list[tuple[Match, float]]:
        out = []
        for m in rule.matches(self.wm, self.kb.fuzzy_vars):
            if m.key in self._fired:
                continue
            s = rule.strength(m)
            if s >= self.threshold:
                out.append((m, s))
        return out

    def _act(self, match: Match, strength: float, new_facts: list[Fact]) -> set[str]:
        """Fire one match. Returns the predicates whose facts changed."""
        rule = match.rule
        changed: set[str] = set()
        retracted: list[Fact] = []
        for pattern in rule.retractions(match):
            for fact, _ in self.wm.query(pattern):
                self.wm.retract(fact)
                retracted.append(fact)
                changed.add(fact.pred)
        produced = rule.fire(match)
        for fact in produced:
            if self.wm.assert_fact(fact):
                new_facts.append(fact)
                changed.add(fact.pred)
        self.trace.record(rule, match, strength, produced, retracted)
        return changed

    # ================================================================ backward chaining

    def backward_chain(self, goal: Sequence[Any]) -> BackwardResult:
        """Try to prove a goal such as ("COMPLETE", "q2").

        Sub-goals are tried in the order the goal rule lists them, so the first
        sub-goal that fails is the highest-priority gap in the answer (R16).
        """
        start = time.perf_counter()
        goal = tuple(goal)
        steps: list[str] = []
        failures: list[ProofFailure] = []
        self._active: set[tuple] = set()
        self._derived_by: dict[tuple, str] = {}
        for b in self._prove(goal, {}, 0, failures, steps):
            return BackwardResult(goal, True, b, steps=steps,
                                  elapsed_ms=(time.perf_counter() - start) * 1000)

        result = BackwardResult(goal, False, steps=steps)
        top = failures[-1] if failures else None
        result.failure = top
        if top is not None:
            sub = top.causes[0] if top.causes else top
            result.failed_subgoal = sub.goal if sub is not top else None
            result.blocking, result.followup, result.followup_rule = self._followup_for(sub)
        result.elapsed_ms = (time.perf_counter() - start) * 1000
        return result

    def _followup_for(self, node: ProofFailure) -> tuple[Fact | None, str | None, str | None]:
        """Walk down the failure chain to the fact that blocked it, and pick the follow-up.

        Preference: the follow-up of the rule that derived the blocking fact
        (e.g. R7 derived MISSING-RESULT → "What was the outcome?"), otherwise the
        follow-up attached to the failing sub-goal rule itself.
        """
        chain = [node]
        while chain[-1].causes:
            chain.append(chain[-1].causes[0])
        for n in reversed(chain):
            if n.blocking is not None:
                producer = self.trace.producer(n.blocking)
                rule_id = producer.rule_id if producer else n.blocking.source
                rule = self._rule_or_none(rule_id)
                if rule and rule.followup:
                    return n.blocking, rule.followup, rule.id
                return n.blocking, None, None
        for n in chain:
            rule = self._rule_or_none(n.rule_id)
            if rule and rule.followup:
                return None, rule.followup, rule.id
        return None, None, None

    def _rule_or_none(self, rule_id: str | None) -> Rule | None:
        try:
            return self.kb.rule(rule_id) if rule_id else None
        except KeyError:
            return None

    def _prove(self, pattern: tuple, b: Bindings, depth: int,
               failures: list[ProofFailure], steps: list[str]) -> Iterator[Bindings]:
        goal = substitute(pattern, b)
        goal_text = _pstr(goal)
        proved = False

        for fact, b2 in self.wm.query(pattern, b):
            proved = True
            steps.append(f"{'  ' * depth}{fact} is a known fact")
            yield b2

        if depth >= self.max_depth or goal in self._active:
            if not proved:
                failures.append(ProofFailure(goal_text, "search depth or cycle limit reached"))
            return

        local: list[ProofFailure] = []
        self._active.add(goal)
        try:
            for rule in self.kb.rules_concluding(pattern[0]):
                for head in rule.concludes(pattern[0]):
                    init = unify_head(head, goal)
                    if init is None:
                        continue
                    for rb in self._prove_conditions(rule, 0, init, depth + 1, local, steps):
                        out = dict(b)
                        for g, h in zip(goal[1:], head[1:]):
                            if is_var(g) and g != WILDCARD:
                                value = rb.get(h) if is_var(h) else h
                                if value is not None:
                                    out[g] = value
                        proved = True
                        self._derived_by.setdefault(substitute(goal, out), rule.id)
                        steps.append(f"{'  ' * depth}{_pstr(substitute(goal, out))} "
                                     f"proved by {rule.id}")
                        yield out
        finally:
            self._active.discard(goal)

        if not proved:
            reason = ("no fact or rule establishes it" if not local
                      else "every rule that could establish it failed")
            failures.append(ProofFailure(goal_text, reason, causes=local))

    def _prove_conditions(self, rule: Rule, i: int, b: Bindings, depth: int,
                          failures: list[ProofFailure], steps: list[str]) -> Iterator[Bindings]:
        if i == len(rule.conditions):
            yield b
            return
        cond = rule.conditions[i]
        here = cond.describe(b)

        def fail(reason: str, blocking: Fact | None = None,
                 causes: list[ProofFailure] | None = None) -> None:
            failures.append(ProofFailure(_pstr(substitute(cond.pattern, b)) if cond.pattern
                                         else here, reason, rule.id, here, blocking,
                                         causes or []))

        match cond.kind:
            case "fact":
                sub: list[ProofFailure] = []
                any_proof = False
                for b2 in self._prove(cond.pattern, b, depth, sub, steps):
                    any_proof = True
                    yield from self._prove_conditions(rule, i + 1, b2, depth, failures, steps)
                if not any_proof:
                    fail("sub-goal could not be proven", causes=sub)
            case "not":
                found = self.wm.query(cond.pattern, b)
                blocking = found[0][0] if found else None
                if blocking is None:
                    # Negation as failure: also try to derive it before concluding it is absent.
                    for b2 in self._prove(cond.pattern, b, depth, [], []):
                        derived = substitute(cond.pattern, b2)
                        blocking = Fact(derived[0], derived[1:],
                                        source=self._derived_by.get(derived, ""))
                        break
                if blocking is None:
                    yield from self._prove_conditions(rule, i + 1, b, depth, failures, steps)
                else:
                    fail(f"blocked by {blocking}", blocking=blocking)
            case "test":
                if evaluate(cond.expr, b):
                    yield from self._prove_conditions(rule, i + 1, b, depth, failures, steps)
                else:
                    fail(f"{cond.expr} is false")
            case "bind":
                yield from self._prove_conditions(rule, i + 1,
                                                  {**b, cond.target: evaluate(cond.expr, b)},
                                                  depth, failures, steps)
            case "count":
                n = len({f.key for f, _ in self.wm.query(cond.pattern, b)})
                if cond.target in b and b[cond.target] != n:
                    fail(f"count is {n}, not {b[cond.target]}")
                else:
                    yield from self._prove_conditions(rule, i + 1, {**b, cond.target: n},
                                                      depth, failures, steps)
            case "fuzzy":
                value, var, set_name = cond.fuzzy
                x = b[value] if is_var(value) else value
                degree = self.kb.fuzzy_vars[var].degree(set_name, float(x))
                if degree >= self.threshold:
                    yield from self._prove_conditions(rule, i + 1, b, depth, failures, steps)
                else:
                    fail(f"{var} {x} is {set_name} only to degree {degree:.2f}")


def _condition_preds(rule: Rule) -> set[str]:
    preds = getattr(rule, "_cond_preds", None)
    if preds is None:
        preds = {c.pattern[0] for c in rule.conditions if c.kind in ("fact", "not", "count")}
        rule._cond_preds = preds
    return preds


def _pstr(terms: Sequence[Any]) -> str:
    return "(" + " ".join(str(t) for t in terms) + ")"
