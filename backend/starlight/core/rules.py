"""IF-THEN rules loaded from the knowledge base YAML.

A rule's IF part is an ordered list of conditions, evaluated left to right:

  fact:  [PRED, ?x, lit]      a matching fact exists (binds variables)
         as: ?name            optional alias, so an action can take its span
  not:   [PRED, ?x, ?_]       no matching fact exists (negation as failure)
  test:  "?r > 2"             expression over bound variables (also [?r, ">", 2])
  fuzzy: [?d, answer_duration, SHORT]
                              membership degree of a value in a fuzzy set
  bind:  {?n: "floor((?d - 5) / 6)"}
                              compute a new variable
  count: [ISSUE, ?a, clarity, ?_]
         as: ?n               number of distinct matching facts (0 if none); the
                              pattern's own variables stay local to the count

The THEN part is a list of actions:

  assert:  [PRED, ?x, ...]    add a fact; optional  span: [?alias, ...]
  retract: [PRED, ?x, ...]    remove matching facts

A rule's strength for one match is min(fuzzy degrees) x min(matched fact
certainties); it fires only if the strength reaches the engine's threshold.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .expr import evaluate, variables_in
from .facts import WILDCARD, Bindings, Fact, is_var, substitute, unbound_vars
from .fuzzy import FuzzyVariable
from .memory import WorkingMemory
from .timespan import TimeSpan

CONDITION_KINDS = ("fact", "not", "test", "fuzzy", "bind", "count")


class RuleError(ValueError):
    pass


@dataclass(frozen=True)
class Condition:
    kind: str
    pattern: tuple = ()          # fact / not
    expr: str = ""               # test / bind
    target: str = ""             # bind: variable being computed
    fuzzy: tuple = ()            # fuzzy: (value, variable, set)
    alias: str | None = None     # fact: "as" name

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Condition:
        kinds = [k for k in d if k in CONDITION_KINDS]
        if len(kinds) != 1:
            raise RuleError(f"condition needs exactly one of {CONDITION_KINDS}: {dict(d)}")
        kind = kinds[0]
        value = d[kind]
        match kind:
            case "fact" | "not":
                if not isinstance(value, Sequence) or isinstance(value, str) or not value:
                    raise RuleError(f"{kind} condition needs a non-empty list: {value!r}")
                alias = _yaml_term(d["as"]) if "as" in d else None
                return cls(kind, pattern=_yaml_terms(value), alias=alias)
            case "count":
                if not isinstance(value, Sequence) or isinstance(value, str) or not value:
                    raise RuleError(f"count condition needs a non-empty list: {value!r}")
                target = _yaml_term(d.get("as"))
                if not is_var(target):
                    raise RuleError(f"count condition needs 'as: ?var': {dict(d)}")
                return cls(kind, pattern=_yaml_terms(value), target=target)
            case "test":
                return cls(kind, expr=_test_to_expr(value))
            case "fuzzy":
                if len(value) != 3:
                    raise RuleError(f"fuzzy condition needs [value, variable, set]: {value!r}")
                return cls(kind, fuzzy=_yaml_terms(value))
            case "bind":
                if not isinstance(value, Mapping) or len(value) != 1:
                    raise RuleError(f"bind needs one {{?var: expression}}: {value!r}")
                (target, expr), = value.items()
                target = str(target) if str(target).startswith("?") else "?" + str(target)
                return cls(kind, target=target, expr=str(expr))
        raise AssertionError(kind)

    def binds(self) -> list[str]:
        """Variables this condition introduces when it succeeds."""
        if self.kind == "fact":
            vs = [t for t in self.pattern if is_var(t) and t != WILDCARD]
            return vs + ([self.alias] if self.alias else [])
        if self.kind in ("bind", "count"):
            return [self.target]
        return []

    def uses(self) -> list[str]:
        """Variables that must already be bound before this condition runs."""
        if self.kind in ("test", "bind"):
            return variables_in(self.expr)
        if self.kind == "fuzzy":
            return [self.fuzzy[0]] if is_var(self.fuzzy[0]) else []
        return []

    def describe(self, bindings: Mapping[str, Any] | None = None) -> str:
        b = bindings or {}
        match self.kind:
            case "fact":
                return _pattern_str(substitute(self.pattern, b))
            case "not":
                return "NOT " + _pattern_str(substitute(self.pattern, b))
            case "test":
                return self.expr
            case "fuzzy":
                value, var, set_name = self.fuzzy
                return f"{var} {b.get(value, value)} is {set_name}"
            case "bind":
                return f"{self.target} = {self.expr}"
            case "count":
                return f"{self.target} = COUNT {_pattern_str(substitute(self.pattern, b))}"
        return self.kind


@dataclass(frozen=True)
class Action:
    kind: str                         # "assert" | "retract"
    pattern: tuple
    span_from: tuple[str, ...] = ()   # aliases whose spans form the new fact's span

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Action:
        for kind in ("assert", "retract"):
            if kind in d:
                span = d.get("span", ())
                span = (span,) if isinstance(span, (str, Mapping)) else span
                return cls(kind, _yaml_terms(d[kind]), _yaml_terms(span))
        raise RuleError(f"action needs 'assert' or 'retract': {dict(d)}")


@dataclass(frozen=True)
class Match:
    """One way a rule's conditions can all be satisfied."""
    rule: Rule
    bindings: Mapping[str, Any]
    facts: tuple[Fact, ...]                     # matched positive facts, in condition order
    aliases: Mapping[str, Fact] = field(default_factory=dict)
    degrees: tuple[tuple[str, float], ...] = ()  # (description, degree) for each fuzzy condition
    counted: tuple[Fact, ...] = ()               # facts counted by count conditions (for HOW only)

    @property
    def key(self) -> tuple:
        """Identity for refraction: a rule fires at most once per match key."""
        return (self.rule.id, tuple(f.key for f in self.facts),
                tuple(sorted((k, repr(v)) for k, v in self.bindings.items())))


@dataclass(frozen=True)
class Failure:
    """Why a rule could not fire (used by WHY-NOT explanations and backward chaining)."""
    rule_id: str
    index: int                 # index of the failing condition (len(conditions) = too weak)
    condition: str
    reason: str
    blocking: Fact | None = None
    strength: float | None = None
    bindings: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rule": self.rule_id,
            "condition": self.condition,
            "reason": self.reason,
            "blocking": self.blocking.to_dict() if self.blocking else None,
            "strength": self.strength,
        }


class Rule:
    def __init__(
        self,
        id: str,
        conditions: Sequence[Condition],
        actions: Sequence[Action],
        *,
        name: str = "",
        group: str = "",
        priority: int = 0,
        status: str = "hypothesis",
        source: str = "",
        description: str = "",
        followup: str | None = None,
        feedback: Mapping[str, Any] | None = None,
    ) -> None:
        self.id = id
        self.name = name or id
        self.group = group
        self.priority = priority
        self.status = status
        self.source = source
        self.description = description
        self.conditions = tuple(conditions)
        self.actions = tuple(actions)
        self.followup = followup
        self.feedback = dict(feedback) if feedback else None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Rule:
        try:
            rule_id = str(d["id"])
            conditions = [Condition.from_dict(c) for c in d.get("if", [])]
            actions = [Action.from_dict(a) for a in d.get("then", [])]
        except (KeyError, TypeError, RuleError) as e:
            raise RuleError(f"rule {d.get('id', '?')}: {e}") from None
        return cls(
            rule_id, conditions, actions,
            name=d.get("name", ""), group=d.get("group", ""),
            priority=int(d.get("priority", 0)), status=d.get("status", "hypothesis"),
            source=d.get("source", ""), description=d.get("description", ""),
            followup=d.get("followup"), feedback=d.get("feedback"),
        )

    # ------------------------------------------------------------------ matching

    def matches(self, wm: WorkingMemory, fuzzy_vars: Mapping[str, FuzzyVariable],
                bindings: Mapping[str, Any] | None = None) -> Iterator[Match]:
        """Every complete match of the conditions against working memory."""
        yield from self._walk(0, dict(bindings or {}), _Acc(), wm, fuzzy_vars, None)

    def strength(self, match: Match) -> float:
        fuzzy = min((d for _, d in match.degrees), default=1.0)
        certainty = min((f.certainty for f in match.facts), default=1.0)
        return fuzzy * certainty

    def fire(self, match: Match) -> list[Fact]:
        """New facts produced by the assert actions for this match."""
        strength = self.strength(match)
        default_span = TimeSpan.cover(f.span for f in match.facts)
        produced = []
        for action in self.actions:
            if action.kind != "assert":
                continue
            missing = unbound_vars(action.pattern, match.bindings)
            if missing:
                raise RuleError(f"rule {self.id}: unbound {missing} in assert {action.pattern}")
            terms = substitute(action.pattern, match.bindings)
            if action.span_from:
                span = TimeSpan.cover(match.aliases[a].span for a in action.span_from
                                      if a in match.aliases)
            else:
                span = default_span
            produced.append(Fact(terms[0], terms[1:], span, strength, self.id))
        return produced

    def retractions(self, match: Match) -> list[tuple]:
        return [substitute(a.pattern, match.bindings) for a in self.actions
                if a.kind == "retract"]

    def concludes(self, pred: str) -> list[tuple]:
        """Assert patterns of this rule whose predicate is pred."""
        return [a.pattern for a in self.actions if a.kind == "assert" and a.pattern[0] == pred]

    def first_failure(self, wm: WorkingMemory, fuzzy_vars: Mapping[str, FuzzyVariable],
                      bindings: Mapping[str, Any] | None = None,
                      threshold: float = 0.5) -> Failure | None:
        """None if the rule can fire; otherwise the most informative reason it cannot.

        "Most informative" = the failure that got furthest through the condition list.
        """
        failures: list[Failure] = []
        best_strength = None
        best_match = None
        for m in self._walk(0, dict(bindings or {}), _Acc(), wm, fuzzy_vars, failures.append):
            s = self.strength(m)
            if s >= threshold:
                return None
            if best_strength is None or s > best_strength:
                best_strength, best_match = s, m
        if best_match is not None:
            weakest = min(best_match.degrees, key=lambda d: d[1], default=None)
            detail = f" ({weakest[0]} only to degree {weakest[1]:.2f})" if weakest else ""
            return Failure(self.id, len(self.conditions), "strength",
                           f"strength {best_strength:.2f} is below {threshold}{detail}",
                           strength=best_strength, bindings=best_match.bindings)
        if not failures:
            return Failure(self.id, 0, "", "rule has no way to match")
        return max(failures, key=lambda f: f.index)

    # ------------------------------------------------------------------ internals

    def _walk(self, i: int, b: Bindings, acc: _Acc, wm: WorkingMemory,
              fuzzy_vars: Mapping[str, FuzzyVariable],
              on_fail: Callable[[Failure], None] | None) -> Iterator[Match]:
        if i == len(self.conditions):
            yield Match(self, b, acc.facts, acc.aliases, acc.degrees, acc.counted)
            return
        cond = self.conditions[i]

        def fail(reason: str, blocking: Fact | None = None) -> None:
            if on_fail:
                on_fail(Failure(self.id, i, cond.describe(b), reason, blocking, bindings=b))

        def go_on(b2: Bindings, acc2: _Acc = acc) -> Iterator[Match]:
            return self._walk(i + 1, b2, acc2, wm, fuzzy_vars, on_fail)

        match cond.kind:
            case "fact":
                found = wm.query(cond.pattern, b)
                if not found:
                    fail(f"no fact matches {cond.describe(b)}")
                for fact, b2 in found:
                    aliases = acc.aliases
                    if cond.alias:
                        aliases = {**aliases, cond.alias: fact}
                        b2 = {**b2, cond.alias: fact.key}
                    yield from go_on(b2, acc.add(facts=(fact,), aliases=aliases))
            case "not":
                found = wm.query(cond.pattern, b)
                if found:
                    fail(f"found {found[0][0]}", blocking=found[0][0])
                else:
                    yield from go_on(b)
            case "test":
                if evaluate(cond.expr, b):
                    yield from go_on(b)
                else:
                    shown = ", ".join(f"{v} = {_fmt(b[v])}" for v in variables_in(cond.expr)
                                      if v in b)
                    fail(f"{cond.expr} is false" + (f" ({shown})" if shown else ""))
            case "fuzzy":
                value, var_name, set_name = cond.fuzzy
                x = b[value] if is_var(value) else value
                if var_name not in fuzzy_vars:
                    raise RuleError(f"rule {self.id}: unknown fuzzy variable {var_name!r}")
                degree = fuzzy_vars[var_name].degree(set_name, float(x))
                if degree > 0:
                    yield from go_on(b, acc.add(degrees=((cond.describe(b), degree),)))
                else:
                    fail(f"{var_name} {_fmt(x)} is not {set_name} (degree 0)")
            case "bind":
                value = evaluate(cond.expr, b)
                if cond.target in b and b[cond.target] != value:
                    fail(f"{cond.target} is {b[cond.target]}, not {value}")
                else:
                    yield from go_on({**b, cond.target: value})
            case "count":
                counted = tuple({f.key: f for f, _ in wm.query(cond.pattern, b)}.values())
                n = len(counted)
                if cond.target in b and b[cond.target] != n:
                    fail(f"{cond.describe(b)} is {n}, not {b[cond.target]}")
                else:
                    yield from go_on({**b, cond.target: n}, acc.add(counted=counted))

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "group": self.group,
                "priority": self.priority, "status": self.status, "source": self.source,
                "description": self.description, "followup": self.followup,
                "feedback": self.feedback}

    def __repr__(self) -> str:
        return f"Rule({self.id} {self.name!r}, group={self.group!r}, priority={self.priority})"


@dataclass(frozen=True)
class _Acc:
    """What a partial match has collected so far."""
    facts: tuple[Fact, ...] = ()
    aliases: Mapping[str, Fact] = field(default_factory=dict)
    degrees: tuple[tuple[str, float], ...] = ()
    counted: tuple[Fact, ...] = ()

    def add(self, *, facts: tuple = (), aliases: Mapping[str, Fact] | None = None,
            degrees: tuple = (), counted: tuple = ()) -> _Acc:
        return _Acc(self.facts + facts, self.aliases if aliases is None else aliases,
                    self.degrees + degrees, self.counted + counted)


def _yaml_term(t: Any) -> Any:
    """Undo a YAML quirk: inside [...] PyYAML reads ?a as the mapping {a: None}."""
    if isinstance(t, Mapping) and len(t) == 1:
        (key, value), = t.items()
        if value is None:
            return "?" + str(key)
    return t


def _yaml_terms(terms: Sequence[Any]) -> tuple:
    return tuple(_yaml_term(t) for t in terms)


def _test_to_expr(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and len(value) == 3:
        lhs, op, rhs = _yaml_terms(value)
        return f"{_term(lhs)} {op} {_term(rhs)}"
    raise RuleError(f"test needs an expression string or [lhs, op, rhs]: {value!r}")


def _term(t: Any) -> str:
    return t if is_var(t) else repr(t)


def _pattern_str(terms: Sequence[Any]) -> str:
    return "(" + " ".join(_fmt(t) for t in terms) + ")"


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v)
