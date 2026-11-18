"""The knowledge base: rules, fuzzy variables and the fact vocabulary, loaded from YAML.

Experts edit the YAML files; no code changes are needed to add or adjust a rule.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .facts import WILDCARD, is_var
from .fuzzy import FuzzyVariable
from .rules import Rule, RuleError

GOAL_GROUP = "goal"   # rules used only by backward chaining


@dataclass(frozen=True)
class Problem:
    level: str            # "error" | "warning"
    rules: tuple[str, ...]
    message: str

    def __str__(self) -> str:
        return f"{self.level.upper()} [{', '.join(self.rules)}] {self.message}"


class KnowledgeBase:
    def __init__(
        self,
        rules: Sequence[Rule],
        fuzzy_vars: Mapping[str, FuzzyVariable] | None = None,
        vocabulary: Mapping[str, Mapping[str, Any]] | None = None,
        contradictions: Iterable[Sequence[str]] = (),
    ) -> None:
        self.rules = list(rules)
        self.fuzzy_vars = dict(fuzzy_vars or {})
        self.vocabulary = dict(vocabulary or {})
        self.contradictions = [tuple(pair) for pair in contradictions]
        self._by_id = {r.id: r for r in self.rules}

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, directory: str | Path) -> KnowledgeBase:
        """Read rules.yaml, fuzzy.yaml and (optionally) vocabulary.yaml from a directory."""
        d = Path(directory)
        rules_doc = _read_yaml(d / "rules.yaml")
        fuzzy_doc = _read_yaml(d / "fuzzy.yaml") if (d / "fuzzy.yaml").exists() else {}
        vocab_doc = _read_yaml(d / "vocabulary.yaml") if (d / "vocabulary.yaml").exists() else {}
        return cls.from_dicts(rules_doc, fuzzy_doc, vocab_doc)

    @classmethod
    def from_dicts(cls, rules_doc: Mapping, fuzzy_doc: Mapping | None = None,
                   vocab_doc: Mapping | None = None) -> KnowledgeBase:
        rules = [Rule.from_dict(r) for r in rules_doc.get("rules", [])]
        fuzzy = {name: FuzzyVariable.from_dict(name, spec)
                 for name, spec in (fuzzy_doc or {}).get("variables", {}).items()}
        vocab_doc = vocab_doc or {}
        return cls(rules, fuzzy, vocab_doc.get("predicates", {}),
                   vocab_doc.get("contradictions", []))

    # ------------------------------------------------------------------ access

    def rule(self, rule_id: str) -> Rule:
        return self._by_id[rule_id]

    def rules_in(self, groups: str | Iterable[str] | None = None) -> list[Rule]:
        """Rules in the given group(s). None = every non-goal rule."""
        if groups is None:
            return [r for r in self.rules if r.group != GOAL_GROUP]
        wanted = {groups} if isinstance(groups, str) else set(groups)
        return [r for r in self.rules if r.group in wanted]

    def rules_concluding(self, pred: str) -> list[Rule]:
        return [r for r in self.rules if r.concludes(pred)]

    @property
    def groups(self) -> list[str]:
        return list(dict.fromkeys(r.group for r in self.rules))

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)

    # ------------------------------------------------------------------ validation

    def validate(self) -> list[Problem]:
        """Consistency checks over the whole rule base (Lecture 4's 'rules about rules')."""
        problems: list[Problem] = []
        seen: dict[str, int] = {}
        for r in self.rules:
            seen[r.id] = seen.get(r.id, 0) + 1
        problems += [Problem("error", (rid,), f"duplicate rule id ({n} times)")
                     for rid, n in seen.items() if n > 1]

        concluded = {a.pattern[0] for r in self.rules for a in r.actions if a.kind == "assert"}
        for r in self.rules:
            problems += self._check_rule(r, concluded)
        problems += self._check_contradictions()
        return problems

    def _check_rule(self, r: Rule, concluded: set[str]) -> list[Problem]:
        out: list[Problem] = []

        def err(msg: str, level: str = "error") -> None:
            out.append(Problem(level, (r.id,), msg))

        bound: set[str] = set()
        aliases: set[str] = set()
        for c in r.conditions:
            missing = [v for v in c.uses() if v not in bound]
            if missing:
                err(f"condition '{c.describe()}' uses unbound {missing}")
            if c.kind in ("fact", "not", "count"):
                self._check_pattern(c.pattern, err, concluded, from_condition=True)
            if c.kind == "fuzzy":
                _, var, set_name = c.fuzzy
                if var not in self.fuzzy_vars:
                    err(f"unknown fuzzy variable {var!r}")
                elif set_name not in self.fuzzy_vars[var].sets:
                    err(f"fuzzy variable {var!r} has no set {set_name!r}")
            if c.alias:
                aliases.add(c.alias)
            bound.update(c.binds())

        if not r.actions:
            err("rule has no actions")
        for a in r.actions:
            if a.kind == "assert":
                if WILDCARD in a.pattern:
                    err(f"assert {list(a.pattern)} contains the wildcard ?_")
                missing = [t for t in a.pattern if is_var(t) and t != WILDCARD and t not in bound]
                if missing:
                    err(f"assert {list(a.pattern)} uses unbound {missing}")
                self._check_pattern(a.pattern, err, concluded, from_condition=False)
            for alias in a.span_from:
                if alias not in aliases:
                    err(f"span refers to {alias}, which no condition names with 'as'")
        if r.followup is not None and not r.followup.strip():
            err("empty follow-up text")
        fb = r.feedback or {}
        if fb.get("polarity") == "negative" and not str(fb.get("tip", "")).strip():
            err("negative feedback has no tip (R52: every negative item needs a concrete tip)")
        return out

    def _check_pattern(self, pattern: tuple, err, concluded: set[str], *, from_condition: bool
                       ) -> None:
        pred = pattern[0]
        if not self.vocabulary:
            return
        spec = self.vocabulary.get(pred)
        if spec is None:
            if from_condition and pred in concluded:
                err(f"predicate {pred} is derived by rules but missing from the vocabulary",
                    "warning")
            else:
                err(f"unknown predicate {pred}")
            return
        args = spec.get("args", [])
        if len(pattern) - 1 != len(args):
            err(f"{pred} takes {len(args)} args {args}, pattern has {len(pattern) - 1}")

    def _check_contradictions(self) -> list[Problem]:
        out = []
        for p, q in self.contradictions:
            for a in self.rules_concluding(p):
                for b in self.rules_concluding(q):
                    if a is b:
                        out.append(Problem("error", (a.id,), f"asserts both {p} and {q}"))
                    elif not self._separated(a, b):
                        out.append(Problem(
                            "warning", (a.id, b.id),
                            f"may both fire on the same facts, concluding {p} and {q}"))
        return out

    def _separated(self, a: Rule, b: Rule) -> bool:
        """Can the two rules' conditions never hold together? (conservative check)"""
        def preds(rule: Rule, kind: str) -> set[str]:
            return {c.pattern[0] for c in rule.conditions if c.kind == kind}

        if preds(a, "fact") & preds(b, "not") or preds(b, "fact") & preds(a, "not"):
            return True
        # One rule requires the other's conclusion to be absent.
        concl_a = {x.pattern[0] for x in a.actions if x.kind == "assert"}
        concl_b = {x.pattern[0] for x in b.actions if x.kind == "assert"}
        if preds(a, "not") & concl_b or preds(b, "not") & concl_a:
            return True
        for ca in a.conditions:
            for cb in b.conditions:
                if ca.kind == cb.kind == "fuzzy" and ca.fuzzy[1] == cb.fuzzy[1]:
                    var = self.fuzzy_vars.get(ca.fuzzy[1])
                    if var and _max_overlap(var, ca.fuzzy[2], cb.fuzzy[2]) < 0.5:
                        return True
                if ca.kind == cb.kind == "test" and ca.uses() and set(ca.uses()) == set(cb.uses()):
                    return True   # same quantity tested both ways; assume the author split it
        return False


def _max_overlap(var: FuzzyVariable, s1: str, s2: str) -> float:
    mf1, mf2 = var.sets[s1], var.sets[s2]
    lo = min(mf1.a, mf2.a)
    hi = max(x for x in (mf1.c, mf1.d, mf2.c, mf2.d) if x != float("inf")) + 1
    steps = 400
    return max(min(mf1.degree(x), mf2.degree(x))
               for x in (lo + (hi - lo) * i / steps for i in range(steps + 1)))


def _read_yaml(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise RuleError(f"{path.name}: invalid YAML: {e}") from None
