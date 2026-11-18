"""Rule matching, strength, spans, failures, and knowledge-base loading/validation."""

import textwrap

import pytest
import yaml
from helpers import KNOWLEDGE, answer_facts

from starlight.core import Fact, KnowledgeBase, Rule, RuleError, TimeSpan, WorkingMemory


def rule_from_yaml(src: str) -> Rule:
    return Rule.from_dict(yaml.safe_load(textwrap.dedent(src)))


# ---------------------------------------------------------------- matching


def test_yaml_variable_quirk_is_normalized(kb):
    r7 = kb.rule("R7")
    assert r7.conditions[0].pattern == ("ANSWER", "?a")
    assert r7.conditions[1].pattern == ("STAR-SEGMENT", "?a", "result", "?_")
    r3 = kb.rule("R3")
    assert r3.conditions[1].target == "?n"
    assert kb.rule("R5").actions[0].span_from == ("?opening",)


def test_r7_matches_only_without_result(kb):
    r7 = kb.rule("R7")
    with_result = WorkingMemory(answer_facts())
    without = WorkingMemory(answer_facts(result=False))
    assert list(r7.matches(with_result, kb.fuzzy_vars)) == []
    [m] = r7.matches(without, kb.fuzzy_vars)
    assert m.bindings["?a"] == "q1"
    produced = r7.fire(m)
    assert [str(f) for f in produced] == ["(MISSING-RESULT q1)",
                                          "(ISSUE q1 organization missing-result)"]
    # span of an "absence" conclusion = the whole answer it matched
    assert produced[0].span == TimeSpan(10_000, 110_000)
    assert produced[0].source == "R7"


def test_fuzzy_strength_and_alias_span(kb):
    wm = WorkingMemory(answer_facts(seconds=75))       # SHORT to degree 0.5
    [m] = kb.rule("R12").matches(wm, kb.fuzzy_vars)
    assert kb.rule("R12").strength(m) == pytest.approx(0.5)

    wm = WorkingMemory(answer_facts(context="no"))
    [m] = kb.rule("R5").matches(wm, kb.fuzzy_vars)
    missing_context = kb.rule("R5").fire(m)[0]
    assert missing_context.span == TimeSpan(10_000, 35_000)   # the opening only


def test_strength_uses_fact_certainty():
    r = rule_from_yaml("""
        id: T1
        if: [{fact: [A, ?x]}]
        then: [{assert: [B, ?x]}]
    """)
    wm = WorkingMemory([Fact.of("A", 1, certainty=0.6)])
    [m] = r.matches(wm, {})
    assert r.strength(m) == pytest.approx(0.6)
    assert r.fire(m)[0].certainty == pytest.approx(0.6)


def test_bind_and_test_conditions(kb):
    wm = WorkingMemory([Fact.of("SESSION-DURATION", 45)])
    [m] = kb.rule("R3").matches(wm, kb.fuzzy_vars)
    assert m.bindings["?n"] == 6
    assert m.bindings["?budget"] == 400_000      # (45 - 5) min / 6 questions


def test_first_failure_reports_the_furthest_condition(kb):
    wm = WorkingMemory(answer_facts())            # has a result
    failure = kb.rule("R7").first_failure(wm, kb.fuzzy_vars)
    assert failure.index == 1
    assert failure.blocking == Fact.of("STAR-SEGMENT", "q1", "result", 3)

    wm = WorkingMemory(answer_facts(we_per_i=1.2))
    failure = kb.rule("R6").first_failure(wm, kb.fuzzy_vars)
    assert "?ratio > 2 is false" in failure.reason and "1.2" in failure.reason

    wm = WorkingMemory(answer_facts(seconds=80))  # SHORT degree 0.33 < 0.5
    failure = kb.rule("R12").first_failure(wm, kb.fuzzy_vars)
    assert failure.strength == pytest.approx(1 / 3)
    assert "below 0.5" in failure.reason

    assert kb.rule("R7").first_failure(WorkingMemory(answer_facts(result=False)),
                                       kb.fuzzy_vars) is None


@pytest.mark.parametrize("bad", [
    {"id": "X", "if": [{"fact": "notalist"}], "then": []},
    {"id": "X", "if": [{"fact": ["A"], "not": ["B"]}], "then": []},
    {"id": "X", "if": [{"fuzzy": ["?x", "pace"]}], "then": []},
    {"id": "X", "if": [], "then": [{"print": ["A"]}]},
])
def test_malformed_rules_are_rejected(bad):
    with pytest.raises(RuleError):
        Rule.from_dict(bad)


# ---------------------------------------------------------------- knowledge base


def test_real_knowledge_base_is_valid(kb):
    assert kb.validate() == []
    assert {"R1", "R7", "R16", "R16.5", "R17"} <= {r.id for r in kb}
    assert [r.id for r in kb.rules_concluding("MISSING-RESULT")] == ["R7"]
    assert all(r.group != "goal" for r in kb.rules_in())


def test_validate_catches_seeded_mistakes(kb):
    bad = yaml.safe_load(textwrap.dedent("""
        rules:
          - id: B1
            if: [{fact: [ANSWER, ?a]}, {test: "?ratio > 2"}]
            then: [{assert: [UNCLEAR-OWNERSHIP, ?a]}]
          - id: B2
            if: [{fact: [ANSWR, ?a]}]
            then: [{assert: [MISSING-RESULT, ?b]}]
          - id: B3
            if: [{fact: [ANSWER, ?a]}, {fuzzy: [?a, pace, VERY-FAST]}]
            then: [{assert: [TOO-SHORT, ?a, extra]}]
          - id: B4
            if: [{fact: [ANSWER, ?a]}]
            then: [{assert: [CLEAR-OWNERSHIP, ?a]}]
          - id: B4
            if: [{fact: [ANSWER, ?a]}]
            then: [{assert: [MISSING-RESULT, ?a], span: [?nothing]}]
    """))
    vocab = yaml.safe_load((KNOWLEDGE / "vocabulary.yaml").read_text())
    fuzzy = yaml.safe_load((KNOWLEDGE / "fuzzy.yaml").read_text())
    problems = KnowledgeBase.from_dicts(bad, fuzzy, vocab).validate()
    text = "\n".join(map(str, problems))
    assert "duplicate rule id" in text
    assert "uses unbound ['?ratio']" in text                # B1
    assert "unknown predicate ANSWR" in text                 # B2
    assert "uses unbound ['?b']" in text                     # B2
    assert "has no set 'VERY-FAST'" in text                  # B3
    assert "TOO-SHORT takes 1 args" in text                  # B3
    assert "span refers to ?nothing" in text                 # B4 (second)
    assert "may both fire" in text and "CLEAR-OWNERSHIP" in text   # B1 vs B4

