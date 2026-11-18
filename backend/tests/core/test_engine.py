"""Forward chaining, conflict resolution, backward chaining and explanations."""

import time

import pytest
import yaml
from helpers import answer_facts

from starlight.core import (
    ExplanationTrace,
    Fact,
    InferenceEngine,
    InferenceError,
    KnowledgeBase,
    WorkingMemory,
)

ANALYSIS = ["answer_analysis", "delivery"]


def facts_of(engine, pred):
    return sorted(str(f) for f in engine.wm.facts(pred))


# ---------------------------------------------------------------- forward chaining


def test_strong_answer_raises_no_issues(engine):
    engine.wm.assert_all(answer_facts())
    new = engine.forward_chain(ANALYSIS)
    assert new == []
    assert len(engine.trace) == 0


def test_missing_result_and_fillers(engine):
    engine.wm.assert_all(answer_facts(result=False, filler_rate=8.0, fillers=3))
    engine.forward_chain(ANALYSIS)
    assert facts_of(engine, "MISSING-RESULT") == ["(MISSING-RESULT q1)"]
    # R14b marks every filler at its own timestamp
    assert facts_of(engine, "CLARITY-ISSUE") == [
        "(CLARITY-ISSUE q1 filler 12000)", "(CLARITY-ISSUE q1 filler 17000)",
        "(CLARITY-ISSUE q1 filler 22000)"]
    spans = sorted(f.span.start_ms for f in engine.wm.facts("CLARITY-ISSUE"))
    assert spans == [12_000, 17_000, 22_000]


def test_refraction_and_repeat_calls(engine):
    engine.wm.assert_all(answer_facts(result=False))
    engine.forward_chain(ANALYSIS)
    fired = len(engine.trace)
    assert engine.forward_chain(ANALYSIS) == []      # nothing new fires twice
    assert len(engine.trace) == fired


def test_fuzzy_threshold(engine):
    engine.wm.assert_all(answer_facts(seconds=80))   # SHORT = 0.33 → does not fire
    engine.forward_chain(ANALYSIS)
    assert engine.wm.facts("TOO-SHORT") == []

    e2 = InferenceEngine(engine.kb)
    e2.wm.assert_all(answer_facts(seconds=70))       # SHORT = 0.67 → fires at 0.67
    e2.forward_chain(ANALYSIS)
    [too_short] = e2.wm.facts("TOO-SHORT")
    assert too_short.certainty == pytest.approx(2 / 3)


def test_question_planning_rules(engine):
    engine.wm.assert_all([
        Fact.of("SESSION-DURATION", 15),
        Fact.of("JD-CUE", "leadership", "mentor"),
        Fact.of("JD-CUE", "teamwork", "cross-functional"),
        Fact.of("JD-CUE", "teamwork", "stakeholders"),
        Fact.of("PLANNED", "q1", "teamwork"),
    ])
    engine.forward_chain("question_planning")
    assert facts_of(engine, "QUESTION-COUNT") == ["(QUESTION-COUNT 1)"]
    assert facts_of(engine, "QUESTION-BUDGET") == ["(QUESTION-BUDGET 600000)"]
    assert facts_of(engine, "TARGET-COMPETENCY") == [
        "(TARGET-COMPETENCY leadership HIGH)", "(TARGET-COMPETENCY teamwork HIGH)"]
    # R4: teamwork already has a question while leadership is uncovered
    assert facts_of(engine, "BLOCK-COMPETENCY") == ["(BLOCK-COMPETENCY teamwork)"]


def test_r15_over_budget(engine):
    engine.wm.assert_all([Fact.of("ELAPSED", "q1", 400_000), Fact.of("BUDGET", "q1", 360_000)])
    engine.forward_chain("session_control")
    assert facts_of(engine, "SKIP-FOLLOWUP") == ["(SKIP-FOLLOWUP q1 over-budget)"]


def test_conflict_resolution_order():
    kb = KnowledgeBase.from_dicts(yaml.safe_load("""
        rules:
          - {id: LOW,      priority: 1, if: [{fact: [A]}], then: [{assert: [LOW-FIRED]}]}
          - {id: HIGH,     priority: 9, if: [{fact: [A]}], then: [{assert: [HIGH-FIRED]}]}
          - {id: SPECIFIC, priority: 1, if: [{fact: [A]}, {fact: [B]}],
             then: [{assert: [SPECIFIC-FIRED]}]}
    """))
    engine = InferenceEngine(kb, WorkingMemory([Fact.of("A"), Fact.of("B")]))
    engine.forward_chain()
    assert [f.rule_id for f in engine.trace.firings] == ["HIGH", "SPECIFIC", "LOW"]


def test_retract_and_runaway_guard():
    kb = KnowledgeBase.from_dicts(yaml.safe_load("""
        rules:
          - {id: CONSUME, if: [{fact: [TODO, ?x]}],
             then: [{retract: [TODO, ?x]}, {assert: [DONE, ?x]}]}
          - {id: COUNT, if: [{fact: [N, ?n]}, {bind: {?m: "?n + 1"}}],
             then: [{assert: [N, ?m]}]}
    """))
    engine = InferenceEngine(kb, WorkingMemory([Fact.of("TODO", 1), Fact.of("TODO", 2)]))
    engine.forward_chain(None)
    assert facts_of(engine, "TODO") == [] and len(engine.wm.facts("DONE")) == 2

    runaway = InferenceEngine(kb, WorkingMemory([Fact.of("N", 0)]), max_cycles=50)
    with pytest.raises(InferenceError):
        runaway.forward_chain()


def test_forward_chaining_performance(kb):
    engine = InferenceEngine(kb)
    for i in range(60):                                   # 60 answers ≈ 1000 facts
        engine.wm.assert_all(answer_facts(f"q{i}", start=i * 200_000, result=i % 2 == 0,
                                          filler_rate=8.0, fillers=3, we_per_i=3))
    assert len(engine.wm) >= 900
    t = time.perf_counter()
    engine.forward_chain(ANALYSIS)
    elapsed = time.perf_counter() - t
    assert len(engine.wm.facts("MISSING-RESULT")) == 30
    assert elapsed < 2.0, f"forward chaining took {elapsed:.2f}s"


# ---------------------------------------------------------------- backward chaining


def run_followup(kb, **overrides):
    engine = InferenceEngine(kb)
    engine.wm.assert_all(answer_facts(**overrides))
    engine.forward_chain(ANALYSIS)
    return engine, engine.backward_chain(("COMPLETE", "q1"))


def test_complete_answer_is_proven(kb):
    _, result = run_followup(kb)
    assert result.proven and result.followup is None
    assert any("proved by R16" in s for s in result.steps)


def test_missing_result_asks_for_outcome(kb):
    _, result = run_followup(kb, result=False)
    assert not result.proven
    assert result.failed_subgoal == "(RESULT-STATED q1)"
    assert str(result.blocking) == "(MISSING-RESULT q1)"
    assert (result.followup, result.followup_rule) == ("What was the outcome?", "R7")


@pytest.mark.parametrize("overrides, subgoal, rule", [
    ({"relevance": 0.1, "conditional": 0.9, "result": False}, "(ON-TOPIC q1)", "R10"),
    ({"conditional": 0.9, "we_per_i": 5, "result": False}, "(REAL-EXAMPLE q1)", "R9"),
    ({"we_per_i": 5, "result": False}, "(CLEAR-OWNERSHIP q1)", "R6"),
    ({"quantified": False}, "(RESULT-SUPPORTED q1)", "R8"),
])
def test_r16_priority_order(kb, overrides, subgoal, rule):
    _, result = run_followup(kb, **overrides)
    assert result.failed_subgoal == subgoal
    assert result.followup_rule == rule
    assert result.followup == kb.rule(rule).followup


def test_backward_chaining_without_prior_forward_pass(kb):
    """Negation as failure also tries to derive the blocking fact backward."""
    engine = InferenceEngine(kb)
    engine.wm.assert_all(answer_facts(result=False))
    result = engine.backward_chain(("COMPLETE", "q1"))
    assert result.failed_subgoal == "(RESULT-STATED q1)"
    assert result.followup == "What was the outcome?"


def test_backward_chaining_unknown_answer(kb):
    engine = InferenceEngine(kb)
    result = engine.backward_chain(("COMPLETE", "q9"))
    assert not result.proven and result.followup is None


def test_backward_chaining_is_fast(kb):
    _, result = run_followup(kb, result=False)
    assert result.elapsed_ms < 50


# ---------------------------------------------------------------- explanations


def test_how_explains_back_to_observed_facts(kb):
    engine, _ = run_followup(kb, result=False)
    how = engine.trace.how(engine.wm.facts("MISSING-RESULT")[0])
    assert how["rule"] == "R7"
    assert how["because"][0]["fact"]["text"] == "(ANSWER q1)"
    assert how["because"][0]["observed_by"] == "M4"


def test_why_not_explains_missing_strength(kb):
    engine = InferenceEngine(kb)
    engine.wm.assert_all(answer_facts(result=False))
    engine.wm.assert_fact(Fact.of("STRENGTH", "q1", "organization", "clear-structure"))
    engine.wm.assert_fact(Fact.of("STRENGTH", "q1", "examples", "real-example"))
    engine.forward_chain(ANALYSIS)
    engine.forward_chain("feedback")
    # examples has no issue → it leads; organization is blocked by R7's issue
    assert [str(f) for f in engine.wm.facts("LEAD-STRENGTH")] == [
        "(LEAD-STRENGTH q1 examples real-example)"]
    why = engine.trace.why_not(["LEAD-STRENGTH", "q1", "organization", "?k"], kb, engine.wm)
    assert not why["holds"]
    [reason] = why["reasons"]
    assert reason["rule"] == "R17"
    assert reason["blocking"]["text"] == "(ISSUE q1 organization missing-result)"
    assert reason["blocking_how"]["rule"] == "R7"


def test_why_not_for_extractor_only_fact(kb):
    why = ExplanationTrace().why_not(["PACE", "q1", "?w"], kb, WorkingMemory())
    assert "only be observed by an extractor" in why["reasons"][0]["reason"]
