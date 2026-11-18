"""Every rule added in Phase 2 (R18–R53) has at least one case that makes it fire."""

import textwrap

import pytest
import yaml
from helpers import answer_facts

from starlight.core import Fact, InferenceEngine, KnowledgeBase, Rule, RuleError, TimeSpan

ANALYSIS = ["answer_analysis", "delivery"]
COMPETENCIES = ["teamwork", "leadership", "conflict", "challenges", "problem-solving"]


def run(kb, facts, *stages):
    engine = InferenceEngine(kb)
    engine.wm.assert_all(facts)
    for groups in stages:
        engine.forward_chain(groups)
    return engine


def strs(engine, pred):
    return sorted(str(f) for f in engine.wm.facts(pred))


def fired(engine):
    return {f.rule_id for f in engine.trace.firings}


def planning(minutes, cues, fmt="phone", extra=()):
    facts = [Fact.of("SESSION-DURATION", minutes), Fact.of("INTERVIEW-FORMAT", fmt)]
    facts += [Fact.of("COMPETENCY", c) for c in COMPETENCIES]
    facts += [Fact.of("JD-CUE", comp, cue) for comp, cue in cues]
    return facts + list(extra)


# ---------------------------------------------------------------- the count condition


def test_count_condition_binds_number_of_distinct_facts():
    rule = Rule.from_dict(yaml.safe_load(textwrap.dedent("""
        id: T
        if:
          - fact: [ANSWER, ?a]
          - count: [FILLER, ?a, ?_, ?_]
            as: ?n
        then: [{assert: [N, ?a, ?n]}]
    """)))
    kb = KnowledgeBase([rule])
    engine = run(kb, answer_facts(fillers=3) + answer_facts("q2"), None)
    assert strs(engine, "N") == ["(N q1 3)", "(N q2 0)"]
    # counted facts appear in the HOW explanation but do not lower the strength
    how = engine.trace.how(engine.wm.first(("N", "q1", 3)))
    assert sum("FILLER" in b["fact"]["text"] for b in how["because"]) == 3


def test_count_condition_needs_a_variable():
    with pytest.raises(RuleError):
        Rule.from_dict({"id": "X", "if": [{"count": ["A"]}], "then": []})


def test_negative_feedback_without_tip_is_a_problem(kb):
    rules = {"rules": [{"id": "X", "if": [{"fact": ["ANSWER", "?a"]}],
                        "then": [{"assert": ["TOO-SHORT", "?a"]}],
                        "feedback": {"criterion": "organization", "polarity": "negative",
                                     "text": "Too short."}}]}
    problems = KnowledgeBase.from_dicts(rules).validate()
    assert any("R52" in p.message for p in problems)


def test_rule_count(kb):
    numbered = {r.id.rstrip("abcd").split(".")[0] for r in kb}
    assert len(numbered) >= 50
    assert all(r.status for r in kb) and all(r.source for r in kb)


# ---------------------------------------------------------------- planning (R18–R24)


def test_r18_r20_and_r24_fill(kb):
    engine = run(kb, planning(45, [("conflict", "negotiate"), ("problem-solving", "debug"),
                                   ("challenges", "fast-paced")]), "question_planning")
    assert {"R18", "R19", "R20", "R24"} <= fired(engine)
    # N = 6 > 3 HIGH competencies → the other two are added as MEDIUM
    assert strs(engine, "TARGET-COMPETENCY") == [
        "(TARGET-COMPETENCY challenges HIGH)", "(TARGET-COMPETENCY conflict HIGH)",
        "(TARGET-COMPETENCY leadership MEDIUM)", "(TARGET-COMPETENCY problem-solving HIGH)",
        "(TARGET-COMPETENCY teamwork MEDIUM)"]


def test_r21_defaults_for_vague_jd(kb):
    engine = run(kb, planning(20, []), "question_planning")
    assert "R21" in fired(engine)
    assert strs(engine, "TARGET-COMPETENCY") == [
        "(TARGET-COMPETENCY challenges MEDIUM)", "(TARGET-COMPETENCY problem-solving MEDIUM)",
        "(TARGET-COMPETENCY teamwork MEDIUM)"]


def test_r21_does_not_fire_with_high_competency(kb):
    engine = run(kb, planning(15, [("teamwork", "team")]), "question_planning")
    assert "R21" not in fired(engine)


def test_r22_early_career(kb):
    engine = run(kb, planning(15, [], extra=[Fact.of("EARLY-CAREER-CUE", "intern")]),
                 "question_planning")
    assert strs(engine, "PREFER-EXAMPLES") == ["(PREFER-EXAMPLES academic)"]


def test_r23_ranks_when_more_high_than_questions(kb):
    cues = [("leadership", "lead"), ("leadership", "mentor"), ("teamwork", "team"),
            ("conflict", "negotiate")]
    engine = run(kb, planning(15, cues), "question_planning")      # N = 1, three HIGH
    assert strs(engine, "CUE-WEIGHT") == [
        "(CUE-WEIGHT conflict 1)", "(CUE-WEIGHT leadership 2)", "(CUE-WEIGHT teamwork 1)"]
    assert "R24" not in fired(engine)


# ---------------------------------------------------------------- session control (R25–R29)


@pytest.mark.parametrize("fmt, cap, rule", [("phone", 1, "R25"), ("panel", 2, "R26"),
                                            ("video", 0, "R27")])
def test_followup_cap_by_format(kb, fmt, cap, rule):
    engine = run(kb, [Fact.of("INTERVIEW-FORMAT", fmt), Fact.of("FOLLOWUPS-ASKED", "q1", 1)],
                 "session_control")
    assert strs(engine, "FOLLOWUP-CAP") == [f"(FOLLOWUP-CAP {cap})"]
    assert rule in fired(engine)
    # R28: one follow-up already asked
    assert bool(engine.wm.facts("SKIP-FOLLOWUP")) == (cap <= 1)


def test_r29_go_to_closing(kb):
    engine = run(kb, [Fact.of("TIME-LEFT", 200_000), Fact.of("NEXT-QUESTION", "q3"),
                      Fact.of("BUDGET", "q3", 400_000)], "session_control")
    assert strs(engine, "GO-TO-CLOSING") == ["(GO-TO-CLOSING q3)"]


# ---------------------------------------------------------------- analysis (R30–R40)


def answer(**overrides):
    return answer_facts(**overrides)


def with_facts(base, *facts):
    return base + list(facts)


SITUATION = TimeSpan(10_000, 35_000)


@pytest.mark.parametrize("facts, conclusion", [
    (with_facts(answer(seconds=10), Fact.of("WORD-COUNT", "q1", 18)), "(BARE-ANSWER q1)"),
    (answer(), "(WELL-STRUCTURED q1)"),
    (with_facts(answer(), Fact.of("STORY-COUNT", "q1", 2)), "(MULTIPLE-EXAMPLES q1)"),
    ([f for f in answer() if f.args[1:2] != ("situation",)]
     + [Fact.of("GENERIC-CLAIM", "q1", "I always", 12_000)], "(GENERIC-ANSWER q1)"),
    (with_facts(answer(we_per_i=1.5), Fact.of("QUESTION", "q1", "conflict"),
                Fact.of("BLAME-CUE", "q1", "his fault", 30_000)), "(BLAME-FRAMING q1)"),
    (with_facts(answer(), Fact.of("NEGATIVE-OUTCOME", "q1", "we missed the deadline")),
     "(MISSING-LESSON q1)"),
    (with_facts(answer(), Fact.of("QUESTION", "q1", "leadership")),
     "(WEAK-COMPETENCY-EVIDENCE q1)"),
    (with_facts(answer(), Fact.of("SITUATION-SHARE", "q1", 0.45)), "(LONG-SETUP q1)"),
])
def test_analysis_rules_fire(kb, facts, conclusion):
    engine = run(kb, facts, ANALYSIS)
    pred = conclusion[1:].split()[0]
    assert conclusion in strs(engine, pred)


def test_r36_not_when_lesson_stated(kb):
    facts = with_facts(answer(), Fact.of("NEGATIVE-OUTCOME", "q1", "missed"),
                       Fact.of("REFLECTION", "q1"))
    engine = run(kb, facts, ANALYSIS)
    assert engine.wm.facts("MISSING-LESSON") == []
    assert "(STRENGTH q1 examples shows-learning)" in strs(engine, "STRENGTH")   # R32


def test_r37_not_with_initiative_verb(kb):
    facts = with_facts(answer(), Fact.of("QUESTION", "q1", "leadership"),
                       Fact.of("INITIATIVE-VERB", "q1", "organized", 40_000))
    assert run(kb, facts, ANALYSIS).wm.facts("WEAK-COMPETENCY-EVIDENCE") == []


def test_r40_marks_the_situation_segment(kb):
    engine = run(kb, with_facts(answer(), Fact.of("SITUATION-SHARE", "q1", 0.45)), ANALYSIS)
    assert engine.wm.facts("LONG-SETUP")[0].span == SITUATION


# ---------------------------------------------------------------- delivery (R41–R45)


def test_r41_slow_pace(kb):
    engine = run(kb, answer(wpm=90), ANALYSIS)
    assert "(ISSUE q1 clarity slow-pace)" in strs(engine, "ISSUE")
    assert "clear-delivery" not in {f.args[2] for f in engine.wm.facts("STRENGTH")}


def test_r42_marks_each_long_pause(kb):
    pauses = [Fact.of("LONG-PAUSE", "q1", at, 2500, span=TimeSpan(at, at + 2500))
              for at in (20_000, 50_000, 80_000)]
    engine = run(kb, answer() + pauses, ANALYSIS)
    assert sorted(f.span.start_ms for f in engine.wm.facts("CLARITY-ISSUE")) == [
        20_000, 50_000, 80_000]
    assert run(kb, answer() + pauses[:2], ANALYSIS).wm.facts("CLARITY-ISSUE") == []


def test_r43_hedging_rate(kb):
    hedges = [Fact.of("HEDGE", "q1", "I think", 15_000 + i * 10_000,
                      span=TimeSpan(15_000 + i * 10_000, 15_500 + i * 10_000))
              for i in range(6)]
    engine = run(kb, answer(seconds=100) + hedges, ANALYSIS)     # 6 in 100 s = 3.6/min
    assert "(ISSUE q1 clarity hedging)" in strs(engine, "ISSUE")
    assert len(engine.wm.facts("CLARITY-ISSUE")) == 6
    assert run(kb, answer(seconds=100) + hedges[:5], ANALYSIS).wm.facts("ISSUE") == []


# ---------------------------------------------------------------- rating (R46–R50)


def ratings(kb, facts):
    engine = run(kb, facts, ANALYSIS, "rating")
    return engine, {f.args[1]: f.args[2] for f in engine.wm.facts("RATING")}


def test_strong_answer_rates_strong_everywhere(kb):
    _, r = ratings(kb, answer())
    assert r == {"relevance": "strong", "clarity": "strong", "organization": "strong",
                 "examples": "strong"}


def test_weak_answer_ratings(kb):
    facts = answer(result=False, relevance=0.5, wpm=190, filler_rate=8.0, fillers=3,
                   we_per_i=3)
    _, r = ratings(kb, facts)
    assert r == {"relevance": "adequate",        # R47a: 0.3 ≤ 0.5 < 0.6
                 "clarity": "needs-work",        # R48c: fast pace + fillers
                 "organization": "adequate",     # R49b: missing result only
                 "examples": "adequate"}         # R50b: unclear ownership only


def test_off_topic_and_unfinished_example(kb):
    _, r = ratings(kb, answer(relevance=0.1, result=False))
    assert r["relevance"] == "needs-work"        # R46
    assert r["examples"] == "adequate"           # R50d: no issue, but no result


def test_two_example_issues_need_work(kb):
    _, r = ratings(kb, answer(we_per_i=3, quantified=False))
    assert r["examples"] == "needs-work"         # R50c: ownership + unsupported result


def test_why_not_strong_names_the_blocking_issue(kb):
    engine, _ = ratings(kb, answer(result=False))
    why = engine.trace.why_not(["RATING", "q1", "organization", "strong"], kb, engine.wm)
    [reason] = why["reasons"]
    assert reason["rule"] == "R49a"
    assert reason["blocking"]["text"] == "(ISSUE q1 organization missing-result)"
    assert reason["blocking_how"]["rule"] == "R7"


# ---------------------------------------------------------------- feedback (R51, R53)


def test_r51_feedback_cap(kb):
    facts = answer(result=False, we_per_i=3, conditional=0.9, wpm=190, context="no")
    engine = run(kb, facts, ANALYSIS, "feedback")
    assert len({f.args[1:] for f in engine.wm.facts("ISSUE")}) > 3
    assert strs(engine, "FEEDBACK-CAP") == ["(FEEDBACK-CAP q1 3)"]


def test_r53_session_pattern(kb):
    facts = answer_facts("q1", result=False) + answer_facts("q2", start=300_000, result=False)
    engine = run(kb, facts, ANALYSIS, "feedback")
    assert strs(engine, "SESSION-PATTERN") == [
        "(SESSION-PATTERN organization missing-result 2)"]


# ---------------------------------------------------------------- backward chaining (R16)


def followup(kb, facts):
    engine = run(kb, facts, ANALYSIS)
    return engine.backward_chain(("COMPLETE", "q1"))


def test_bare_answer_comes_first(kb):
    result = followup(kb, with_facts(answer(seconds=8, result=False, relevance=0.1),
                                     Fact.of("WORD-COUNT", "q1", 12)))
    assert result.failed_subgoal == "(ANSWERED q1)"
    assert result.followup == "Take your time — can you walk me through it?"


def test_generic_answer_asks_for_a_specific_time(kb):
    facts = [f for f in answer() if f.args[1:2] != ("situation",)]
    facts.append(Fact.of("GENERIC-CLAIM", "q1", "I always", 12_000))
    result = followup(kb, facts)
    assert result.failed_subgoal == "(REAL-EXAMPLE q1)"
    assert result.followup_rule == "R34"


def test_missing_lesson_is_asked_last(kb):
    facts = with_facts(answer(), Fact.of("NEGATIVE-OUTCOME", "q1", "we missed it"))
    result = followup(kb, facts)
    assert (result.failed_subgoal, result.followup) == ("(LESSON-STATED q1)",
                                                        "What did you learn from it?")
    # with an earlier gap, that one wins
    facts = with_facts(answer(quantified=False), Fact.of("NEGATIVE-OUTCOME", "q1", "missed"))
    assert followup(kb, facts).followup_rule == "R8"
