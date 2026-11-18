"""TimeSpan, Fact, WorkingMemory, fuzzy sets and the expression language."""

import math

import pytest

from starlight.core import Fact, FuzzyVariable, MembershipFunction, TimeSpan, WorkingMemory
from starlight.core.expr import ExpressionError, evaluate, variables_in

# ---------------------------------------------------------------- TimeSpan


def test_timespan_basics():
    a, b = TimeSpan(0, 1000), TimeSpan(3000, 4000)
    assert a.duration_ms() == 1000
    assert a.merge(b) == TimeSpan(0, 4000)
    assert not a.overlaps(b)
    assert TimeSpan(0, 1000).overlaps(TimeSpan(999, 2000))
    assert not TimeSpan(0, 1000).overlaps(TimeSpan(1000, 2000))  # touching is not overlapping
    assert TimeSpan.cover([None, b, a]) == TimeSpan(0, 4000)
    assert TimeSpan.cover([None]) is None
    with pytest.raises(ValueError):
        TimeSpan(5, 1)


# ---------------------------------------------------------------- Fact


def test_fact_matching_and_identity():
    f = Fact.of("STAR-SEGMENT", "q2", "result", 3, span=TimeSpan(0, 10), source="M5")
    assert str(f) == "(STAR-SEGMENT q2 result 3)"
    assert f.matches(["STAR-SEGMENT", "?a", "result", "?_"]) == {"?a": "q2"}
    assert f.matches(["STAR-SEGMENT", "?a", "action", "?_"]) is None
    assert f.matches(["STAR-SEGMENT", "?a", "?a", "?_"]) is None       # ?a must be consistent
    assert f.matches(["STAR-SEGMENT", "?a", "result"]) is None          # arity
    assert f.matches(["STAR-SEGMENT", "?a", "?l", "?i"], {"?a": "q1"}) is None
    # identity ignores span, certainty and source
    assert f == Fact.of("STAR-SEGMENT", "q2", "result", 3)
    assert f.with_span(TimeSpan(5, 6)).span == TimeSpan(5, 6)
    with pytest.raises(ValueError):
        Fact.of("X", certainty=1.5)


def test_working_memory():
    wm = WorkingMemory()
    assert wm.assert_fact(Fact.of("PACE", "q1", 150))
    assert not wm.assert_fact(Fact.of("PACE", "q1", 150))      # duplicate
    wm.assert_fact(Fact.of("PACE", "q2", 190))
    assert len(wm) == 2
    assert [b["?a"] for _, b in wm.query(["PACE", "?a", "?w"])] == ["q1", "q2"]
    assert wm.get("PACE", "q2", 190) is not None
    a, b = wm.facts()
    assert wm.timestamp(a) < wm.timestamp(b)
    # higher certainty replaces, but is not "new"
    assert not wm.assert_fact(Fact.of("PACE", "q1", 150, certainty=1.0))
    assert wm.retract(["PACE", "?_", "?_"]) == 2
    assert len(wm) == 0


# ---------------------------------------------------------------- fuzzy


def test_milestone_membership_functions(kb):
    dur, pace = kb.fuzzy_vars["answer_duration"], kb.fuzzy_vars["pace"]
    assert dur.degree("SHORT", 30) == 1
    assert dur.degree("SHORT", 75) == pytest.approx(0.5)
    assert dur.degree("IDEAL", 75) == pytest.approx(0.5)
    assert dur.degree("IDEAL", 120) == 1
    assert dur.degree("LONG", 165) == pytest.approx(0.5)
    assert dur.degree("LONG", 10_000) == 1
    assert pace.degree("FAST", 170) == pytest.approx(0.5)
    assert pace.degree("NORMAL", 170) == pytest.approx(1 / 3)
    assert pace.degree("SLOW", 115) == pytest.approx(0.25)    # the SLOW/NORMAL gap noted in PLAN
    assert pace.fuzzify(145) == {"SLOW": 0, "NORMAL": 1, "FAST": 0}


def test_membership_function_shapes():
    tri = MembershipFunction("T", 0, 5, 5, 10)
    assert tri.degree(5) == 1 and tri.degree(2.5) == 0.5 and tri.degree(11) == 0
    right = MembershipFunction("R", 1, 2, math.inf, math.inf)
    assert right.degree(1e9) == 1
    with pytest.raises(ValueError):
        MembershipFunction("bad", 3, 2, 4, 5)
    var = FuzzyVariable.from_dict("x", {"sets": {"LOW": [0, 0, 1, 2]}})
    with pytest.raises(KeyError):
        var.degree("HIGH", 1)


# ---------------------------------------------------------------- expressions


def test_expressions():
    b = {"?d": 15, "?r": 2.5}
    assert evaluate("floor((?d - 5) / 6)", b) == 1
    assert evaluate("?r > 2 and ?d >= 15", b) is True
    assert evaluate("max(1, ?d // 60)", b) == 1
    assert evaluate("?x == 'no'", {"?x": "no"}) is True
    assert variables_in("?elapsed > ?budget + 30000") == ["?elapsed", "?budget"]


@pytest.mark.parametrize("src", [
    "__import__('os').system('true')",
    "(1).__class__",
    "open('/etc/passwd')",
    "[x for x in (1, 2)]",
    "lambda: 1",
])
def test_expressions_reject_unsafe_code(src):
    with pytest.raises(ExpressionError):
        evaluate(src, {})


def test_expression_unbound_variable():
    with pytest.raises(ExpressionError, match="unbound variable"):
        evaluate("?nope > 1", {})
