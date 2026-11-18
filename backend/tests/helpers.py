"""Shared test helpers (importable as `helpers`; tests/ is on pythonpath)."""

from __future__ import annotations

import sys
from pathlib import Path

from starlight.core import Fact, TimeSpan, Transcript

BACKEND = Path(__file__).resolve().parents[1]
KNOWLEDGE = BACKEND / "knowledge"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPO = BACKEND.parent

sys.path.insert(0, str(BACKEND / "scripts"))
from make_fixture import build


def spoken(text: str, offset_ms: int = 0, wpm: float = 150.0, qid: str = "q1") -> Transcript:
    """Transcript with synthetic word timings; write [pause 2.5] for a silence."""
    return Transcript.from_dict(build(text, qid=qid, question="", competency="teamwork",
                                      offset_ms=offset_ms, wpm=wpm))


def answer_facts(qid: str = "q1", start: int = 10_000, **overrides) -> list[Fact]:
    """M4/M5 facts for a strong 100-second answer. Override any measure by keyword.

    Pass result=False to drop the Result segment, quantified=False to drop the number,
    fillers=n to add n filler words.
    """
    end = start + int(overrides.get("seconds", 100) * 1000)
    whole = TimeSpan(start, end)
    third = (end - start) // 4

    def seg(i: int) -> TimeSpan:
        return TimeSpan(start + i * third, start + (i + 1) * third)

    facts = [
        Fact.of("ANSWER", qid, span=whole, source="M4"),
        Fact.of("ANSWER-DURATION", qid, overrides.get("seconds", 100), span=whole, source="M4"),
        Fact.of("PACE", qid, overrides.get("wpm", 145), span=whole, source="M4"),
        Fact.of("FILLER-RATE", qid, overrides.get("filler_rate", 1.0), source="M4"),
        Fact.of("CONTEXT-IN-OPENING", qid, overrides.get("context", "yes"), span=seg(0),
                source="M5"),
        Fact.of("PRONOUN-RATIO", qid, overrides.get("we_per_i", 0.5), span=seg(2), source="M5"),
        Fact.of("ACTION-SHARE", qid, overrides.get("action_share", 0.5), source="M5"),
        Fact.of("CONDITIONAL-SHARE", qid, overrides.get("conditional", 0.0), span=seg(2),
                source="M5"),
        Fact.of("RELEVANCE", qid, overrides.get("relevance", 0.7), source="M5"),
        Fact.of("STAR-SEGMENT", qid, "situation", 0, span=seg(0), source="M5"),
        Fact.of("STAR-SEGMENT", qid, "task", 1, span=seg(1), source="M5"),
        Fact.of("STAR-SEGMENT", qid, "action", 2, span=seg(2), source="M5"),
    ]
    if overrides.get("result", True):
        facts.append(Fact.of("STAR-SEGMENT", qid, "result", 3, span=seg(3), source="M5"))
        if overrides.get("quantified", True):
            facts.append(Fact.of("QUANTIFIED-RESULT", qid, "300 milliseconds",
                                 span=TimeSpan(seg(3).start_ms + 500, seg(3).start_ms + 1500),
                                 source="M5"))
    for i in range(overrides.get("fillers", 0)):
        at = start + 2000 + i * 5000
        facts.append(Fact.of("FILLER", qid, "um", at, span=TimeSpan(at, at + 300), source="M4"))
    return facts
