"""M4 → working memory: turn delivery measures into facts (PLAN §4, speech facts)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core import Fact, Transcript
from ..lexicon import Lexicon
from .delivery import DeliveryAnalyzer

SOURCE = "M4"
MIN_WORDS_FOR_PACE = 10     # pace is meaningless for a handful of words


def speech_facts(qid: str, turns: Transcript | Sequence[Transcript],
                 lexicon: Lexicon | None = None, nlp: Any = None,
                 analyzer: DeliveryAnalyzer | None = None) -> list[Fact]:
    """ANSWER, ANSWER-DURATION, WORD-COUNT, PACE, FILLER, FILLER-RATE and LONG-PAUSE facts.

    turns: every recorded turn of the question (main answer + follow-up answers).
    """
    d = analyzer or DeliveryAnalyzer(turns, lexicon, nlp)
    span = d.span

    def fact(*terms: Any, **kw: Any) -> Fact:
        return Fact.of(*terms, source=SOURCE, **{"span": span, **kw})

    facts = [
        fact("ANSWER", qid),
        fact("ANSWER-DURATION", qid, round(d.duration_ms() / 1000, 1)),
        fact("WORD-COUNT", qid, d.word_count()),
    ]
    if d.word_count() >= MIN_WORDS_FOR_PACE:
        facts.append(fact("PACE", qid, round(d.pace_wpm())))
    fillers = d.filler_events()
    facts.append(fact("FILLER-RATE", qid, round(d.filler_rate(), 2)))
    facts += [fact("FILLER", qid, m.phrase.lower(), m.at_ms, span=m.span) for m in fillers]
    facts += [fact("LONG-PAUSE", qid, p.at_ms, p.ms, span=p.span) for p in d.long_pauses()]
    return facts
