"""M4 delivery measures over the timed words of one answer (all of its turns).

An answer to a question can span several turns (the main answer and follow-up
answers). Each turn is measured on its own, so the silence while the interviewer
asks a follow-up is never counted as a pause or as speaking time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from ..core import TimeSpan, Transcript, Word
from ..lexicon import Lexicon, PhraseMatch, match_phrases, tokens

LONG_PAUSE_MS = 2000      # PLAN §5 M4: a gap between words longer than 2 s


@dataclass(frozen=True)
class Pause:
    at_ms: int            # when the silence starts (end of the word before it)
    ms: int

    @property
    def span(self) -> TimeSpan:
        return TimeSpan(self.at_ms, self.at_ms + self.ms)


class DeliveryAnalyzer:
    def __init__(self, turns: Transcript | Sequence[Transcript], lexicon: Lexicon | None = None,
                 nlp: Any = None, *, long_pause_ms: int = LONG_PAUSE_MS) -> None:
        turns = [turns] if isinstance(turns, Transcript) else list(turns)
        self.turns = [list(t.words) for t in turns if t.words]
        self.lexicon = lexicon or Lexicon.default()
        self.nlp = nlp
        self.long_pause_ms = long_pause_ms

    # ---------------------------------------------------------------- basic measures

    @property
    def words(self) -> list[Word]:
        return [w for turn in self.turns for w in turn]

    @property
    def span(self) -> TimeSpan | None:
        return TimeSpan.cover(TimeSpan(t[0].start_ms, t[-1].end_ms) for t in self.turns)

    def word_count(self) -> int:
        return sum(1 for w in self.words if tokens(w.text))

    def duration_ms(self) -> int:
        """Time from the first to the last word of each turn, summed over turns."""
        return sum(t[-1].end_ms - t[0].start_ms for t in self.turns)

    def speaking_ms(self) -> int:
        """Duration without the long pauses."""
        return self.duration_ms() - sum(p.ms for p in self.long_pauses())

    def pace_wpm(self) -> float:
        """Words per minute of speaking time (long pauses excluded)."""
        ms = self.speaking_ms()
        return self.word_count() * 60_000 / ms if ms > 0 else 0.0

    # ---------------------------------------------------------------- events

    def long_pauses(self) -> list[Pause]:
        out = []
        for turn in self.turns:
            for a, b in pairwise(turn):
                gap = b.start_ms - a.end_ms
                if gap > self.long_pause_ms:
                    out.append(Pause(a.end_ms, gap))
        return out

    def filler_events(self) -> list[PhraseMatch]:
        """Filler words and phrases. "like" counts only when it is used as a filler."""
        fillers = [f for f in self.lexicon["fillers"] if f.lower() != "like"]
        out: list[PhraseMatch] = []
        for turn in self.turns:
            out += match_phrases(turn, fillers)
            out += [m for m in match_phrases(turn, ["like"]) if self._like_is_filler(turn, m)]
        return sorted(out, key=lambda m: m.at_ms)

    def filler_rate(self) -> float:
        """Fillers per minute of answer duration."""
        ms = self.duration_ms()
        return len(self.filler_events()) * 60_000 / ms if ms > 0 else 0.0

    # ---------------------------------------------------------------- "like"

    def _like_is_filler(self, turn: list[Word], m: PhraseMatch) -> bool:
        if self.nlp is not None:
            return self._spacy_pos(turn, m.start) == "INTJ"
        # Without spaCy: a comma sets it off ("it was, like, hard" / "like, we").
        this = turn[m.start].text.strip()
        before = turn[m.start - 1].text.strip() if m.start > 0 else ""
        return this.endswith(",") or before.endswith(",") or m.start == 0

    def _spacy_pos(self, turn: list[Word], index: int) -> str | None:
        key = id(turn)
        cache = getattr(self, "_docs", None)
        if cache is None:
            cache = self._docs = {}
        if key not in cache:
            starts, pos = [], 0
            for w in turn:
                starts.append(pos)
                pos += len(w.text.strip()) + 1
            cache[key] = (self.nlp(" ".join(w.text.strip() for w in turn)), starts)
        doc, starts = cache[key]
        lo = starts[index]
        hi = lo + len(turn[index].text.strip())
        for tok in doc:
            if lo <= tok.idx < hi and tok.text.lower() == "like":
                return tok.pos_
        return None
