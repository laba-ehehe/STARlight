"""Word lists from knowledge/lexicons.yaml, and phrase matching over timed words.

Shared by M1 (job description cues), M4 (fillers) and M5 (STAR cues, hedges, ...).
Matching works on normalized tokens: lowercase, curly quotes straightened,
punctuation dropped, hyphens split ("cross-functional" → cross, functional).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from .core import TimeSpan, Word

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"

_PUNCT = re.compile(r"[^\w' ]+")


def tokens(text: str) -> list[str]:
    """Normalized tokens of a word or phrase."""
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"[\-–—/]", " ", text)
    return [t.strip("'") for t in _PUNCT.sub("", text).split() if t.strip("'")]


@dataclass(frozen=True)
class PhraseMatch:
    phrase: str           # the lexicon entry that matched, e.g. "you know"
    start: int            # index of the first matched word
    end: int              # index after the last matched word
    span: TimeSpan
    text: str             # the words as transcribed, e.g. "you know,"

    @property
    def at_ms(self) -> int:
        return self.span.start_ms


def match_phrases(words: Sequence[Word], phrases: Iterable[str]) -> list[PhraseMatch]:
    """Every occurrence of any phrase in the words, longest match first, no overlaps."""
    stream: list[tuple[str, int]] = [(t, i) for i, w in enumerate(words) for t in tokens(w.text)]
    by_first: dict[str, list[tuple[tuple[str, ...], str]]] = {}
    for phrase in phrases:
        toks = tuple(tokens(phrase))
        if toks:
            by_first.setdefault(toks[0], []).append((toks, phrase))
    for options in by_first.values():
        options.sort(key=lambda o: -len(o[0]))

    out: list[PhraseMatch] = []
    used_until = -1                        # index into stream
    for k, (tok, _) in enumerate(stream):
        if k <= used_until:
            continue
        for toks, phrase in by_first.get(tok, ()):
            n = len(toks)
            if tuple(t for t, _ in stream[k:k + n]) == toks:
                first, last = stream[k][1], stream[k + n - 1][1]
                span = TimeSpan(words[first].start_ms, words[last].end_ms)
                text = " ".join(w.text.strip() for w in words[first:last + 1])
                out.append(PhraseMatch(phrase, first, last + 1, span, text))
                used_until = k + n - 1
                break
    return out


class Lexicon:
    """The parsed lexicons.yaml. Sections are accessed by name: lex["hedges"]."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        self.data = dict(data)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Lexicon:
        path = Path(path) if path else KNOWLEDGE_DIR / "lexicons.yaml"
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    @staticmethod
    @cache
    def default() -> Lexicon:
        return Lexicon.load()

    def __getitem__(self, section: str) -> Any:
        return self.data[section]

    def get(self, section: str, default: Any = None) -> Any:
        return self.data.get(section, default)

    def find(self, words: Sequence[Word], section: str, *keys: str) -> list[PhraseMatch]:
        """Matches of one section (or a sub-list, e.g. find(words, "star_cues", "result"))."""
        node = self.data[section]
        for key in keys:
            node = node[key]
        return match_phrases(words, node)
