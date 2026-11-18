"""Media objects (Mercuri-style): every medium supports OPEN, READ, WRITE, CONTROL, CLOSE.

AudioClip     one recorded answer turn (WebM/Opus from the browser, or any file PyAV reads)
TextDocument  a job description (plain text or PDF) or a question text
Transcript    words with timestamps on the session timeline
"""

from __future__ import annotations

import json
import re
import shutil
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

from .timespan import TimeSpan

PCM_RATE = 16_000


class MediaObject(ABC):
    """Base for every medium. Subclasses override the five polymorphic operations."""

    @abstractmethod
    def open(self) -> MediaObject: ...

    @abstractmethod
    def read(self) -> Any: ...

    @abstractmethod
    def write(self, dest: str | Path) -> Path: ...

    @abstractmethod
    def control(self, command: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()


# ====================================================================== audio

class AudioClip(MediaObject):
    """One recorded answer turn.

    offset_ms is where the recording starts on the session timeline, so a time t
    inside the file corresponds to session time offset_ms + t.
    """

    def __init__(self, path: str | Path, offset_ms: int = 0) -> None:
        self.path = Path(path)
        self.offset_ms = int(offset_ms)
        self._pcm: np.ndarray | None = None

    def open(self) -> AudioClip:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        return self

    def read(self) -> np.ndarray:
        return self.to_pcm16k()

    def write(self, dest: str | Path) -> Path:
        dest = Path(dest)
        shutil.copyfile(self.path, dest)
        return dest

    def control(self, command: str, **kwargs: Any) -> Any:
        match command:
            case "remux":
                return self.remux(kwargs["dest"])
            case "info":
                return {"path": str(self.path), "offset_ms": self.offset_ms,
                        "duration_ms": self.duration_ms()}
        raise ValueError(f"AudioClip does not support control {command!r}")

    def close(self) -> None:
        self._pcm = None

    def to_pcm16k(self) -> np.ndarray:
        """Decode to 16 kHz, 16-bit, mono PCM (what the speech recognizer expects)."""
        if self._pcm is None:
            import av

            chunks: list[np.ndarray] = []
            resampler = av.AudioResampler(format="s16", layout="mono", rate=PCM_RATE)
            with av.open(str(self.path)) as container:
                stream = container.streams.audio[0]
                for frame in container.decode(stream):
                    for out in resampler.resample(frame):
                        chunks.append(out.to_ndarray().reshape(-1))
                for out in resampler.resample(None):
                    chunks.append(out.to_ndarray().reshape(-1))
            self._pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        return self._pcm

    def duration_ms(self) -> int:
        """Measured by decoding: browser WebM files often carry no duration header."""
        return round(len(self.to_pcm16k()) * 1000 / PCM_RATE)

    @property
    def span(self) -> TimeSpan:
        return TimeSpan(self.offset_ms, self.offset_ms + self.duration_ms())

    def remux(self, dest: str | Path) -> Path:
        """Copy the Opus stream into an Ogg file without re-encoding.

        MediaRecorder's WebM has no duration or cue index, so browsers seek in it
        poorly. The Ogg copy is seekable and keeps the 32 kbps compressed audio.
        """
        import av

        dest = Path(dest)
        with av.open(str(self.path)) as src, av.open(str(dest), "w", format="ogg") as out:
            in_stream = src.streams.audio[0]
            if in_stream.codec_context.name != "opus":
                raise ValueError(f"remux to Ogg needs Opus audio, got {in_stream.codec_context.name}")
            out_stream = out.add_stream_from_template(in_stream)
            for packet in src.demux(in_stream):
                if packet.dts is None:
                    continue
                packet.stream = out_stream
                out.mux(packet)
        return dest


# ====================================================================== text

@dataclass(frozen=True)
class Cue:
    competency: str
    cue: str
    matched: str
    start: int   # character offsets in the document text
    end: int


class TextDocument(MediaObject):
    """A job description (pasted text or PDF) or a question text."""

    def __init__(self, text: str | None = None, path: str | Path | None = None) -> None:
        if text is None and path is None:
            raise ValueError("TextDocument needs text or a path")
        self.path = Path(path) if path is not None else None
        self._text = text

    def open(self) -> TextDocument:
        if self._text is None:
            self._text = self.extract_text()
        return self

    def read(self) -> str:
        return self.text

    def write(self, dest: str | Path) -> Path:
        dest = Path(dest)
        dest.write_text(self.text, encoding="utf-8")
        return dest

    def control(self, command: str, **kwargs: Any) -> Any:
        if command == "find_cues":
            return self.find_cues(kwargs["lexicon"])
        raise ValueError(f"TextDocument does not support control {command!r}")

    def close(self) -> None:
        pass

    @property
    def text(self) -> str:
        if self._text is None:
            self._text = self.extract_text()
        return self._text

    def extract_text(self) -> str:
        if self._text is not None:
            return self._text
        assert self.path is not None
        if self.path.suffix.lower() == ".pdf":
            import pdfplumber

            with pdfplumber.open(self.path) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages).strip()
        return self.path.read_text(encoding="utf-8", errors="replace").strip()

    def find_cues(self, lexicon: Mapping[str, Sequence[str]]) -> dict[str, list[Cue]]:
        """Find cue words per competency, tolerant of simple inflections and hyphens.

        lexicon: {"leadership": ["lead", "mentor", "own"], "teamwork": ["cross-functional"]}
        """
        found: dict[str, list[Cue]] = {}
        for competency, cues in lexicon.items():
            hits = []
            for cue in cues:
                for m in _cue_regex(cue).finditer(self.text):
                    hits.append(Cue(competency, cue, m.group(0), m.start(), m.end()))
            if hits:
                found[competency] = sorted(hits, key=lambda c: c.start)
        return found


def _cue_regex(cue: str) -> re.Pattern:
    parts = [re.escape(p) for p in re.split(r"[\s\-]+", cue.strip()) if p]
    body = r"[\s\-]?".join(parts)
    return re.compile(rf"\b{body}(?:s|es|ed|d|ing|ership)?\b", re.IGNORECASE)


# ====================================================================== transcript

@dataclass(frozen=True)
class Word:
    text: str
    start_ms: int
    end_ms: int
    probability: float = 1.0

    @property
    def span(self) -> TimeSpan:
        return TimeSpan(self.start_ms, self.end_ms)

    @property
    def norm(self) -> str:
        return _normalize(self.text)

    def to_dict(self) -> dict:
        return {"w": self.text, "start_ms": self.start_ms, "end_ms": self.end_ms,
                "p": round(self.probability, 3)}


@dataclass(frozen=True)
class Sentence:
    index: int
    words: tuple[Word, ...]

    @property
    def text(self) -> str:
        return _join(self.words)

    @property
    def span(self) -> TimeSpan:
        return TimeSpan(self.words[0].start_ms, self.words[-1].end_ms)

    def to_dict(self) -> dict:
        return {"index": self.index, "text": self.text, "span": self.span.to_dict()}


_ABBREVIATIONS = {"e.g.", "i.e.", "dr.", "mr.", "mrs.", "ms.", "u.s.", "etc.", "vs.", "st."}
_SENTENCE_END = re.compile(r"[.?!][\"')\]]*$")


class Transcript(MediaObject):
    """Words with timestamps on the session timeline."""

    def __init__(self, words: Sequence[Word] = (), path: str | Path | None = None,
                 meta: Mapping[str, Any] | None = None) -> None:
        self.words = list(words)
        self.path = Path(path) if path is not None else None
        self.meta = dict(meta or {})

    # ---------------------------------------------------------------- media operations

    def open(self) -> Transcript:
        if self.path is not None and not self.words:
            loaded = Transcript.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
            self.words, self.meta = loaded.words, loaded.meta
        return self

    def read(self) -> list[Word]:
        return self.words

    def write(self, dest: str | Path) -> Path:
        dest = Path(dest)
        dest.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return dest

    def control(self, command: str, **kwargs: Any) -> Any:
        if command == "shift":
            return self.shifted(kwargs["offset_ms"])
        raise ValueError(f"Transcript does not support control {command!r}")

    def close(self) -> None:
        pass

    # ---------------------------------------------------------------- queries

    @property
    def text(self) -> str:
        return _join(self.words)

    @property
    def span(self) -> TimeSpan | None:
        if not self.words:
            return None
        return TimeSpan(self.words[0].start_ms, self.words[-1].end_ms)

    def words_in(self, span: TimeSpan) -> list[Word]:
        """Words whose time overlaps the span (zero-length words count if inside it)."""
        return [w for w in self.words
                if w.span.overlaps(span) or span.start_ms <= w.start_ms < span.end_ms]

    def sentences(self, max_gap_ms: int = 1500) -> list[Sentence]:
        """Split on sentence-ending punctuation (Whisper punctuates its output).

        If the transcript has no punctuation at all, split on pauses longer than
        max_gap_ms instead.
        """
        if not self.words:
            return []
        punctuated = any(_SENTENCE_END.search(w.text) for w in self.words)
        groups: list[list[Word]] = [[]]
        for i, w in enumerate(self.words):
            if groups[-1] and not punctuated \
                    and w.start_ms - self.words[i - 1].end_ms > max_gap_ms:
                groups.append([])
            groups[-1].append(w)
            if punctuated and _SENTENCE_END.search(w.text) \
                    and w.text.lower().strip("\"')]") not in _ABBREVIATIONS:
                groups.append([])
        return [Sentence(i, tuple(g)) for i, g in enumerate(x for x in groups if x)]

    def span_of(self, text: str) -> TimeSpan | None:
        """Time span of the first occurrence of a word sequence (case/punctuation-insensitive)."""
        target = [t for t in (_normalize(x) for x in text.split()) if t]
        if not target:
            return None
        norms = [w.norm for w in self.words]
        n = len(target)
        for i in range(len(norms) - n + 1):
            if norms[i:i + n] == target:
                return TimeSpan(self.words[i].start_ms, self.words[i + n - 1].end_ms)
        return None

    def shifted(self, offset_ms: int) -> Transcript:
        return Transcript([Word(w.text, w.start_ms + offset_ms, w.end_ms + offset_ms,
                                w.probability) for w in self.words], self.path, self.meta)

    # ---------------------------------------------------------------- (de)serialization

    def to_dict(self) -> dict:
        return {**self.meta, "words": [w.to_dict() for w in self.words]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Transcript:
        words = [Word(w["w"], int(w["start_ms"]), int(w["end_ms"]), float(w.get("p", 1.0)))
                 for w in d.get("words", [])]
        meta = {k: v for k, v in d.items() if k != "words"}
        return cls(words, meta=meta)

    @classmethod
    def load(cls, path: str | Path) -> Transcript:
        return cls(path=path).open()

    def __len__(self) -> int:
        return len(self.words)


def _normalize(token: str) -> str:
    return re.sub(r"[^\w']+", "", token.lower())


def _join(words: Sequence[Word]) -> str:
    return " ".join(w.text.strip() for w in words if w.text.strip())
