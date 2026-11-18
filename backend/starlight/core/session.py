"""One practice session: its inputs, its questions and answers, and its clock."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .media import AudioClip, TextDocument, Transcript
from .timespan import TimeSpan

FORMATS = ("phone", "video", "panel")
MIN_DURATION, MAX_DURATION = 15, 60


@dataclass
class PlannedQuestion:
    qid: str
    text: str
    competency: str
    budget_ms: int
    reason: str = ""          # e.g. "R2: JD mentions 'cross-functional'"

    def to_dict(self) -> dict:
        return {"id": self.qid, "text": self.text, "competency": self.competency,
                "budget_ms": self.budget_ms, "reason": self.reason}


@dataclass
class AnswerTurn:
    """One recorded turn: the main answer (q2.0) or a follow-up answer (q2.1, q2.2)."""
    turn_id: str
    qid: str
    prompt: str                         # what the student was asked in this turn
    clip: AudioClip | None = None
    transcript: Transcript | None = None
    asked_at_ms: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_followup(self) -> bool:
        return not self.turn_id.endswith(".0")

    @property
    def span(self) -> TimeSpan | None:
        if self.transcript is not None and self.transcript.span is not None:
            return self.transcript.span
        if self.clip is not None:
            return self.clip.span
        return None

    def to_dict(self) -> dict:
        span = self.span
        return {
            "turn": self.turn_id,
            "prompt": self.prompt,
            "followup": self.is_followup,
            "asked_at_ms": self.asked_at_ms,
            "span": span.to_dict() if span else None,
            "audio": str(self.clip.path) if self.clip else None,
            "audio_offset_ms": self.clip.offset_ms if self.clip else None,
            "transcript": self.transcript.to_dict() if self.transcript else None,
            **self.extra,
        }


class Session:
    def __init__(self, job_description: TextDocument | None, interview_format: str,
                 duration_min: int, *, session_id: str | None = None,
                 clock: Callable[[], float] | None = None) -> None:
        if interview_format not in FORMATS:
            raise ValueError(f"interview format must be one of {FORMATS}")
        if not MIN_DURATION <= duration_min <= MAX_DURATION:
            raise ValueError(f"duration must be {MIN_DURATION}-{MAX_DURATION} minutes")
        self.id = session_id or uuid.uuid4().hex[:12]
        self.job_description = job_description
        self.interview_format = interview_format
        self.duration_min = int(duration_min)
        self.created_at = datetime.now(UTC)
        self.questions: list[PlannedQuestion] = []
        self.answers: list[AnswerTurn] = []
        self._clock = clock or (lambda: time.monotonic() * 1000)
        self._started_ms: float | None = None
        self._ended_ms: float | None = None

    # ---------------------------------------------------------------- clock

    def start(self) -> None:
        if self._started_ms is None:
            self._started_ms = self._clock()

    def end(self) -> None:
        if self._ended_ms is None:
            self._ended_ms = self._clock()

    @property
    def duration_ms(self) -> int:
        return self.duration_min * 60_000

    def elapsed_ms(self) -> int:
        if self._started_ms is None:
            return 0
        now = self._ended_ms if self._ended_ms is not None else self._clock()
        return int(now - self._started_ms)

    def remaining_ms(self) -> int:
        return max(0, self.duration_ms - self.elapsed_ms())

    # ---------------------------------------------------------------- content

    def add_question(self, question: PlannedQuestion) -> None:
        self.questions.append(question)

    def add_answer(self, turn: AnswerTurn) -> None:
        if not any(q.qid == turn.qid for q in self.questions):
            raise KeyError(f"no planned question {turn.qid!r}")
        self.answers.append(turn)

    def answers_for(self, qid: str) -> list[AnswerTurn]:
        return [a for a in self.answers if a.qid == qid]

    def question(self, qid: str) -> PlannedQuestion:
        for q in self.questions:
            if q.qid == qid:
                return q
        raise KeyError(qid)

    def to_description(self) -> dict:
        """Session description (MPEG-7 style): Session DS → Question DS → Answer DS."""
        jd = self.job_description
        return {
            "session": {
                "id": self.id,
                "created_at": self.created_at.isoformat(),
                "format": self.interview_format,
                "duration_min": self.duration_min,
                "elapsed_ms": self.elapsed_ms(),
                "job_description": {
                    "source": str(jd.path) if jd and jd.path else "text",
                    "text": jd.text if jd else "",
                },
                "questions": [
                    {**q.to_dict(), "answers": [a.to_dict() for a in self.answers_for(q.qid)]}
                    for q in self.questions
                ],
            }
        }
