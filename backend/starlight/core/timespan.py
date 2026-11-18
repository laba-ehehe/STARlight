"""Time spans on the shared session timeline.

Every time in STARlight is an integer number of milliseconds measured from the
start of the session. A span is half-open: [start_ms, end_ms).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, order=True)
class TimeSpan:
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if self.end_ms < self.start_ms:
            raise ValueError(f"TimeSpan ends before it starts: {self.start_ms} > {self.end_ms}")

    def overlaps(self, other: TimeSpan) -> bool:
        """True if the spans share any time. Touching spans do not overlap."""
        return self.start_ms < other.end_ms and other.start_ms < self.end_ms

    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def merge(self, other: TimeSpan) -> TimeSpan:
        """Smallest span covering both spans (gaps between them included)."""
        return TimeSpan(min(self.start_ms, other.start_ms), max(self.end_ms, other.end_ms))

    @staticmethod
    def cover(spans: Iterable[TimeSpan | None]) -> TimeSpan | None:
        """Merge all non-None spans; None if there are none."""
        result: TimeSpan | None = None
        for span in spans:
            if span is not None:
                result = span if result is None else result.merge(span)
        return result

    def shifted(self, offset_ms: int) -> TimeSpan:
        return TimeSpan(self.start_ms + offset_ms, self.end_ms + offset_ms)

    def to_dict(self) -> dict:
        return {"start_ms": self.start_ms, "end_ms": self.end_ms}

    @classmethod
    def from_dict(cls, d: dict) -> TimeSpan:
        return cls(int(d["start_ms"]), int(d["end_ms"]))

    def __str__(self) -> str:
        return f"[{_fmt(self.start_ms)}–{_fmt(self.end_ms)}]"


def _fmt(ms: int) -> str:
    seconds, millis = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}:{seconds:02d}.{millis // 100}"
