"""M4 speech recognition: AudioClip → Transcript on the session timeline (PLAN D4, D5).

faster-whisper is imported lazily so the rest of the package (and CI) works
without the `speech` extra installed.
"""

from __future__ import annotations

import os
import time
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from ..core import AudioClip, Transcript, Word

# Live pass: fast model, greedy decoding (decision ≤ 5 s). Post-session pass: accuracy.
# Defaults come from the benchmark (docs/benchmark.md); override per machine with
# STARLIGHT_LIVE_MODEL / STARLIGHT_POST_MODEL / STARLIGHT_WHISPER_DEVICE (cpu, mps, cuda).
LIVE_MODEL, LIVE_BEAM = os.environ.get("STARLIGHT_LIVE_MODEL", "tiny.en"), 1
POST_MODEL, POST_BEAM = os.environ.get("STARLIGHT_POST_MODEL", "small.en"), 5
DEVICE = os.environ.get("STARLIGHT_WHISPER_DEVICE", "cpu")
# Whisper tends to drop "um"/"uh"; a prompt full of fillers makes it keep them (D5).
FILLER_PROMPT = "Umm, so, uh, I was, like, working on... you know, hmm, I mean, let me think."


class SpeechTranscriber:
    """Wraps one loaded WhisperModel. Load it once at startup and reuse it."""

    def __init__(self, model_name: str = LIVE_MODEL, *, device: str = DEVICE,
                 compute_type: str | None = None, cpu_threads: int = 0, beam_size: int = 5,
                 download_root: str | Path | None = None, local_files_only: bool = False) -> None:
        from faster_whisper import WhisperModel

        self.model_name = model_name
        self.beam_size = beam_size
        compute_type = compute_type or ("int8" if device == "cpu" else "float16")
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type,
                                  cpu_threads=cpu_threads, local_files_only=local_files_only,
                                  download_root=str(download_root) if download_root else None)

    @staticmethod
    @cache
    def shared(model_name: str = LIVE_MODEL, beam_size: int = 5) -> SpeechTranscriber:
        """One process-wide instance per (model, beam size)."""
        return SpeechTranscriber(model_name, beam_size=beam_size)

    @staticmethod
    def live() -> SpeechTranscriber:
        return SpeechTranscriber.shared(LIVE_MODEL, LIVE_BEAM)

    @staticmethod
    def post_session() -> SpeechTranscriber:
        return SpeechTranscriber.shared(POST_MODEL, POST_BEAM)

    def transcribe(self, clip: AudioClip | str | Path, offset_ms: int | None = None, *,
                   initial_prompt: str | None = FILLER_PROMPT, **options: Any) -> Transcript:
        """Words with session-time timestamps (clip time + clip.offset_ms)."""
        if not isinstance(clip, AudioClip):
            clip = AudioClip(clip, offset_ms or 0)
        offset = clip.offset_ms if offset_ms is None else int(offset_ms)
        audio = clip.open().to_pcm16k().astype(np.float32) / 32768.0

        started = time.perf_counter()
        options.setdefault("beam_size", self.beam_size)
        segments, info = self.model.transcribe(
            audio, language="en", word_timestamps=True, vad_filter=True,
            initial_prompt=initial_prompt, condition_on_previous_text=False, **options)
        words = [Word(w.word.strip(), offset + round(w.start * 1000),
                      offset + round(w.end * 1000), float(w.probability))
                 for seg in segments for w in (seg.words or []) if w.word.strip()]
        elapsed_ms = round((time.perf_counter() - started) * 1000)

        meta = {"offset_ms": offset, "audio_ms": round(info.duration * 1000),
                "model": self.model_name, "beam_size": options["beam_size"],
                "transcribe_ms": elapsed_ms, "source": str(clip.path)}
        return Transcript(_monotonic(words), meta=meta)


def _monotonic(words: list[Word]) -> list[Word]:
    """Whisper occasionally overlaps neighbouring words by a few ms; clamp so time only moves on."""
    out: list[Word] = []
    for w in words:
        start = max(w.start_ms, out[-1].end_ms) if out else w.start_ms
        out.append(Word(w.text, start, max(start, w.end_ms), w.probability))
    return out
