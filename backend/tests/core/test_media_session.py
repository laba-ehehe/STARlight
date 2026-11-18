"""AudioClip, TextDocument, Transcript and Session."""

from fractions import Fraction
from itertools import pairwise

import av
import numpy as np
import pytest
from helpers import FIXTURES, REPO

from starlight.core import (
    AnswerTurn,
    AudioClip,
    PlannedQuestion,
    Session,
    TextDocument,
    TimeSpan,
    Transcript,
    Word,
)


def make_webm(path, seconds=2.5, rate=48_000):
    """Write a WebM/Opus file the way a browser's MediaRecorder would."""
    with av.open(str(path), "w", format="webm") as out:
        stream = out.add_stream("libopus", rate=rate)
        stream.bit_rate = 32_000
        stream.layout = "mono"
        t = np.arange(int(seconds * rate)) / rate
        samples = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
        frame_size = 960
        for i in range(0, len(samples), frame_size):
            chunk = samples[i:i + frame_size]
            if len(chunk) < frame_size:
                chunk = np.pad(chunk, (0, frame_size - len(chunk)))
            frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = rate
            frame.pts = i
            frame.time_base = Fraction(1, rate)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return path


# ---------------------------------------------------------------- audio


def test_audio_clip_decodes_to_16k_mono(tmp_path):
    clip = AudioClip(make_webm(tmp_path / "a.webm"), offset_ms=42_000).open()
    pcm = clip.to_pcm16k()
    assert pcm.dtype == np.int16 and pcm.ndim == 1
    assert clip.duration_ms() == pytest.approx(2500, abs=80)
    assert clip.span.start_ms == 42_000
    assert clip.control("info")["duration_ms"] == clip.duration_ms()


def test_audio_clip_remux_to_ogg(tmp_path):
    clip = AudioClip(make_webm(tmp_path / "a.webm"))
    ogg = clip.control("remux", dest=tmp_path / "a.ogg")
    with av.open(str(ogg)) as c:
        stream = c.streams.audio[0]
        assert stream.codec_context.name == "opus"
        assert c.duration is not None                    # seekable, has a duration
        assert c.duration / 1_000_000 == pytest.approx(2.5, abs=0.1)
    assert AudioClip(ogg).duration_ms() == pytest.approx(clip.duration_ms(), abs=30)


def test_audio_clip_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        AudioClip(tmp_path / "nope.webm").open()


# ---------------------------------------------------------------- text


def test_text_document_cues():
    doc = TextDocument("You will lead a cross functional team, mentoring interns and "
                       "working with stakeholders. Leadership experience preferred.")
    cues = doc.find_cues({"leadership": ["lead", "mentor"],
                          "teamwork": ["cross-functional", "stakeholder"],
                          "conflict": ["negotiate"]})
    assert [c.matched for c in cues["leadership"]] == ["lead", "mentoring", "Leadership"]
    assert [c.matched for c in cues["teamwork"]] == ["cross functional", "stakeholders"]
    assert "conflict" not in cues


def test_text_document_reads_pdf():
    doc = TextDocument(path=REPO / "STARlight - Project Milestone #1.pdf").open()
    assert "Team Information" in doc.read()
    assert "behavioral" in doc.text


# ---------------------------------------------------------------- transcript


@pytest.fixture
def transcript():
    return Transcript.load(FIXTURES / "transcripts" / "strong.json")


def test_transcript_sentences_and_spans(transcript):
    sentences = transcript.sentences()
    assert len(sentences) == 8
    assert sentences[0].text.startswith("Last spring, I was")
    assert sentences[0].span.start_ms == 42_000
    assert all(a.span.end_ms <= b.span.start_ms for a, b in pairwise(sentences))
    span = transcript.span_of("search time dropped from four seconds")
    assert span is not None
    assert [w.text for w in transcript.words_in(span)] == \
        ["search", "time", "dropped", "from", "four", "seconds"]
    assert transcript.span_of("never said this") is None


def test_transcript_round_trip_and_shift(tmp_path, transcript):
    path = transcript.write(tmp_path / "t.json")
    again = Transcript.load(path)
    assert again.text == transcript.text and again.meta["question_id"] == "q1"
    shifted = again.control("shift", offset_ms=1000)
    assert shifted.words[0].start_ms == transcript.words[0].start_ms + 1000


def test_transcript_without_punctuation_splits_on_pauses():
    words = [Word("so", 0, 200), Word("i", 250, 400), Word("started", 2500, 2800),
             Word("again", 2850, 3100)]
    assert [s.text for s in Transcript(words).sentences()] == ["so i", "started again"]


def test_abbreviations_do_not_end_sentences():
    words = [Word("Tools,", 0, 100), Word("e.g.", 100, 200), Word("Git.", 200, 300),
             Word("Done.", 400, 500)]
    assert [s.text for s in Transcript(words).sentences()] == ["Tools, e.g. Git.", "Done."]


# ---------------------------------------------------------------- session


def test_session_clock_and_description():
    now = [1_000.0]
    session = Session(TextDocument("Lead a team."), "phone", 15, session_id="s1",
                      clock=lambda: now[0])
    assert session.elapsed_ms() == 0
    session.start()
    now[0] += 125_000
    assert session.elapsed_ms() == 125_000
    assert session.remaining_ms() == 15 * 60_000 - 125_000

    session.add_question(PlannedQuestion("q1", "Tell me about a time...", "teamwork", 600_000,
                                         "R2: JD mentions 'team'"))
    transcript = Transcript([Word("Hello.", 130_000, 130_400)])
    session.add_answer(AnswerTurn("q1.0", "q1", "Tell me about a time...",
                                  transcript=transcript, asked_at_ms=125_000))
    session.add_answer(AnswerTurn("q1.1", "q1", "What was the outcome?"))
    with pytest.raises(KeyError):
        session.add_answer(AnswerTurn("q9.0", "q9", "?"))

    session.end()
    now[0] += 50_000
    assert session.elapsed_ms() == 125_000                # frozen after end()

    d = session.to_description()["session"]
    assert d["format"] == "phone" and d["job_description"]["text"] == "Lead a team."
    [q] = d["questions"]
    assert q["reason"].startswith("R2")
    assert [a["followup"] for a in q["answers"]] == [False, True]
    assert q["answers"][0]["span"] == TimeSpan(130_000, 130_400).to_dict()


@pytest.mark.parametrize("fmt, minutes", [("zoom", 30), ("phone", 10), ("panel", 61)])
def test_session_rejects_bad_inputs(fmt, minutes):
    with pytest.raises(ValueError):
        Session(None, fmt, minutes)
