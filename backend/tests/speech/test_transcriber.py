"""M4 SpeechTranscriber on synthetic audio (macOS `say` → WebM/Opus, like Chrome records).

Whisper tests use tiny.en and only run when it is already downloaded, so CI never
downloads models. Get it once with:
    python -c "from faster_whisper import WhisperModel; WhisperModel('tiny.en')"
"""

import sys
from itertools import pairwise

import pytest
from helpers import BACKEND, FIXTURES

from starlight.core import AudioClip
from starlight.lexicon import tokens
from starlight.speech.delivery import DeliveryAnalyzer
from starlight.speech.facts import speech_facts

sys.path.insert(0, str(BACKEND / "scripts"))
from make_audio import have_say, speak

AUDIO = FIXTURES / "audio" / "synthetic"
SHORT = AUDIO / "short_answer.webm"
SHORT_TEXT = (AUDIO / "short_answer.txt").read_text()
OFFSET = 42_000


@pytest.fixture(scope="module")
def whisper():
    pytest.importorskip("faster_whisper")
    from starlight.speech.transcriber import SpeechTranscriber

    try:
        return SpeechTranscriber("tiny.en", beam_size=1, local_files_only=True)
    except Exception:  # noqa: BLE001 — any load failure (not cached, offline) means skip
        pytest.skip("Whisper tiny.en is not downloaded")


@pytest.fixture(scope="module")
def short(whisper):
    return whisper.transcribe(AudioClip(SHORT, OFFSET))


# ---------------------------------------------------------------- audio (no Whisper)


def test_synthetic_webm_decodes():
    clip = AudioClip(SHORT, OFFSET)
    assert 9_000 < clip.duration_ms() < 16_000
    assert clip.span.start_ms == OFFSET


@pytest.mark.skipif(not have_say(), reason="needs macOS `say`")
def test_speak_writes_webm_with_the_pause(tmp_path):
    a = AudioClip(speak("One two three.", tmp_path / "a.webm")).duration_ms()
    b = AudioClip(speak("One two three. [pause 2] ", tmp_path / "b.webm")).duration_ms()
    assert (tmp_path / "a.webm").read_bytes()[:4] == b"\x1a\x45\xdf\xa3"   # EBML/WebM magic
    assert 2_000 <= b - a <= 2_700          # `say` adds its own short breath after "three."


# ---------------------------------------------------------------- Whisper


def test_words_match_what_was_said(short):
    said = [t for t in tokens(SHORT_TEXT) if not t.startswith("pause") and not t.isdigit()]
    heard = tokens(short.text)
    common = set(said) & set(heard)
    assert len(common) >= 0.85 * len(set(said)), short.text


def test_timestamps_are_on_the_session_timeline(short):
    words = short.words
    assert OFFSET <= words[0].start_ms < OFFSET + 1_500
    assert words[-1].end_ms <= OFFSET + AudioClip(SHORT).duration_ms() + 200
    assert all(a.end_ms <= b.start_ms for a, b in pairwise(words))
    assert all(w.start_ms <= w.end_ms for w in words)
    assert short.meta["offset_ms"] == OFFSET and short.meta["model"] == "tiny.en"


def test_offset_shifts_every_word(whisper, short):
    other = whisper.transcribe(SHORT, offset_ms=0)
    assert [w.start_ms + OFFSET for w in other.words] == [w.start_ms for w in short.words]


def test_audio_to_delivery_facts(short):
    """Audio → Whisper → M4 facts: the 3 s silence becomes a LONG-PAUSE, "Um" a FILLER."""
    facts = speech_facts("q1", short)
    pauses = [f for f in facts if f.pred == "LONG-PAUSE"]
    assert len(pauses) == 1 and pauses[0].args[2] == pytest.approx(3_000, abs=800)
    assert any(f.pred == "FILLER" and f.args[1] in {"um", "umm"} for f in facts)
    # Pace ≈ words over the clip length minus the 3 s silence (TTS speaks fast, ~220 wpm).
    d = DeliveryAnalyzer(short)
    expected = d.word_count() * 60_000 / (AudioClip(SHORT).duration_ms() - 3_000)
    assert d.pace_wpm() == pytest.approx(expected, rel=0.2)
