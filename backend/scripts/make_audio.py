"""Synthetic test audio: speak a text with macOS `say` and save it as WebM/Opus.

WebM/Opus is what Chrome's MediaRecorder produces, so these files take the same
decode path as real recordings. They stand in until the real validation samples
are recorded (step 4.4); they are not a substitute for real voices.

Usage:
    python scripts/make_audio.py answer.txt out.webm [--voice Samantha] [--wpm 150]
    python scripts/make_audio.py --fixture strong out.webm    # text of a transcript fixture

Write [pause 2.5] in the text to insert a 2.5-second silence.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import av

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
PAUSE = re.compile(r"\[pause\s+([\d.]+)\]")
OPUS_RATE = 48_000


def have_say() -> bool:
    return shutil.which("say") is not None


def speak(text: str, dest: str | Path, *, voice: str = "Samantha", wpm: int = 150,
          bitrate: int = 32_000) -> Path:
    """Render text to a WebM/Opus file. Requires macOS `say`."""
    if not have_say():
        raise RuntimeError("make_audio needs the macOS `say` command")
    # `say` inserts silence with the embedded command [[slnc ms]].
    script = PAUSE.sub(lambda m: f" [[slnc {round(float(m.group(1)) * 1000)}]] ", text)
    with tempfile.TemporaryDirectory() as tmp:
        aiff = Path(tmp) / "speech.aiff"
        subprocess.run(["say", "-v", voice, "-r", str(wpm), "-o", str(aiff), script],
                       check=True)
        return encode_webm(aiff, dest, bitrate=bitrate)


def encode_webm(src: str | Path, dest: str | Path, *, bitrate: int = 32_000) -> Path:
    """Re-encode any audio file PyAV can read to mono WebM/Opus."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(src)) as inp, av.open(str(dest), "w", format="webm") as out:
        stream = out.add_stream("libopus", rate=OPUS_RATE, layout="mono")
        stream.bit_rate = bitrate
        resampler = av.AudioResampler(format=stream.format.name, layout="mono", rate=OPUS_RATE)
        fifo = av.AudioFifo()
        frame_size = stream.codec_context.frame_size or 960

        def drain(final: bool = False) -> None:
            while fifo.samples >= frame_size or (final and fifo.samples):
                frame = fifo.read(min(frame_size, fifo.samples) if final else frame_size)
                out.mux(stream.encode(frame))

        for frame in inp.decode(audio=0):
            for res in resampler.resample(frame):
                fifo.write(res)
            drain()
        for res in resampler.resample(None):
            fifo.write(res)
        drain(final=True)
        out.mux(stream.encode(None))
    return dest


def fixture_text(name: str) -> str:
    data = json.loads((FIXTURES / "transcripts" / f"{name}.json").read_text())
    return " ".join(w["w"] for w in data["words"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", help="text file (omit with --fixture)")
    ap.add_argument("out")
    ap.add_argument("--fixture", help="speak the words of tests/fixtures/transcripts/NAME.json")
    ap.add_argument("--voice", default="Samantha")
    ap.add_argument("--wpm", type=int, default=150)
    args = ap.parse_args()
    if bool(args.fixture) == bool(args.source):
        ap.error("give either a text file or --fixture NAME")
    text = fixture_text(args.fixture) if args.fixture else Path(args.source).read_text()
    out = speak(text, args.out, voice=args.voice, wpm=args.wpm)
    print(f"→ {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
