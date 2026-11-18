"""Whisper benchmark (IMPLEMENTATION 0.4): how long each model takes per answer.

Usage:
    python scripts/bench_whisper.py rec60.webm rec120.webm         # real recordings
    python scripts/bench_whisper.py --synthetic                    # macOS `say` clips of ~60/120 s
    python scripts/bench_whisper.py --synthetic --out ../docs/benchmark.md

Each model is loaded once and warmed up on a short clip, so the times are what
the live server sees after startup. With --synthetic the spoken text is known,
so the word error rate (WER) is printed too.

Decision rule: the live model is the largest one that transcribes 120 s of audio
in ≤ 3.5 s, leaving ≈ 1.5 s of the 5 s budget for the rules.
"""

from __future__ import annotations

import argparse
import difflib
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make_audio import fixture_text, speak

from starlight.core import AudioClip
from starlight.lexicon import tokens
from starlight.speech.transcriber import SpeechTranscriber

MODELS = ["tiny.en", "base.en", "small.en"]
BUDGET_S = 3.5
SYNTHETIC_S = (60, 120)
FIXTURE_NAMES = ["strong", "no_result", "we_heavy", "hypothetical", "off_topic"]


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = tokens(reference), tokens(hypothesis)
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    errors = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes()
                 if op != "equal")
    return errors / len(ref) if ref else 0.0


def synthetic_clip(seconds: int, dest: Path) -> tuple[Path, str]:
    """Answer text from the fixtures, trimmed so `say` speaks it for about `seconds`."""
    pool = " ".join(fixture_text(n) for n in FIXTURE_NAMES).split()
    n = round(seconds * 170 / 60)                       # `say` speaks ≈ 170 wpm
    for _ in range(3):
        text = " ".join((pool * 3)[:n])
        speak(text, dest)
        got = AudioClip(dest).duration_ms() / 1000
        if abs(got - seconds) <= 3:
            break
        n = round(n * seconds / got)
    return dest, text


def cpu_name() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or platform.machine()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="*")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--device", default="cpu", help="cpu, mps (Apple GPU) or cuda")
    ap.add_argument("--beams", nargs="+", type=int, default=[1, 5],
                    help="beam sizes to try (1 = greedy, the live setting)")
    ap.add_argument("--repeat", type=int, default=3, help="runs per file (median is reported)")
    ap.add_argument("--out", help="also write the markdown table here")
    args = ap.parse_args()

    tmp = tempfile.TemporaryDirectory()
    clips: list[tuple[str, Path, str | None]] = [(Path(p).name, Path(p), None) for p in args.audio]
    if args.synthetic:
        for secs in SYNTHETIC_S:
            path, text = synthetic_clip(secs, Path(tmp.name) / f"{secs}s.webm")
            clips.append((f"synthetic {secs}s", path, text))
    if not clips:
        ap.error("give audio files or --synthetic")
    warm = speak("Warming up the model.", Path(tmp.name) / "warm.webm")

    rows = []
    for name in args.models:
        t0 = time.perf_counter()
        tr = SpeechTranscriber(name, device=args.device)
        load_s = time.perf_counter() - t0
        tr.transcribe(warm)
        for beam in args.beams:
            for label, path, text in clips:
                audio_s = AudioClip(path).duration_ms() / 1000
                runs, result = [], None
                for _ in range(args.repeat):
                    t = time.perf_counter()
                    result = tr.transcribe(path, beam_size=beam)
                    runs.append(time.perf_counter() - t)
                secs = statistics.median(runs)
                row = {"model": name, "beam": beam, "clip": label, "audio_s": audio_s,
                       "secs": secs, "rtf": secs / audio_s, "words": len(result.words),
                       "load_s": load_s, "wer": wer(text, result.text) if text else None}
                rows.append(row)
                print(f"{name:9} beam {beam}  {label:15} audio {audio_s:6.1f}s  "
                      f"transcribe {secs:5.2f}s  x{row['rtf']:.3f}  words {row['words']}"
                      + (f"  WER {row['wer']:.1%}" if text else ""), flush=True)

    table = render(rows, args)
    print("\n" + table)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table)
        print(f"→ {args.out}")
    tmp.cleanup()


def render(rows: list[dict], args: argparse.Namespace) -> str:
    longest = max(r["audio_s"] for r in rows)
    ok = [(r["model"], r["beam"]) for r in rows if r["audio_s"] == longest
          and r["secs"] * 120 / longest <= BUDGET_S]
    pick = next((f"{m}, beam {b}" for m in reversed(args.models) for b in args.beams
                 if (m, b) in ok), None)
    lines = [
        "| Model | Beam | Clip | Audio (s) | Transcribe (s) | × real time | Words | WER |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        w = f"{r['wer']:.1%}" if r["wer"] is not None else "—"
        lines.append(f"| {r['model']} | {r['beam']} | {r['clip']} | {r['audio_s']:.1f} | "
                     f"{r['secs']:.2f} | {r['rtf']:.3f} | {r['words']} | {w} |")
    machine = (f"{cpu_name()}, {os.cpu_count()} cores, {platform.system()} "
               f"{platform.release()}, Python {platform.python_version()}, device {args.device}")
    verdict = (f"**Live model by the rule (≤ {BUDGET_S} s per 120 s answer): `{pick}`**"
               if pick else f"**No model meets ≤ {BUDGET_S} s per 120 s answer on this machine.**")
    return "\n".join([(f"Machine: {machine}. Word timestamps, VAD on, filler prompt; "
                      f"median of {args.repeat} runs after a warm-up."), "",
                      *lines, "", verdict, ""])


if __name__ == "__main__":
    main()
