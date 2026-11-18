"""Turn a plain-text answer into a transcript fixture with synthetic word timings.

Usage:
    python scripts/make_fixture.py answer.txt out.json --question "..." \
        --competency teamwork --qid q1 --offset-ms 42000 --wpm 150

Write [pause 2.5] in the text to insert a 2.5-second silence.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PAUSE = re.compile(r"\[pause\s+([\d.]+)\]")


def build(text: str, *, qid: str, question: str, competency: str, offset_ms: int = 0,
          wpm: float = 150.0) -> dict:
    ms_per_word = 60_000 / wpm
    t = float(offset_ms)
    words = []
    for token in re.split(r"(\[pause\s+[\d.]+\])|\s+", text.strip()):
        if not token:
            continue
        pause = PAUSE.fullmatch(token)
        if pause:
            t += float(pause.group(1)) * 1000
            continue
        # Longer words take a little longer to say; punctuation adds a short breath.
        length = ms_per_word * (0.7 + 0.06 * min(len(token), 10))
        words.append({"w": token, "start_ms": round(t), "end_ms": round(t + length), "p": 0.95})
        t += length + (180 if token[-1] in ".?!" else 0)
    return {"question_id": qid, "question": question, "competency": competency,
            "offset_ms": offset_ms, "words": words}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text_file")
    ap.add_argument("out")
    ap.add_argument("--qid", default="q1")
    ap.add_argument("--question", required=True)
    ap.add_argument("--competency", required=True)
    ap.add_argument("--offset-ms", type=int, default=0)
    ap.add_argument("--wpm", type=float, default=150.0)
    args = ap.parse_args()
    data = build(Path(args.text_file).read_text(), qid=args.qid, question=args.question,
                 competency=args.competency, offset_ms=args.offset_ms, wpm=args.wpm)
    Path(args.out).write_text(json.dumps(data, indent=2))
    print(f"{len(data['words'])} words → {args.out}")


if __name__ == "__main__":
    main()
