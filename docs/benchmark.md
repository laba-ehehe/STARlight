# Whisper benchmark (IMPLEMENTATION 0.4)

**Question:** which Whisper model can transcribe a 2-minute answer fast enough for the live follow-up decision (≤ 5 s total, so ≤ 3.5 s for transcription and ≈ 1.5 s for the rules)?

**How to run:**
```bash
cd backend
python scripts/bench_whisper.py --synthetic                  # macOS `say` clips of 60 s and 120 s
python scripts/bench_whisper.py rec60.webm rec120.webm       # real Chrome recordings
python scripts/bench_whisper.py --synthetic --device mps     # Apple GPU (experimental)
```
Each model is loaded once and warmed up. Times are the median of 3 runs. The synthetic clips use the transcript fixtures' text, so the word error rate (WER) can be measured.

## Run 1: 2026-11-17, Apple M2 (8 cores), CPU int8

The machine was **not idle**: Chrome and a screen recorder kept the load average at 15–17 on 8 cores. Treat these numbers as an upper bound. They are noisy (small.en with beam 1 took longer than with beam 5).

| Model | Beam | Clip | Audio (s) | Transcribe (s) | × real time | Words | WER |
|---|---:|---|---:|---:|---:|---:|---:|
| tiny.en | 1 | synthetic 60s | 59.5 | 2.93 | 0.049 | 173 | 3.5% |
| tiny.en | 1 | synthetic 120s | 119.4 | 3.61 | 0.030 | 367 | 2.7% |
| tiny.en | 5 | synthetic 60s | 59.5 | 3.11 | 0.052 | 173 | 4.0% |
| tiny.en | 5 | synthetic 120s | 119.4 | 10.04 | 0.084 | 367 | 3.3% |
| base.en | 1 | synthetic 60s | 59.5 | 4.41 | 0.074 | 175 | 4.6% |
| base.en | 1 | synthetic 120s | 119.4 | 6.71 | 0.056 | 368 | 2.7% |
| base.en | 5 | synthetic 60s | 59.5 | 6.34 | 0.106 | 175 | 4.6% |
| base.en | 5 | synthetic 120s | 119.4 | 11.03 | 0.092 | 369 | 3.0% |
| small.en | 1 | synthetic 60s | 59.5 | 12.07 | 0.203 | 173 | 3.5% |
| small.en | 1 | synthetic 120s | 119.4 | 79.20 | 0.663 | 366 | 2.2% |
| small.en | 5 | synthetic 60s | 59.5 | 25.24 | 0.424 | 173 | 3.5% |
| small.en | 5 | synthetic 120s | 119.4 | 28.94 | 0.242 | 365 | 1.6% |

A re-run of the 120 s clip with greedy decoding (beam 1) a few minutes later, still under load:

| Device | Model | 3 runs (s) |
|---|---|---|
| CPU int8 | tiny.en | 5.90, 6.34, 8.42 |
| CPU int8 | base.en | 8.76, 12.02, 8.34 |
| MPS float16 | tiny.en | **2.54, 2.55, 2.57** |
| MPS float16 | base.en | 4.58, 11.42, 7.59 |

## Findings

1. **Greedy decoding (beam 1) is the main speed-up:** about 2–3× faster than the default beam 5 on 120 s clips, with the same WER on these clips. The live pass uses beam 1, and the post-session pass keeps beam 5.
2. **Only tiny.en comes near the 3.5 s budget** for a 120 s answer on this machine. base.en needs about 7 s or more on CPU.
3. **small.en is post-session only:** it takes 25–80 s per 2-minute answer, which is fine as a background job after the session.
4. **The Apple GPU backend (MPS) was the fastest and steadiest:** tiny.en took 2.5 s three times in a row. It is experimental in ctranslate2 and only exists on Apple Silicon, so it's an opt-in, not the default.
5. **WER is low for every model (1.6–4.6 %)** because synthetic speech is clean. Real recordings will show bigger differences between models; re-check with the step 4.4 samples.

## Decision (provisional)

- **Live:** `tiny.en`, beam 1, CPU int8.
- **Post-session:** `small.en`, beam 5.

Both can be overridden without code changes:
```bash
export STARLIGHT_LIVE_MODEL=base.en      # if the demo laptop is fast enough
export STARLIGHT_WHISPER_DEVICE=mps      # Apple Silicon GPU
```

**To do before freezing the choice (Phase 8):**
- Re-run on an **idle** demo laptop, with Chrome and the screen recorder closed, using the real 60 s and 120 s Chrome recordings from step 4.4.
- If base.en fits the budget there, switch to it, since it should be more accurate on real speech.

**Fallback if even tiny.en misses the budget on the demo laptop:** transcribe each 10 s upload chunk as it arrives (PLAN risk table). After "Done answering", only the last chunk is left to transcribe, so the wait no longer grows with answer length.
