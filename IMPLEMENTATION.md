# STARlight — Step-by-Step Implementation Guide

Companion to `PLAN.md` (the *what* and *why*). This file is the *how, in order*.
Scope: the **full Milestone #2 system**, with 8 modules, R1–R17 grown to ≥ 50 rules, a live timed session, a synced replay and JSON/XML export.

Conventions:
- **Owner** is the person responsible; anyone can help.
- **Done when** is the check that must pass before the step counts as finished.
- Steps with the same number prefix (e.g. 3.x) can run in parallel with other people's steps once phase 1 is merged.
- Expert sessions are **on hold**. Every expert step has a team fallback (marked ⟲).

---

## Phase 0 — Project setup (Mon 11/16 – Tue 11/17)

### 0.1 Backend skeleton — Lan Anh
```bash
cd STARlight
mkdir -p backend/starlight/{core,jd,planner,speech,content,inference,feedback,api} \
         backend/knowledge backend/schema backend/tests/fixtures/{transcripts,audio,jd} \
         backend/data validation docs
python3.12 -m venv backend/.venv && source backend/.venv/bin/activate
```
`backend/pyproject.toml` dependencies:
```
faster-whisper  spacy>=3.8  pdfplumber  av  pyyaml  jsonschema
fastapi  uvicorn[standard]  python-multipart  pydantic>=2
dev: pytest  pytest-asyncio  httpx  ruff
```
```bash
pip install -e "backend[dev]"
python -m spacy download en_core_web_md
```
Add `__init__.py` to every package and a `.gitignore` covering `.venv/`, `backend/data/`, `node_modules/`, `__pycache__/` and `*.webm` outside fixtures.
**Done when:** `pytest` runs (0 tests) and `python -c "import faster_whisper, spacy, av, pdfplumber"` works on all 5 laptops.

### 0.2 Frontend skeleton — Nancy
```bash
npm create vite@latest frontend -- --template react-ts
cd frontend && npm i && npm i -D vitest
```
Add a Vite proxy so `/api` goes to `http://localhost:8000`.
**Done when:** `npm run dev` shows the default page and `npx tsc --noEmit` passes.

### 0.3 CI and repo rules — Lan Anh
- `.github/workflows/ci.yml`: backend job (`ruff check`, `pytest`), frontend job (`tsc --noEmit`, `vitest run`).
- Protect `main`: pull request + 1 approval, CI green.
- Create the module branches: `core-m6`, `m1-m5`, `m2-m7`, `m4-m8`, `m3-frontend`.

### 0.4 Whisper benchmark — Bhavana (decides the live model)
Write `backend/scripts/bench_whisper.py`:
```python
from faster_whisper import WhisperModel
import time, sys
for name in ["tiny.en", "base.en", "small.en"]:
    m = WhisperModel(name, device="cpu", compute_type="int8")
    for path in sys.argv[1:]:
        t = time.perf_counter()
        segs, info = m.transcribe(path, word_timestamps=True, vad_filter=True)
        words = [w for s in segs for w in s.words]   # generator: must consume it
        print(name, path, f"{time.perf_counter()-t:.1f}s", info.duration, len(words))
```
Run it on a 60 s and a 120 s recording made in Chrome (WebM) on the **slowest** laptop.
**Done when:** a table goes in `docs/benchmark.md` and the live model is chosen: the largest model that transcribes 120 s of audio in ≤ 3.5 s, leaving ≈ 1.5 s for rules.

### 0.5 Transcript fixture format — Parth + Lan Anh
`tests/fixtures/transcripts/<name>.json` is the format M4 outputs and M5 consumes. Hand-write 5 fixtures now (strong, no-result, we-heavy, hypothetical, off-topic) so M5, M6 and M7 can work with no audio:
```json
{
  "question_id": "q1",
  "question": "Tell me about a time you worked on a team under a tight deadline.",
  "competency": "teamwork",
  "offset_ms": 42000,
  "words": [
    {"w": "Last", "start_ms": 42310, "end_ms": 42520, "p": 0.98},
    {"w": "semester", "start_ms": 42520, "end_ms": 43010, "p": 0.97}
  ]
}
```
Tip: write the answer text, then generate the timings with a script at ~150 wpm, adding gaps where you want long pauses.

---

## Phase 1 — Shared core `starlight.core` (Lan Anh, Tue 11/17 – Thu 11/19)

**Merge stubs first (all signatures with `raise NotImplementedError`) on 11/17** so everyone codes against the real interfaces. Then fill them in, in this order, each with tests in `tests/core/`.

| Step | File | Build | Done when (tests) |
|---|---|---|---|
| 1.1 | `timespan.py` | `TimeSpan(start_ms, end_ms)` frozen dataclass; `overlaps`, `duration_ms`, `merge` | merge of disjoint spans covers both; touching spans don't overlap |
| 1.2 | `facts.py` | `Fact(pred, args, span, certainty, source)`; `matches(pattern)` where a pattern is `[PRED, ?x, "lit"]` and returns a bindings dict; `with_span`; `__str__` → `(MISSING-RESULT q2)` | `?x` binds; the same `?x` twice must bind equal values; literals compare exactly |
| 1.3 | `memory.py` | `WorkingMemory`: dict `pred → list[Fact]`; `assert_fact` (no duplicates, returns bool "new"), `retract`, `query(pattern) → list[(fact, bindings)]`, `clear` | duplicate assert returns False; query by predicate is O(facts of that pred) |
| 1.4 | `fuzzy.py` | `MembershipFunction(a,b,c,d).degree(x)` trapezoid with `inf` support; `FuzzyVariable(name, sets).fuzzify(x) → {set: degree}` | SHORT(75) = 0.5, IDEAL(75) = 0.5, LONG(1000) = 1, FAST(170) = 0.5 |
| 1.5 | `rules.py` | `Rule` from a YAML dict. `matches(wm, fuzzy_vars)` yields every bindings set satisfying all conditions (`fact`, `not`, `test`, `fuzzy`), evaluated left to right, joining bindings. `strength(bindings)` = min(fuzzy degrees) × min(fact certainties). `fire(bindings)` returns the new facts with span = merge of matched positive facts' spans | R7 fires on a WM with ANSWER and no result segment; R12 strength = SHORT degree |
| 1.6 | `kb.py` | `KnowledgeBase.load(dir)` reads `rules.yaml` + `fuzzy.yaml`; `rules_in(group)`; `validate()` returns a list of problems (unknown predicate, unbound variable, contradictory conclusions) | loading R1–R17 gives 0 problems; a seeded bad rule gives 1 |
| 1.7 | `explain.py` | `ExplanationTrace.record(rule, bindings, matched, strength, produced)`; `how(fact)` → nested dict (rule, matched facts + spans, recursively); `why_not(pattern, kb, wm)` → for each rule whose `then` could produce it: the first failing condition and what WM held | `how(MISSING-RESULT q1)` names R7; `why_not` for the Strong rating names the missing fact |
| 1.8 | `engine.py` | `InferenceEngine(kb, wm, trace)`. `forward_chain(groups)`: loop {build agenda of (rule, bindings) not yet fired with strength ≥ 0.5 → `resolve_conflicts` → fire top → record} until the agenda is empty. `resolve_conflicts`: sort by priority desc, then condition count desc, then newest matched fact. `backward_chain(goal)`: see 1.9 | refraction (no infinite loop); higher priority fires first; 1000 facts × 50 rules < 100 ms |
| 1.9 | `engine.py` | `backward_chain(("COMPLETE", "q2"))`: the goal's sub-goals come from a `goals:` block in `rules.yaml` (R16 order). For each sub-goal: proven if its blocking fact is absent after forward chaining *or* a rule can derive the positive fact. Returns the first unproven sub-goal + its follow-up text, or `None` | fixture "no-result" → MISSING-RESULT; "hypothetical + no-result" → HYPOTHETICAL (priority wins) |
| 1.10 | `media.py` | `MediaObject` ABC with `open/read/write/control/close`. `AudioClip(path, offset_ms)`: `to_pcm16k()` via PyAV resample to 16 kHz s16 mono → numpy; `duration_ms()`. `TextDocument`: `extract_text()` (pdfplumber for PDFs), `find_cues(lexicon)`. `Transcript(words)`: `words_in(span)`, `sentences()` (spaCy sentence split mapped back to word spans), `span_of(text)` | a Chrome WebM decodes to 16 kHz; `sentences()` spans line up with word times |
| 1.11 | `session.py` | `Session(id, jd, format, duration_min, started_at)`: `add_answer`, `elapsed_ms`, `remaining_ms`, `to_description()` (dict matching the JSON Schema) | `to_description()` validates against the schema (step 2.5) |

`rules.yaml` `goals:` block (used by 1.9):
```yaml
goals:
  COMPLETE:
    - {subgoal: ON-TOPIC,          blocked_by: OFF-TOPIC,           rule: R10}
    - {subgoal: REAL-EXAMPLE,      blocked_by: HYPOTHETICAL-ANSWER, rule: R9}
    - {subgoal: CLEAR-OWNERSHIP,   blocked_by: UNCLEAR-OWNERSHIP,   rule: R6}
    - {subgoal: RESULT-STATED,     blocked_by: MISSING-RESULT,      rule: R7}
    - {subgoal: RESULT-SUPPORTED,  blocked_by: UNSUPPORTED-RESULT,  rule: R8}
```

---

## Phase 2 — Knowledge files (Aradhana, Tue 11/17 – Sat 11/21)

| Step | File | Build | Done when |
|---|---|---|---|
| 2.1 | `knowledge/rules.yaml` | R1–R17 in the PLAN §3.3 format, `status: hypothesis` | `KnowledgeBase.validate()` → 0 problems |
| 2.2 | `knowledge/fuzzy.yaml` | the 6 trapezoids from Milestone #2 (answer_duration: SHORT/IDEAL/LONG; pace: SLOW/NORMAL/FAST) | 1.4 tests read from this file |
| 2.3 | `knowledge/lexicons.yaml` | competency cue lists for JDs (5 competencies × 15+ words); fillers; hedges; STAR cue phrases per label; result-evidence cues ("increased", "saved", "manager said"); initiative verbs; blame cues | reviewed by Parth (M1/M5 use it) |
| 2.4 | `knowledge/questions.yaml` | ≥ 6 questions per competency (≥ 30), each `{id, competency, text, formats: [phone, video, panel], difficulty}` | M2 tests can load it |
| 2.5 | `schema/session-description.schema.json` | MPEG-7-style: Session DS → Question DS → Answer DS → Segment DS (STAR label, span, descriptors: pace, pauses, fillers) → Feedback DS (criterion, polarity, rating, span, rule_id, explanation) | example JSON validates with `jsonschema` |
| 2.6 | `rules.yaml` | add R18–R53 from PLAN §6, adjusting to the vocabulary | ≥ 50 rules, validate → 0 problems, each new rule has a fixture that fires it |
| 2.7 | `knowledge/CHANGELOG.md` | log every threshold or rule change with reason and date | goes into the final report |

⟲ Expert review on hold: the team marks rules `status: team-reviewed` after one other member reads each.

---

## Phase 3 — Extractors (start as soon as core stubs merge, 11/17)

### M4 Speech Processor — Bhavana (11/17 – 11/20)
| Step | Build | Done when |
|---|---|---|
| 4.1 | `speech/transcriber.py` `SpeechTranscriber(model_name)`: loads the `WhisperModel` **once** at startup. `transcribe(clip: AudioClip) → Transcript`: `word_timestamps=True`, `vad_filter=True`, `initial_prompt="Umm, so, uh, I was, like, working on..."`; seconds → ms; add `clip.offset_ms` | a 60 s Chrome recording gives a Transcript whose first word ≈ its offset |
| 4.2 | `speech/delivery.py` `DeliveryAnalyzer(transcript, span)`: `pace_wpm()` = words ÷ (speaking time excluding gaps > 2 s); `filler_events()` = lexicon fillers + "like" only when spaCy tags INTJ; `long_pauses()` = gaps > 2 s | numbers match a hand count on 2 recordings within ±10 % |
| 4.3 | `speech/facts.py` `speech_facts(qid, transcript, span) → list[Fact]`: ANSWER, ANSWER-DURATION, WORD-COUNT, PACE, FILLER (each with span), FILLER-RATE, LONG-PAUSE | the facts render as `(PACE q1 152)` etc. |
| 4.4 | Record the **10 validation samples** (PLAN §7 table) in Chrome as WebM into `tests/fixtures/audio/`, with `samples.yaml` listing question + intended defect | by **Thu 11/19** |

### M5 Content Feature Extractor — Parth (11/17 – 11/21)
| Step | Build | Done when |
|---|---|---|
| 5.1 | `content/star.py` `StarSegmenter(transcript).label_sentences()`: score each sentence per label = cue-phrase hits (lexicon) + grammar features (first-person subject + past-tense verb → action; NUM/PERCENT entity or "as a result" → result) + position prior (first 25 % +situation, last 25 % +result). Pick the max label; merge consecutive equal labels into segments | 5 text fixtures labelled sensibly |
| 5.2 | `content/analyzer.py` `ContentAnalyzer(transcript, segments, question)`: `pronoun_ratio()` (we/us/our vs I/me/my in Action; 0 "I" → 99), `tense_profile()` (share of action verbs with modal "would" or "'d" vs `Tense=Past`), `has_quantified_result()` (CARDINAL/PERCENT/MONEY/TIME/QUANTITY entity inside Result), `relevance()` = 0.5 × lexicon overlap + 0.5 × `doc.similarity(question_doc)`, plus hedges, reflection, story count (number of separate Situation segments) | each function unit-tested on fixtures |
| 5.3 | `content/facts.py` `content_facts(qid, ...) → list[Fact]` for every M5 row in PLAN §4 | a fixture produces the expected fact list (snapshot test) |
| 5.4 | Accuracy check: hand-label every sentence of the 10 samples (S/T/A/R) in `validation/star_labels.csv`; script prints precision/recall per label | numbers in `docs/star-accuracy.md` |

### M1 JD Analyzer — Parth (11/20 – 11/21)
| Step | Build | Done when |
|---|---|---|
| 1a | `jd/analyzer.py` `JobDescriptionAnalyzer.parse(text_or_pdf_path) → TextDocument` | a PDF and pasted text give the same text |
| 1b | `extract_competencies() → list[Fact]`: lemmatize, match `lexicons.yaml` cues; assert `JD-CUE`, `INTERVIEW-FORMAT`, `SESSION-DURATION` | 5 real internship JDs in `fixtures/jd/` → after R1/R2/R18–R21, sensible `TARGET-COMPETENCY` facts |

---

## Phase 4 — Reasoning modules

### M6 FollowUpSelector — Lan Anh (11/19)
`inference/followup.py` `FollowUpSelector(kb).select(answer_qid, time_left_ms, wm) → FollowUp | None`:
1. Run `forward_chain(["answer_analysis", "delivery", "session_control"])`.
2. If a session-control conclusion says skip (R15 over budget, R28 cap reached, R27 video format, R29 no time), return `None` and record a pacing note.
3. Else `backward_chain(("COMPLETE", qid))` → follow-up text, or `None` if complete.
4. Time the whole call; return `decision_ms` with it.

**Done when:** on each of the 5 fixtures the selector picks the expected follow-up in < 200 ms (without ASR).

### M2 Question Planner — Aradhana (11/18 – 11/20)
| Step | Build | Done when |
|---|---|---|
| 2a | `planner/question.py` `Question.render(context)` (fills `{role}`, `{company}`) | |
| 2b | `planner/bank.py` `QuestionBank.load(path)`, `find(competency, format)` (excluding already-used ids) | |
| 2c | `planner/planner.py` `SessionPlanner.plan(session)`: assert M1 facts → forward-chain `question_planning` → N from R3 → pick questions in competency order (R4, R23, R24) → `budget_for(q)` = (D − 5) × 60000 / N ms; assert QUESTION and BUDGET facts | 15-min phone → 1 question; 45-min panel → 6, covering ≥ 3 competencies |

### M7 Feedback and Report — Aradhana (11/20 – 11/23)
| Step | Build | Done when |
|---|---|---|
| 7a | `feedback/item.py` `FeedbackItem(criterion, polarity, text, tip, span, rule_id, how, why_not)`, `to_dict()` | |
| 7b | `feedback/generator.py` `FeedbackGenerator(engine).build(qid)`: forward-chain `rating` + `feedback` groups; turn every conclusion that has a `feedback:` block into an item; `rate(criterion)` reads the rating facts; attach `trace.how()` and, for anything not Strong, `trace.why_not(("RATING", qid, criterion, "strong"))`; apply R51 (cap 3) and R17 (strength first) | fixture "no-result" → organization not Strong, why_not names R7 |
| 7c | `feedback/report.py` `SessionReport(session, items).to_json()` (validate against the schema) and `to_xml()` (same tree via `xml.etree.ElementTree`, MPEG-7-style tag names) | both exports open and validate |

---

## Phase 5 — Web API and storage (Bhavana, 11/19 – 11/22)

| Step | Build | Done when |
|---|---|---|
| 8.1 | `api/repository.py` `SessionRepository` on SQLite (`sqlite3` stdlib): tables `sessions, questions, turns(sid, tid, qid, offset_ms, path, status), chunks(sid, tid, seq, size), reports(sid, json)`; `save`, `load`, `store_audio(sid, tid, seq, bytes)` writing `data/sessions/{sid}/{tid}.part{seq}` | |
| 8.2 | `api/app.py` FastAPI app; on startup load the KB, spaCy model and Whisper model once (they are slow to load) | `uvicorn starlight.api.app:app --reload` starts in < 20 s |
| 8.3 | `POST /sessions` (JSON or multipart with PDF) → `SessionController.create_session` → M1 → M2 → returns plan with the rule trace for each question | |
| 8.4 | `PUT /sessions/{sid}/turns/{tid}/chunks/{seq}` (raw body, `?offset_ms=` on seq 0); idempotent on (tid, seq) | re-sending a chunk doesn't duplicate it |
| 8.5 | `POST /sessions/{sid}/turns/{tid}/finish` → concatenate parts in seq order into `{tid}.webm` → **remux to `{tid}.ogg` (Opus) with PyAV** so the browser can seek (MediaRecorder WebM has no duration/cues) → M4 live model → M5 → FollowUpSelector → `{action, text, decision_ms}` | a 2-min answer returns in ≤ 5 s on the demo laptop |
| 8.6 | `POST /sessions/{sid}/end` → `BackgroundTasks`: re-transcribe all turns with the post-session model, rebuild WM from scratch, full forward chain, M7 report, save | |
| 8.7 | `GET /report` (202 + `{"status": "processing"}` until ready), `GET /turns/{tid}/audio` (`FileResponse`; check that seeking works in Chrome), `GET /description.json`, `GET /description.xml` | |
| 8.8 | `tests/api/` with `httpx` + FastAPI `TestClient`, using a fixture WebM | full create → upload → finish → end → report cycle passes |

---

## Phase 6 — Frontend M3 (Nancy, 11/17 – 11/24)

Build against `src/api/mock.ts` (fixture JSON responses) until Phase 5 is merged; then flip one flag.

| Step | Build | Done when |
|---|---|---|
| 6.1 | `src/api/client.ts`: typed functions for every endpoint; TypeScript types mirroring the JSON Schema | `tsc` passes |
| 6.2 | **Setup page:** JD textarea + PDF upload, format radio, duration slider 15–60, "Test microphone" with level meter (`AudioContext` + `AnalyserNode`) | |
| 6.3 | **Plan page:** question list, budgets, "chosen because R2: JD mentions 'cross-functional'" | |
| 6.4 | `Recorder` class: `getUserMedia({audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true}})`; mime = first of `audio/webm;codecs=opus`, `audio/mp4` passing `MediaRecorder.isTypeSupported`; `new MediaRecorder(stream, {mimeType, audioBitsPerSecond: 32000})`; `start(10000)`; `ondataavailable` → `uploadChunk(seq++, blob)` with 3 retries; `stop()` → wait for the last chunk → `finish` | a 35 s answer uploads 4 chunks and finish succeeds |
| 6.5 | `SessionTimer`: session clock + question clock (`performance.now()`), `remaining()`, `isOverBudget()`; turns red past budget | |
| 6.6 | **Live page:** question card; read-aloud via `speechSynthesis.speak(new SpeechSynthesisUtterance(text))`; Start / Done / Pause / End; "thinking…" state while `finish` runs; then show the follow-up (and read it aloud) or advance | full loop works against the mock |
| 6.7 | **Report page:** per-question ratings table (4 criteria × Strong/Adequate/Needs work); feedback list with timestamps | |
| 6.8 | `ReplayPlayer` + `TranscriptView`: `<audio>` per turn; words as `<span>`s; on `timeupdate` + `requestAnimationFrame`, binary-search the current word and highlight it; feedback markers on a timeline bar; clicking a marker or sentence → `seek(ms − offset_ms)` | clicking a marker lands within 0.3 s of the word |
| 6.9 | "Why this feedback?" drawer: render `how` as a tree (rule → facts → spans, each clickable) and `why_not` as "Not Strong because …" | |
| 6.10 | Export buttons (JSON, XML); error states (mic denied, upload failed, report processing) | |

---

## Phase 7 — Integration milestones (Lan Anh leads)

### 7.1 CLI end-to-end slice — **Thu 11/19**
`backend/starlight/cli.py`:
```bash
starlight run --jd fixtures/jd/intern1.txt --format phone --duration 15 \
              --answers fixtures/audio/sample02.webm
```
Prints: plan with rule reasons → transcript → facts → follow-up decision with timing → report JSON.
Uses the **real** modules where ready and fixture stubs where not.
**Done when:** sample02 (no result) prints `FOLLOW-UP: What was the outcome? (R7, 3.1 s)`.

### 7.2 Web end-to-end slice — **Sun 11/22**
Browser → setup → one question → answer without a result → follow-up asked aloud → answer it → end → report with replay.
**Done when:** the four demo steps in Milestone #2 §5 work once, end to end, on one laptop.

### 7.3 Daily integration (11/19 → 11/29)
- Rebase module branches onto `main` daily; Lan Anh merges after CI is green.
- E2E test `tests/e2e/test_session.py` runs the CLI on 3 recordings and checks the expected follow-ups and ratings.

---

## Phase 8 — Validation and tuning (11/23 – 11/29)

| Step | Owner | Build |
|---|---|---|
| 8.1 | Bhavana | ⟲ Experts on hold → **2–3 team members who did not write the rules** (e.g. Nancy + Bhavana) blind-rate the 10 samples on the 4 criteria (Strong/Adequate/Needs work) and note the follow-up they'd ask, in `validation/human_ratings.csv`. Keep the same form so expert ratings can be added later |
| 8.2 | Lan Anh | `validation/compare.py`: run STARlight on the 10 samples → `system_ratings.csv`; print exact agreement %, weighted Cohen's kappa per criterion, follow-up agreement, median/max `decision_ms` |
| 8.3 | Aradhana | Tune fuzzy thresholds and rule constants where disagreements cluster; log each change in `CHANGELOG.md`; re-run 8.2; report before/after |
| 8.4 | Parth | STAR labelling accuracy (5.4) and fixes |
| 8.5 | All | Bug bash Sat 11/28: each person runs 3 full sessions in Chrome and files issues |

**Feature freeze: Sun 11/29.**

---

## Phase 9 — Delivery (11/29 – 12/2)

| Step | Owner | Build |
|---|---|---|
| 9.1 | Nancy | Demo script (≈ 5 min, Milestone #2 §5 steps 1–4) + recorded backup video (11/30) |
| 9.2 | Each owner | Report section for their modules by 11/30: design, key classes, tests, results |
| 9.3 | Aradhana | Knowledge acquisition + rule base section; full rule table; changelog; note that rules are team-drafted heuristics pending expert review |
| 9.4 | Lan Anh | Assemble and edit the report (12/1). Fix the Milestone #2 inconsistencies (3 experts vs "two"; 7 vs 5 knowledge-acquisition steps; 5 s latency). Include benchmark, validation table and the Whisper-filler limitation |
| 9.5 | Lan Anh | Tag `v1.0`, README run instructions verified on a clean clone, submit by **Wed 12/2 11:59 pm** |

---

## Critical path (what blocks what)

```
0.1 skeleton ─► 1.x core stubs (11/17) ─┬─► M4, M5, M1, M2, M7, M8 build in parallel
0.5 fixtures ───────────────────────────┤
0.4 benchmark ─► live model choice ─────┘
1.5–1.9 engine ─► M6 selector ─► 7.1 CLI slice (11/19) ─► M8 finish endpoint ─► 7.2 web slice (11/22)
4.4 samples (11/19) ─► 8.1 ratings ─► 8.2 compare ─► 8.3 tuning ─► freeze (11/29)
```
If anything on this path slips by more than a day, raise it at the next Mon/Thu sync and cut from the "nice" list first:
XML export polish → panel/video format rules → hedging/pause rules → read-aloud.
