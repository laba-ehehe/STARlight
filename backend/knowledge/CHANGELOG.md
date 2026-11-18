# Knowledge base changelog

Every change to a rule, threshold, fuzzy set, lexicon or question goes here: what changed, why, and the date. This log feeds the "knowledge acquisition" section of the final report.

Rule status: `hypothesis` (drafted from coaching practice) → `team-reviewed` → `confirmed` / `revised` / `rejected` after expert review.

## 2026-11-17 — Phase 2: R18–R53, lexicons, question bank, schema

### Rule language
- Added a `count` condition (`count: [ISSUE, ?a, clarity, ?_]` + `as: ?n`): binds the number of distinct matching facts, 0 if none. Needed for "≥ 3 long pauses" (R42), hedges per minute (R43), ratings (R47–R50), the feedback cap (R51) and cross-answer patterns (R53). Counted facts appear in HOW explanations but do not change a rule's strength or span.
- Added a `rating` group, run after `answer_analysis` + `delivery` and before `feedback`, since its rules count or negate ISSUE facts.

### Rules added (all `status: hypothesis`, source "PLAN §6 draft")
| Rules | Group | Notes |
|---|---|---|
| R18–R20 | question_planning | conflict / problem-solving / challenges cues → HIGH, same shape as R1/R2 |
| R21 | question_planning | vague JD (no HIGH) → teamwork, problem-solving, challenges at MEDIUM |
| R22 | question_planning | early-career cue → `PREFER-EXAMPLES academic`; M2 prefers `academic: true` questions |
| R23 | question_planning | more HIGH competencies than N → `CUE-WEIGHT` per competency; M2 keeps the top N |
| R24 | question_planning | N > target competencies → add remaining competencies at MEDIUM, one at a time, until they cover N |
| R25–R27 | session_control | follow-up cap: phone 1, panel 2, one-way video 0 |
| R28 | session_control | follow-ups asked ≥ cap → `SKIP-FOLLOWUP cap-reached` |
| R29 | session_control | time left < next question's budget → `GO-TO-CLOSING` |
| R30 | answer_analysis | < 15 s and < 25 words → `BARE-ANSWER`, follow-up "Take your time — can you walk me through it?" |
| R31, R32, R38, R39 | answer_analysis | strengths: STAR structure, reflection, clear ownership, measurable impact |
| R33, R34, R35, R36, R37, R40 | answer_analysis | multiple stories, generic answer, blame framing, missing lesson, weak leadership evidence, long setup |
| R41–R43 | delivery | slow pace, ≥ 3 long pauses (each marked), > 3 hedges/min (each marked) |
| R44, R45 | delivery | strengths: clear delivery, good length |
| R46, R47a/b | rating | relevance: off-topic → needs work; < 0.6 or a relevance issue → adequate; otherwise strong |
| R48a/b/c, R49a/b/c | rating | clarity / organization: 0 issue kinds → strong, 1 → adequate, ≥ 2 → needs work |
| R50a/b/c/d | rating | examples: same counts, but "strong" also needs a Result segment (no result → adequate) |
| R51 | feedback | > 3 issue kinds in one answer → `FEEDBACK-CAP 3` (M7 keeps the highest-priority three) |
| R53 | feedback | same issue kind in ≥ 2 answers → `SESSION-PATTERN` |

### Deviations from the PLAN §6 draft
- **R52** ("every negative item gets a concrete tip") is a knowledge-base constraint rather than a runtime rule: `KnowledgeBase.validate()` reports an error for any rule with negative feedback and no `tip`. R52 is the only number missing from R1–R53.
- **R16** has two new sub-goals: `ANSWERED` first (blocked by R30's `BARE-ANSWER`) and `LESSON-STATED` last (blocked by R36's `MISSING-LESSON`). `REAL-EXAMPLE` is now also blocked by `GENERIC-ANSWER` (R34), the "same slot as HYPOTHETICAL" the plan describes.
- **R38** asserts `STRENGTH … clear-ownership` instead of `CLEAR-OWNERSHIP`, so that goal facts are only derived by the R16 goal rules.
- **R48** counts kinds of clarity issue (fast pace, fillers, pauses, hedging, slow pace), not individual filler or pause events: one filler-heavy answer would otherwise always reach "needs work". Thresholds 0 / 1 / ≥ 2 kinds.
- **R14a** now marks `(CLARITY-ISSUE ?a fast-pace all)` instead of `(… pace all)`, to tell it apart from R41 slow pace.
- Ratings use `not ISSUE` for "strong", so WHY-NOT on a missing Strong rating names the exact issue and, through HOW, the rule that raised it.

### Thresholds to tune on the 10 validation samples
R30 (15 s / 25 words), R33 (2 stories), R40 (35 % setup), R42 (3 pauses), R43 (3 hedges/min), R44 (3 fillers/min), R47 (relevance 0.6), R48–R50 (issue-count cut-offs), R51 (cap 3).

### Lexicons (`lexicons.yaml`)
First version: 15–18 cues for each of the 5 competencies, early-career cues, fillers, hedges, STAR cue phrases per label, result evidence, initiative verbs, blame cues, generic claims, negative outcomes, reflection cues, pronouns, conditional markers.

### Question bank (`questions.yaml`)
First version: 35 questions, 7 per competency, each tagged with formats, difficulty 1–3 and `academic` (answerable with a class or club example). Every format has at least 2 questions per competency.

### Schema (`schema/session-description.schema.json`)
First version: Session → Question → Answer → Segment, with Descriptors, Ratings, Feedback (with HOW / WHY-NOT) and FollowUpDecision. `Session.to_description()` validates against it.
