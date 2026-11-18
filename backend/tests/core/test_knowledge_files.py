"""lexicons.yaml, questions.yaml and the session description schema."""

import copy
import json
import string
from collections import Counter

import jsonschema
import pytest
import yaml
from helpers import BACKEND, FIXTURES, KNOWLEDGE

from starlight.core import (
    AnswerTurn,
    AudioClip,
    PlannedQuestion,
    Session,
    TextDocument,
    Transcript,
)

COMPETENCIES = {"teamwork", "leadership", "conflict", "challenges", "problem-solving"}
SCHEMA_DIR = BACKEND / "schema"


@pytest.fixture(scope="module")
def lexicons():
    return yaml.safe_load((KNOWLEDGE / "lexicons.yaml").read_text())


@pytest.fixture(scope="module")
def questions():
    return yaml.safe_load((KNOWLEDGE / "questions.yaml").read_text())["questions"]


@pytest.fixture(scope="module")
def validator():
    schema = json.loads((SCHEMA_DIR / "session-description.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


# ---------------------------------------------------------------- lexicons


def test_lexicon_sections(lexicons):
    assert set(lexicons["competencies"]) == COMPETENCIES
    assert all(len(cues) >= 15 for cues in lexicons["competencies"].values())
    for key in ("early_career", "fillers", "hedges", "star_cues", "result_evidence",
                "initiative_verbs", "blame_cues", "generic_claims", "negative_outcomes",
                "reflection", "pronouns", "conditional_markers"):
        assert lexicons[key], key
    assert set(lexicons["star_cues"]) == {"situation", "task", "action", "result"}


def test_lexicon_lists_have_no_duplicates(lexicons):
    def lists(node, path=""):
        if isinstance(node, list):
            yield path, node
        elif isinstance(node, dict):
            for k, v in node.items():
                yield from lists(v, f"{path}.{k}")

    for path, items in lists(lexicons):
        dupes = [w for w, n in Counter(str(i).lower() for i in items).items() if n > 1]
        assert not dupes, f"{path}: {dupes}"


def test_competency_cues_find_matches_in_a_job_description(lexicons):
    jd = TextDocument("Software Engineering Intern. You will collaborate with a "
                      "cross-functional team, debug production issues, and adapt to "
                      "changing priorities in a fast-paced environment.")
    cues = jd.find_cues(lexicons["competencies"])
    assert {"teamwork", "problem-solving", "challenges"} <= set(cues)
    assert jd.find_cues({"early": lexicons["early_career"]})["early"][0].matched == "Intern"


# ---------------------------------------------------------------- questions


def test_question_bank(questions):
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)) >= 30
    by_comp = Counter(q["competency"] for q in questions)
    assert set(by_comp) == COMPETENCIES and min(by_comp.values()) >= 6
    for q in questions:
        assert set(q["formats"]) <= {"phone", "video", "panel"} and q["formats"]
        assert q["difficulty"] in (1, 2, 3)
        assert isinstance(q["academic"], bool)
        slots = {f for _, f, _, _ in string.Formatter().parse(q["text"]) if f}
        assert slots <= {"role", "company"}, q["id"]


@pytest.mark.parametrize("fmt", ["phone", "video", "panel"])
def test_every_format_covers_every_competency(questions, fmt):
    covered = Counter(q["competency"] for q in questions if fmt in q["formats"])
    assert set(covered) == COMPETENCIES
    assert min(covered.values()) >= 2      # enough for a repeat in a long session


# ---------------------------------------------------------------- schema


def test_example_description_validates(validator):
    example = json.loads((SCHEMA_DIR / "examples" / "session.example.json").read_text())
    validator.validate(example)


@pytest.mark.parametrize("path, value", [
    (("session", "format"), "zoom"),
    (("session", "duration_min"), 90),
    (("session", "questions", 0, "ratings", "clarity"), "great"),
    (("session", "questions", 0, "answers", 0, "span", "start_ms"), -5),
    (("session", "questions", 0, "feedback", 0, "polarity"), "meh"),
])
def test_invalid_descriptions_are_rejected(validator, path, value):
    example = json.loads((SCHEMA_DIR / "examples" / "session.example.json").read_text())
    bad = copy.deepcopy(example)
    node = bad
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(bad)


def test_session_to_description_validates(validator, tmp_path):
    """Step 1.11's done-when: Session.to_description() matches the schema."""
    session = Session(TextDocument("Lead a team."), "panel", 45, session_id="s1")
    session.start()
    session.add_question(PlannedQuestion("q1", "Tell me about a time...", "teamwork",
                                         400_000, "R2: JD mentions 'team'"))
    transcript = Transcript.load(FIXTURES / "transcripts" / "strong.json")
    clip = AudioClip(tmp_path / "q1.0.webm", offset_ms=42_000)
    session.add_answer(AnswerTurn("q1.0", "q1", "Tell me about a time...", clip=clip,
                                  transcript=transcript, asked_at_ms=40_000))
    session.add_answer(AnswerTurn("q1.1", "q1", "What was the outcome?"))
    session.end()
    validator.validate(session.to_description())
