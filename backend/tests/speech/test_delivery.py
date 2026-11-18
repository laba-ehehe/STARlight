"""Lexicon phrase matching, M4 DeliveryAnalyzer and speech_facts."""

import pytest
from helpers import FIXTURES, answer_facts, spoken

from starlight.core import InferenceEngine, TimeSpan, Transcript, Word
from starlight.lexicon import Lexicon, match_phrases, tokens
from starlight.speech.delivery import DeliveryAnalyzer
from starlight.speech.facts import speech_facts

# ---------------------------------------------------------------- lexicon matching


def test_tokens_normalize_case_punctuation_hyphens_and_quotes():
    assert tokens("Cross-functional,") == ["cross", "functional"]
    assert tokens("I’m") == ["i'm"]
    assert tokens("“Done.”") == ["done"]


def test_match_phrases_longest_first_without_overlap():
    words = [Word(t, i * 100, i * 100 + 80) for i, t in
             enumerate(["I", "think,", "you", "know,", "I", "think", "so."])]
    found = match_phrases(words, ["I think", "you know", "think", "I"])
    assert [(m.phrase, m.start, m.end) for m in found] == [
        ("I think", 0, 2), ("you know", 2, 4), ("I think", 4, 6)]
    assert found[1].span == TimeSpan(200, 380) and found[1].text == "you know,"


def test_lexicon_find_sub_list():
    t = spoken("As a result, the build got faster. I learned a lot.")
    found = Lexicon.default().find(t.words, "star_cues", "result")
    assert [m.phrase for m in found] == ["as a result", "I learned"]


# ---------------------------------------------------------------- delivery measures

TEXT = ("Um, so last year I was, like, on a team. [pause 3] Uh, I think we, you know, "
        "finished early. [pause 2.5] I like working with them.")


@pytest.fixture
def delivery():
    return DeliveryAnalyzer(spoken(TEXT, offset_ms=1_000))


def test_long_pauses(delivery):
    pauses = delivery.long_pauses()
    assert [p.ms for p in pauses] == [pytest.approx(3000, abs=200), pytest.approx(2500, abs=200)]
    words = delivery.words
    assert pauses[0].at_ms == words[9].end_ms             # right after "team."
    assert pauses[0].span.end_ms == words[10].start_ms


def test_fillers_without_spacy(delivery):
    found = [(m.phrase, m.text) for m in delivery.filler_events()]
    assert found == [("um", "Um,"), ("like", "like,"), ("uh", "Uh,"), ("you know", "you know,")]


def test_fillers_with_spacy(nlp):
    d = DeliveryAnalyzer(spoken("So like we had to redo it, and I like that we did."), nlp=nlp)
    assert [m.start for m in d.filler_events()] == [1]    # only the first "like"


def test_pace_excludes_long_pauses(delivery):
    n = delivery.word_count()
    assert n == 23
    with_pauses = n * 60_000 / delivery.duration_ms()
    assert delivery.pace_wpm() > with_pauses
    assert delivery.speaking_ms() == delivery.duration_ms() - sum(
        p.ms for p in delivery.long_pauses())


def test_synthetic_pace_is_close_to_requested():
    t = Transcript.load(FIXTURES / "transcripts" / "strong.json")
    assert DeliveryAnalyzer(t).pace_wpm() == pytest.approx(150, rel=0.05)


def test_turns_are_measured_separately():
    main = spoken("I set up the cache and it worked well for us.", offset_ms=0)
    follow = spoken("Search went from four seconds to under one.", offset_ms=60_000)
    d = DeliveryAnalyzer([main, follow])
    assert d.long_pauses() == []                          # the 55 s between turns is not a pause
    assert d.duration_ms() == (main.span.duration_ms() + follow.span.duration_ms())
    assert d.span == TimeSpan(main.span.start_ms, follow.span.end_ms)


def test_empty_answer():
    d = DeliveryAnalyzer(Transcript([]))
    assert (d.word_count(), d.duration_ms(), d.pace_wpm(), d.filler_rate()) == (0, 0, 0.0, 0.0)
    assert d.span is None


# ---------------------------------------------------------------- facts


def test_speech_facts(delivery):
    facts = speech_facts("q1", [], analyzer=delivery)
    text = [str(f) for f in facts]
    assert text[:3] == ["(ANSWER q1)", f"(ANSWER-DURATION q1 {delivery.duration_ms() / 1000:.1f})",
                        "(WORD-COUNT q1 23)"]
    assert sum(t.startswith("(FILLER q1") for t in text) == 4
    assert sum(t.startswith("(LONG-PAUSE") for t in text) == 2
    assert all(f.source == "M4" for f in facts)
    [answer] = [f for f in facts if f.pred == "ANSWER"]
    assert answer.span == delivery.span
    [um] = [f for f in facts if f.pred == "FILLER" and f.args[1] == "um"]
    assert um.span.start_ms == 1_000 and um.args[2] == 1_000


def test_no_pace_fact_for_a_few_words():
    facts = speech_facts("q1", spoken("Yes, I did."))
    assert not any(f.pred == "PACE" for f in facts)


def test_speech_facts_drive_delivery_rules(kb):
    """M4 facts plug straight into the rules: three pauses → R42 marks each one."""
    t = spoken("I started. [pause 3] Then I fixed it. [pause 3] Then I tested it. [pause 3] "
               "And then I shipped the whole thing to production on Friday.")
    engine = InferenceEngine(kb)
    content = [f for f in answer_facts() if f.source == "M5"]
    engine.wm.assert_all(speech_facts("q1", t) + content)
    engine.forward_chain(["answer_analysis", "delivery"])
    pauses = [f for f in engine.wm.facts("CLARITY-ISSUE") if f.args[1] == "pause"]
    assert len(pauses) == 3
