"""starlight.core: the shared classes every module builds on."""

from .engine import BackwardResult, InferenceEngine, InferenceError, ProofFailure
from .explain import ExplanationTrace, Firing
from .facts import Fact, is_var, substitute
from .fuzzy import FuzzyVariable, MembershipFunction
from .kb import KnowledgeBase, Problem
from .media import AudioClip, MediaObject, Sentence, TextDocument, Transcript, Word
from .memory import WorkingMemory
from .rules import Condition, Failure, Match, Rule, RuleError
from .session import AnswerTurn, PlannedQuestion, Session
from .timespan import TimeSpan

__all__ = [
    "AnswerTurn", "AudioClip", "BackwardResult", "Condition", "ExplanationTrace", "Fact",
    "Failure", "Firing", "FuzzyVariable", "InferenceEngine", "InferenceError", "KnowledgeBase",
    "Match", "MediaObject", "MembershipFunction", "PlannedQuestion", "Problem", "ProofFailure",
    "Rule", "RuleError", "Sentence", "Session", "TextDocument", "TimeSpan", "Transcript",
    "Word", "WorkingMemory", "is_var", "substitute",
]
