from __future__ import annotations

import pytest
from helpers import KNOWLEDGE

from starlight.core import InferenceEngine, KnowledgeBase, WorkingMemory


@pytest.fixture(scope="session")
def kb() -> KnowledgeBase:
    return KnowledgeBase.load(KNOWLEDGE)


@pytest.fixture
def engine(kb) -> InferenceEngine:
    return InferenceEngine(kb, WorkingMemory())


@pytest.fixture(scope="session")
def nlp():
    """spaCy en_core_web_md; tests that need it are skipped when it is not installed."""
    spacy = pytest.importorskip("spacy")
    try:
        return spacy.load("en_core_web_md")
    except OSError:
        pytest.skip("en_core_web_md is not installed (python -m spacy download en_core_web_md)")
