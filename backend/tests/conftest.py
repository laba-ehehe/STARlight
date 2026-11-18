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
