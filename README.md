# STARlight

A multimedia expert system for behavioral interview practice (CIS4930, Fall 2026).
It runs a timed mock interview, listens to spoken answers, asks expert-style follow-up
questions, and returns a transcript with timestamped coaching feedback.

- `PLAN.md` — design, decisions, schedule
- `IMPLEMENTATION.md` — step-by-step build order
- `backend/` — Python: shared core, extractors, inference engine, API
- `backend/knowledge/` — the rule base (YAML, editable without code changes)

## Backend setup

```bash
cd backend
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q          # run the tests
ruff check .       # lint
```

Validate the knowledge base after editing any YAML file:

```bash
python -c "from starlight.core import KnowledgeBase as K; print(*K.load('knowledge').validate() or ['OK'], sep='\n')"
```
