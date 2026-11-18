# Contributing

`main` is protected: nobody pushes to it directly. Every change goes through a pull request.

- No approval is needed, so you can merge your own PR.
- The **backend** CI check (lint, knowledge-base validation, tests) must pass before the merge button unlocks.
- Force-pushing to `main` and deleting it are blocked.

## Workflow

```bash
git switch main && git pull
git switch -c m5-star-segmenter          # one branch per task
# ...work, commit...
git push -u origin m5-star-segmenter
gh pr create --fill                      # or open the PR on github.com
# wait for CI to go green, then:
gh pr merge --squash --delete-branch
```

Branch names: `<module or phase>-<short-topic>`, e.g. `m4-delivery-analyzer`, `kb-rules-r18-r30`, `frontend-replay`.

Keep PRs small (one IMPLEMENTATION.md step is a good size), so they merge the same day and nobody drifts far from `main`.

## Before you open a PR

```bash
cd backend
ruff check .
pytest -q
```

If you change a rule or a threshold in `backend/knowledge/`, add a line to `backend/knowledge/CHANGELOG.md` saying what changed and why.

## Repository settings (one-time, done by the repo owner)

```bash
gh auth login
gh api -X POST repos/laba-ehehe/STARlight/rulesets --input .github/rulesets/main.json
gh api -X PATCH repos/laba-ehehe/STARlight \
  -F delete_branch_on_merge=true -F allow_squash_merge=true \
  -F allow_merge_commit=false -F allow_rebase_merge=false
```

Teammates need **Write** access (Settings → Collaborators) to push branches and merge their PRs.
