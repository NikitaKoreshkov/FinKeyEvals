# Contributing to FinKeyEvals

Thanks for your interest!

## Contributor licence terms

By opening a pull request you agree that copyright of your contribution is
**assigned to FinKey**, and that FinKey may relicense it (including commercially)
while keeping the community edition under AGPL-3.0. This is what lets the project
stay open *and* fundable. You keep the right to use your contribution elsewhere.

If you cannot agree to assignment, open an issue instead of a PR — we'll work it out.

## Ground rules

- Python 3.10+, **zero new required dependencies** — optional extras only
  (`judge`, `pg`). The heuristic tier must keep working offline and keyless.
- Scoring stays deterministic: pure functions, no network — a failing CI run
  must be reproducible.
- Storage-touching changes keep the degradation contract: missing Postgres
  disables persistence to a ring buffer, never crashes the caller.
- Every new behaviour ships with a test in `tests/` (pure-Python first; anything
  requiring live backends goes to `tests/live/`).

## Running the suite

```bash
pip install -e ".[test]"
pytest
```
