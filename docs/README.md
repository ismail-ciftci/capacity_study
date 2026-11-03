# Docs

| File | Role |
|---|---|
| `PROJECT_BRIEF.md` | The spec and source of truth: what is computed, what is decided, what is open, and the confound (C) and evidence-gap (G) registers. Written for agents. |
| `PROJECT_GUIDE.md` | The brief taught as a lecture book for onboarding, with a worked example. Chapter n explains brief section Sn. Adds no decisions; where the two documents are coupled, the guide's readability decides the structure. |
| `literature/index.md` | Every paper the brief cites, with its key and sha256. Originals (`literature/sources/`) and text extractions for search (`literature/text/`) are local only. |
| `../CLAUDE.md` | Implementation conventions where the brief is silent, and the parts of the brief not yet implemented. |

The docs state current guidance only. Past versions, removed items and the runs behind a change live in git history.

## Change protocol

One iteration of the study:

1. **Run and diagnose.** Write the diagnosis as a temporary working note outside `docs/`.
2. **Distil it into missing guidance.** Each problem becomes a Left-open item or a G entry in the brief, phrased as
   what the experiment lacks guidance on, not as what went wrong.
3. **Find guidance, minimally.** Prefer one source that backs a whole mechanism over one source per detail. Add it
   to `literature/` (original, `pdftotext` extraction, index row).
4. **Update in order:** the brief (a decision becomes Ascertained only when the owner confirms); the guide, where
   `pytest tests/test_docs.py` names every chapter whose brief section changed (review the chapter, then copy the
   printed `Synced:` line into it); `CLAUDE.md`; then config, code and tests. Until the code follows, list the gap
   under "Brief ahead of code" in `CLAUDE.md`.
5. **Delete the diagnosis note.** Remove superseded text from the docs instead of marking it removed.

The docs use ASCII notation (`mu`, `sqrt(x)`, `x^2`); `tests/test_docs.py` also checks this.

## Brief to code

| Brief | Code (`src/capacity_study/`) | Config (`configs/study.toml`) | Tests (`tests/`) |
|---|---|---|---|
| S1 The forecasting problem | `train.py` (loss, R^2 against persistence) | `[data]` | `test_pipeline.py` |
| S2 From prices to inputs | `data/features.py`, `data/bank.py` | `[data]` | `test_features.py` |
| S3 The data universe | `universe.py`, `data/fetch.py`, `data/splits.py` | `[data]` | `test_splits.py` |
| S4 The network | `model.py` | `[training]` | `test_pipeline.py`, `test_gpu.py` |
| S5 How it learns | `optim.py`, `train.py` | `[training]` | `test_optim.py`, `test_gpu.py` |
| S6 The recipe space | `recipes.py` (grid, job packing) | `[grid]` | `test_pipeline.py` |
| S7 Hyperparameter tuning | `tuning.py`, `orchestrate.py` (freeze, enqueue) | `[tuning]` | `test_pipeline.py` |
| S8 Seeds and error rates | `rng.py`, `train.py`, `analysis/status.py` (margins) | `[study]`, `[seeds]`, `[analysis]` | `test_pipeline.py`, `test_analysis.py` |
| S9 Measurements | `train.py` (metrics), `analysis/quantities.py` | `[training]`, `[analysis]` | `test_analysis.py` |
| S10 Recipe status | `analysis/status.py` | `[analysis]` | `test_analysis.py` |
| S11 Characteristics | `analysis/characteristics.py` | | `test_analysis.py` |
| S12 Regionalization and visual outputs | `analysis/regions.py`, `analysis/figures.py`, `recipes.py` (adjacency) | | `test_analysis.py`, `test_pipeline.py` |
| S13 Running it | `orchestrate.py`, `ledger.py`, `store.py`, `bench.py`, `cli.py` | `[tuning]` (calibration set) | `test_pipeline.py`, `test_gpu.py` |
| S14, S15 Registers and literature | | | `test_docs.py` (guide sync, notation) |
