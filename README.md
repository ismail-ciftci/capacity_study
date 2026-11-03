# Capacity study

Characteristics of training-saturated MLPs in the bias regime. Spec: `docs/PROJECT_BRIEF.md`, explained in
`docs/PROJECT_GUIDE.md`; `docs/README.md` maps the docs to the code. Implementation conventions: `CLAUDE.md`.

## Running the study

All commands are `uv run python -m capacity_study <command> ...`. Commit the code first: workers refuse a dirty tree
unless `--allow-dirty`.

```
fetch-data   --config configs/study.toml                 # one time; data/raw/ is committed, never refetched
prepare      --config configs/study.toml                 # feature bank -> cache/
init-study   --config configs/study.toml --study runs/main
bench        --study runs/main --device cuda:0           # pick --workers-per-device from steady-state throughput

enqueue      --study runs/main calibration
launch       --study runs/main --devices cuda:0,cuda:1 --workers-per-device W
freeze-ranges --study runs/main                          # after calibration completes; writes ranges.json once
enqueue      --study runs/main tuning                    # adds tuning + final jobs (final waits on its tuning job)
launch       --study runs/main --devices cuda:0,cuda:1 --workers-per-device W
```

Control while running:

```
status  --study runs/main
pause   --study runs/main          # workers checkpoint at the next epoch end and exit (Ctrl+C does the same)
resume  --study runs/main --devices cuda:0,cuda:1 --workers-per-device W
retry   --study runs/main <job_id> # failed jobs are never retried automatically
```

A crash or power loss loses at most one checkpoint interval (25 epochs) of one job; `launch` returns jobs without a
recent heartbeat to the queue, and resumed jobs reproduce an uninterrupted run bit for bit.

## Layout

- `src/capacity_study/` — `data/` (fetch, Haar features, feature bank, splits), `model.py` (stacked MLP),
  `optim.py` (batched schedule-free AdamW port), `train.py` (one job), `tuning.py`, `ledger.py` (SQLite queue),
  `orchestrate.py` (study lifecycle, worker, launcher), `bench.py`, `cli.py`.
- `configs/study.toml` — every result-affecting value.
- `runs/<study>/` — `study.json`, `ledger.sqlite`, `ranges.json`, `jobs/<job_id>/` (`job.json`, `events.jsonl`,
  `metrics/`, `preds/`, `tuning/`, weights, `DONE.json` written last).

## Tests

`uv run pytest` (GPU tests run when a GPU is visible). The optimizer port is checked member by member against the
pinned `schedulefree` package.
