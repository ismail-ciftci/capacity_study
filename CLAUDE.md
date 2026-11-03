# capacity_study

Type: experiment

Characterizes training-saturated MLPs in the bias regime over a closed grid of 480 recipes.

- Spec, source of truth: `docs/PROJECT_BRIEF.md` (agent-facing). `docs/PROJECT_GUIDE.md` teaches it for onboarding
  and adds no decisions; guide chapter n explains brief section Sn, and the guide's readability decides structure where
  the two are coupled. Any brief change is carried into the guide in the same change: `tests/test_docs.py` names the
  stale chapters and prints their new `Synced:` lines. `docs/README.md` maps the docs to code and gives the change
  protocol.
- Literature: `docs/literature/index.md`; originals and searchable text extractions are local only
  (`docs/literature/sources/`, `docs/literature/text/`, gitignored). Grep `text/` to check a `[paper: X]` claim.
- Docs hold current guidance only; history lives in git. Do not add run reports, change logs or "removed" notes.
- Where the brief is silent, warn the user instead of deciding silently. Definition-level answers (what is computed)
  go into the brief once the owner confirms; implementation-level answers go under Conventions below.

## Brief ahead of code

Parts of the brief that code, config or tests do not follow yet; remove each line once they do.
- None currently.

## Conventions

### Vocabulary
- **recipe**: one option per factor (depth, width, regularization, tier, history); `recipe_id` = index in
  `product(depth, width, reg, tier, history)` in that order.
- **trial**: one hyperparameter configuration of a recipe in the tuning stage (16 per recipe, 4 seeds each).
- **seed**: one training run: a trial's seed j in tuning (shared by all trials of the recipe), or one of the 32
  final-stage runs of a recipe.
- **member**: one slice of a batched model: a (trial, seed) pair in tuning, a seed in the final stage.
- **job**: the unit of execution and of results: one batched model packing the two regularization recipes of
  one (phase, tier, history, depth, width); K = 128 members in tuning and calibration, 64 in the final stage.
- **phase**: `calibration` | `tuning` | `final`.
- **study**: one directory `runs/<study>/` bound to one config hash.

### Results integrity
- Only `DONE.json` (written last) marks a job complete. Completed jobs are never overwritten.
- Atomic writes retry `os.replace` on `PermissionError` with backoff, then re-raise: on Windows, antivirus or the
  indexer can hold a transient handle on a checkpoint.
- Workers refuse a dirty git tree unless `--allow-dirty` (diff is recorded), and refuse to resume a checkpoint from
  another commit unless overridden (override is recorded).
- Analysis outputs are derived and regenerable; analysis code only reads raw job outputs.
- Every result-affecting value lives in `configs/*.toml`; no defaults in code for such values.

### Data (brief S2, S3)
- Universe: one-time hard-coded ticker list in `src/capacity_study/universe.py`; not an experiment parameter.
  Test asset JPM. Survivorship bias accepted (C5). Current Yahoo tickers are used for renamed firms: BNY (was BK),
  MRSH (was MMC). Snapshot: 2006-08-22 to 2026-10-30, 5,080 dates, no missing values.
- Prices: split- and dividend-adjusted close (yfinance `auto_adjust=True`). Raw snapshot is committed with a sha256
  manifest; never refetched automatically.
- Data end 2026-10-30. Origin = end - 2 calendar years. History start = origin - {3, 7, 18} calendar years.
  Tuning cutoff = history start + 0.8 x (origin - history start), calendar time.
- Window membership in a stage/history is decided by its **target date**.
- No forward fill; windows containing a missing day are dropped.
- DWT ordering and sign: details finest to coarsest, each (a - b)/sqrt(2), as in the brief S2.
- Bank (`data/bank.py`) stores per window the 19 features, the realized return y, scale = sigma and
  offset = mu - log P(t). The implied return is scale * output + offset; loss, R^2, T1 and prediction snapshots use
  it, and prediction spread uses the raw output (stored as `eval_output` in final snapshots).

### Training (brief S4, S5)
- Optimizer: `BatchedAdamWScheduleFree`, a vectorized port of `schedulefree.AdamWScheduleFree` (pinned version)
  with per-member lr/lambda tensors, verified against the package in tests. Bias correction sits in the Adam
  denominator, as in the package (recorded in brief S5).
- Batched forward/backward via stacked `bmm`; datasets live on the GPU for the worker's lifetime; no host
  transfers per step.

### Tuning (brief S7)
- Halton points: `scipy.stats.qmc.Halton(scramble=True, seed=<fixed>)`, same seed for every recipe; results
  stored in Optuna via `add_trial`.
- Member objective: mean validation R^2 over the last `objective_tail` epochs; a member is infeasible if any loss
  or parameter is non-finite at any epoch. Trial objective: mean over its seeds, infeasible if any seed is (worst
  objective, Optuna state FAIL). The infeasible fraction counts trials. `recipe_<id>.json` keeps the per-seed
  objectives as `seed_objectives`.
- Ties in the objective go to the lowest trial index.
- Edge hit: chosen value within (log range)/n_trials of either range end, per dimension.
- Untunable recipe (all trials infeasible): final stage skipped; analysis assigns sentinel quantities from config
  (+/-1e12, outside any real result; asserted >= 10x beyond every real value, since barely trained nets can reach
  R^2 ~ -1e4), status label `untunable`; two untunable recipes merge (D = 0), never with tunable ones; figures draw
  them in a fixed color outside the scale.

### Statistics (brief S8)
- Variance estimators use ddof = 1 (seed statistics and T1's across-seed variance).

### Hardware (brief S13)
- Two AMD RX 6650 XT (8 GB, ROCm gfx1032) on Windows. Devices are a CLI list; several worker processes per GPU,
  count chosen by `bench`.
- Bench (one RX 6650 XT, 64-member packs, steady-state s/epoch at one worker): d1w8 tier1/3y 0.022,
  d4w8 tier3/18y 0.055, d8w64 tier4-5/18y 1.02. Only tiny jobs scale with more workers; big ones saturate the GPU,
  and 4+ concurrent big jobs thrash (VRAM). Use `--workers-per-device 3` and let the memory-budget claim rule
  (estimates ~2x measured peak) keep big jobs to 2 per GPU. Measured (16x4 tuning): calibration ~17 min, main run ~4.6 h wall
  on both GPUs.
