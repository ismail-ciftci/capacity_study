# capacity_study

Type: experiment

Characterizes training-saturated MLPs in the bias regime over a closed grid of 480 recipes.
Spec: `PROJECT_GUIDE.md` (the brief it references is not in the repo; the guide is treated as the spec).
Where the guide is silent, the decision is recorded below — warn the user about any new gap instead of deciding silently.

## Conventions

### Vocabulary
- **recipe**: one option per factor (depth, width, regularization, tier, history); `recipe_id` = index in
  `product(depth, width, reg, tier, history)` in that order.
- **trial**: one hyperparameter configuration of a recipe in the tuning stage (32 per recipe).
- **seed**: one final-stage training run of a recipe (32 per recipe).
- **member**: one slice of a batched model (a trial or a seed).
- **job**: the unit of execution and of results — one batched model packing the two regularization recipes of
  one (phase, tier, history, depth, width); K = 64 members.
- **phase**: `calibration` | `tuning` | `final`.
- **study**: one directory `runs/<study>/` bound to one config hash.

### Results integrity
- Only `DONE.json` (written last) marks a job complete. Completed jobs are never overwritten.
- Workers refuse a dirty git tree unless `--allow-dirty` (diff is recorded), and refuse to resume a checkpoint from
  another commit unless overridden (override is recorded).
- Analysis outputs are derived and regenerable; analysis code only reads raw job outputs.
- Every result-affecting value lives in `configs/*.toml`; no defaults in code for such values.

### Data
- Universe: one-time hard-coded ticker list in `src/capacity_study/universe.py`; not an experiment parameter.
  Test asset JPM. Survivorship bias accepted (C5). Current Yahoo tickers are used for renamed firms: BNY (was BK),
  MRSH (was MMC). Snapshot: 2006-08-22 to 2026-10-30, 5,080 dates, no missing values.
- Prices: split- and dividend-adjusted close (yfinance `auto_adjust=True`). Raw snapshot is committed with a sha256
  manifest; never refetched automatically.
- Data end 2026-10-30. Origin = end − 2 calendar years. History start = origin − {3, 7, 18} calendar years.
  Tuning cutoff = history start + 0.8 × (origin − history start), calendar time.
- Window membership in a stage/history is decided by its **target date**.
- No forward fill; windows containing a missing day are dropped.
- DWT ordering and sign follow the guide pseudocode exactly: details finest→coarsest, (a − b)/√2, then the
  approximation, then pstd over 4, 8, 16 days.

### Training
- Optimizer: `BatchedAdamWScheduleFree`, a vectorized port of `schedulefree.AdamWScheduleFree` (pinned version)
  with per-member lr/λ tensors, verified against the package in tests. It follows the current package, which
  applies Adam bias correction in the denominator (the guide/paper describe it folded into lr).
- Batched forward/backward via stacked `bmm`; datasets live on the GPU for the worker's lifetime; no host
  transfers per step.

### Tuning
- Halton points: `scipy.stats.qmc.Halton(scramble=True, seed=<fixed>)`, same seed for every recipe; results
  stored in Optuna via `add_trial` (Optuna's QMCSampler samples the first ask-ahead trial randomly).
- Infeasible trial: a non-finite loss or parameter at any step; worst objective; Optuna state FAIL.
- Ties in the objective go to the lowest trial index.
- Edge hit: chosen value within (log range)/32 of either range end, per dimension.
- Untunable recipe (all trials infeasible): final stage skipped; analysis assigns sentinel quantities from config
  (outside any real result, asserted), status label `untunable`; two untunable recipes merge (D = 0), never with
  tunable ones; figures draw them in a fixed color outside the scale.

### Statistics
- Variance estimators use ddof = 1 (seed statistics and T1's across-seed variance).

### Hardware
- AMD RX 6650 XT (8 GB, ROCm gfx1032) on Windows; a second identical card will be added. Devices are a config/CLI
  list; several worker processes per GPU, count chosen by `bench`.
- Bench (2026-11-02, one RX 6650 XT, 64-member packs, steady-state s/epoch at W=1): d1w8 tier1/3y 0.022,
  d4w8 tier3/18y 0.055, d8w64 tier4-5/18y 1.02. Only tiny jobs scale with more workers; big ones saturate the GPU,
  and 4+ concurrent big jobs thrash (VRAM). Use `--workers-per-device 3` and let the memory-budget claim rule
  (estimates ≈ 2× measured peak) keep big jobs to 2 per GPU. Whole study ≈ 8.5 GPU-hours.
