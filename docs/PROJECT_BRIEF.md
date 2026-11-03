# Project Brief: Characteristics of Training-Saturated MLPs in the Bias Regime

Status: living document. Last updated 2026-11-03.

**Conventions**
- **Ascertained**: stated or confirmed by the project owner.
- **Left open**: undecided. Nothing under "Left open" is a decision. Assumptions in use are marked *Working reading:* and live here, not under Ascertained.
- **Parked**: a Left-open item that is evaluated only when the owner asks. It is not raised, reopened, or used as grounds for suggestions otherwise; its working reading is used as is.
- **Flags**: short warnings, not decisions. Each section keeps only its local flags; warnings that cut across sections live once in the confound register (S14) and are referenced as C1-C9.
- **Gaps**: choices without literature backing live once in the evidence-gap register (S15) and are referenced as G1-G17. A gap is not a warning that the choice is wrong; it marks where more research could strengthen the study.
- **Evidence tags**: [calc] own arithmetic; [measured] computed on project data; [estimate] not computed on project data; [inference] reasoned, not tested; [speculation]; [paper: X] from a project paper (listed in `literature/index.md`); [from memory] to be verified. Untagged Ascertained items come from the owner.
- **Current guidance only**: the brief states what holds now. Past versions, removed items and the runs that led to changes live in git history, not here. Exclusions are kept as rules so they are not proposed again.
- Sections are ordered so that each depends only on the sections before it.
- Register IDs (C, G) are never reused; a removed entry is deleted and its number stays unused.
- The brief defines what is computed. Implementation tools not named here are chosen at implementation.
- **Companion guide**: `PROJECT_GUIDE.md` explains this brief for human readers, as a lecture book with a worked example. Brief section Sn and guide chapter n cover the same ground; where the two documents are coupled, the guide's readability decides the structure. The guide adds no decisions; if the two disagree, the brief wins. Each guide chapter records the hash of the brief section it was last checked against, and `tests/test_docs.py` fails when a section changes without its chapter being reviewed.

## S0. Purpose

Characterize MLPs that reach training saturation in the bias regime, for the forecasting problem, representation and data universe defined in S1-S3. The unit of study is the full grid of recipes, each trained with its own tuned hyperparameters over multiple seeds. All recipes are analyzed for training and test R^2, the generalization gap, and the prediction variance across seeds, with the seed std of every reported quantity accounted for; recipe status is one categorical characteristic among them. The main quantities are regionalized: the recipes are partitioned into connected regions with equivalent results, without predefined borders (S12). Findings about the bias regime are read inside valid-status regions. Failure rates and data saturation are characterized over the whole population. The aim is the characteristics of each choice, not a performance ranking. The recipe space is fixed; changing it based on results is out of scope. The characterized recipes are meant for use in a later project, which must train them with the same protocol (S5) for the characteristics to carry over.

## Glossary

Each entry defines one term, uses only plain words or terms defined above it, and carries no numbers (numbers live in their section).

- **Data universe**: the pool of assets and dates the study draws from (S3).
- **Data tier**: the subset of the data universe's assets that a network trains on; one of five (S3).
- **Test asset**: the single asset on which every recipe is evaluated; it is the one-asset tier's asset and belongs to every tier.
- **Window**: one sample: a lookback input ending at its last input date t, together with its target.
- **Window statistics**: the mean mu and population standard deviation sigma of a window's log prices (S2).
- **Target**: the log closing price one trading day after the last input date, log P(t+1) (S1).
- **Forecast**: the network output mapped back with the window statistics, mu + sigma * output (S2).
- **Implied return**: the forecast minus log P(t); every R^2 is computed on it (S1).
- **Task**: predicting a window's target from its input (S1).
- **Factor / option**: a recipe dimension and one of its possible values (S6).
- **Recipe**: one combination of options, one per factor.
- **Seed**: one training run of a recipe. Seeds differ in initialization and data order (S8).
- **Evaluation weights**: the running weighted average of the optimizer's base iterates, which schedule-free AdamW returns as the model; every measurement reads them (S5).
- **Timeline** (by calendar date, all assets):

```
history start -- tuning-stage training --|tuning cutoff|-- validation window --|origin|-- test window
              +---------------- final-stage training data -------------------+
```

  - **Tuning stage**: trains on tuning-stage training data and tunes hyperparameters on the validation window.
  - **Final stage**: retrains with the tuned hyperparameters on the final-stage training data (validation window merged in). Its results are the ones analyzed.
  - **Cutoff**: the tuning cutoff or the origin, the last date of a stage's training data.
- **Trial**: one hyperparameter configuration of a recipe, trained in the tuning stage (S7).
- **Main run**: the run of every recipe under one tuning protocol; its results are the ones analyzed (S13).
- **Calibration run**: a small set of recipes at the extremes of the grid, tuned before the main run with wide ranges; their best values, widened by a margin, set the main run's search ranges. Its results are otherwise discarded (S7, S13).
- **Test-asset training windows**: the test asset's windows in the final-stage training data.
- **R^2**: the comparison metric: R^2 of the implied return relative to naive persistence (an implied return of zero), computed on the test asset.
- **Test curve**: R^2 per epoch on the test window, in the final stage.
- **Training curve**: R^2 per epoch on the test-asset training windows, in the final stage.
- **Training saturation**: the end of training, where both the training curve and the test curve are flat (measured as V1, S9).
- **Prediction variance across seeds** (T1 is built from it): for one window, the variance across seeds of the forecast. Seeds share the target, so it equals the variance of the t+1 error.
- **Seed std**: the standard deviation across seeds of a scalar quantity, such as test R^2.
- **Generalization gap**: training R^2 minus test R^2.
- **Bias regime**: the regime where a training-saturated network's error is dominated by underfitting rather than overfitting.
- **Recipe status**: valid (training-saturated and in the bias regime), failure-unsaturated, or failure-overfit (S10).
- **Data saturation**: the point where more data stops helping; defined over the population only, never per recipe. Distinct from training saturation.
- **Adjacent**: two recipes are adjacent if they differ in one factor by one step; for unordered factors any two options are one step apart (S12).
- **Region**: a connected set of adjacent recipes whose results for one quantity are equivalent (S12).
- **Regionalization**: the procedure that partitions the recipes into regions, separately for each mapped quantity (S12).
- **Characteristic**: a measured property of a recipe, a choice, or a region, not a performance ranking (S11, S12).

## S1. The forecasting problem

**Ascertained**
- Task: from the last 16 trading days of one asset's closing prices, predict the log closing price one trading day ahead. Target: log P(t+1), one step ahead (k = 1).
- The forecast is built from the network output and the window statistics (S2).
- Loss: MSE between forecast and target, in log-price units. It equals the MSE of the implied return against the realized return [calc].
- Comparison metric: R^2 on t+1 of the implied return, not the training loss. Persistence, an implied return of zero, is the reference.
- Train-vs-test comparisons (R^2, gap, T1) use the test asset only, on both sides, with the persistence baseline computed on the same windows. The network still trains on all assets of its tier (S3); this decouples training-set diversity from the comparisons.
- Time-invariance of MLPs is not a research question.

**Left open**
- None currently.

**Flags**: C1, C7.

## S2. From prices to inputs

**Ascertained**
- Lookback: 16 trading days.
- Input, built per window from its 16 log closing prices p_1 ... p_16 (p_16 = log P(t)):
  1. Window statistics: mu = mean of the 16 log prices; sigma = their population standard deviation (divide by n).
  2. Normalized window: (p - mu)/sigma, so every window has mean 0 and standard deviation 1 [paper: Liu, eq. 1].
  3. Orthonormal Haar DWT of the normalized window, (a+/-b)/sqrt(2) at each stage, computed inside the window (causal by construction), at all scales: 15 detail coefficients (8+4+2+1). The approximation coefficient is 0 by construction and is not an input; the level it would carry enters through mu.
  4. Volatility features: the population standard deviation of the normalized log prices over the last 4 and 8 days (over 16 days it is 1 by construction). Their purpose is to spare small networks a nonlinearity they may not approximate, not to add information.
  5. Re-injected statistics: mu and log sigma, appended to the input, so that windows that differ only in level or scale stay distinguishable [paper: Liu: normalization alone over-stationarizes; re-incorporating mu and sigma helps, Table 6, feed-forward variant; appending them to an MLP's input is inference].
  Input dimension: 15 + 2 + 2 = 19. No further scaling or standardization.
- Output: the network predicts the target in window units, (log P(t+1) - mu)/sigma; the forecast is mu + sigma * output [paper: Liu, eq. 2]. A zero output forecasts the window mean. Persistence, (p_16 - mu)/sigma, is a linear function of the normalized details, so reaching it requires using the input [calc].
- The DWT does not apply to the target.

**Left open**
- A window whose sigma is zero (16 equal closes) cannot be normalized. *Working reading:* it is dropped, like a window with a missing day; the current snapshot has none [measured].

**Flags**: C2.

**Gaps**: G1.

## S3. The data universe

**Ascertained**
- US equities. Main sector: financials. Assets pooled by sector; history up to 20 years.
- Five data tiers by asset count and composition:

| Tier | Assets | Composition |
|---|---|---|
| 1 | 1 | financial (the test asset) |
| 2 | 4 | financials |
| 3 | 12 | financials |
| 4 | 24 | financials (similar) |
| 5 | 24 | diverse: 4 each from financials, industrials, health care, information technology, consumer discretionary, utilities |

  Tiers 4 and 5 have the same asset count and therefore the same data size. The financial subsets are nested (tier 1 within 2 within 3 within 4), and tier 5's four financials are tier 2's four.
- Excluded: a 44-asset tier (the 24 financials plus tier 5's 20 non-financials).
- "Consumer" means the GICS consumer discretionary sector.
- Days are trading days.
- History length: 3, 7, or 18 years, counted back from the origin. History includes the validation window, so it is the whole final-stage training data. Tier and history together span 1 asset x 3 years to 24 assets x 18 years, a factor of 144 in data, roughly 750 to 108,000 windows [calc].
- Stages: the tuning stage trains on the first 80% of the final-stage training data and tunes on the last 20% (the validation window). In the final stage the validation window is merged in, so training ends at the origin, and the test window (the last 2 years of the data) follows immediately. Reason for the merge: without it, the test window would lie too far from the training end. Splits are by calendar date across all assets.
- Derived sizes [calc]: about 250 windows per year; test-asset training windows about 750 (3 years) to about 4,500 (18 years); validation window about 0.6, 1.4, and 3.6 years (about 150 windows at tier 1, 3 years); test window about 500 windows.
- No embargo at either cutoff. No in-period purged holdout.
- Boundary convention: the cutoff is the last target date. A window belongs to a stage if its target date is on or before that stage's cutoff, so its last input date is at least one trading day earlier. Every asset in every tier follows this rule. No label crosses a cutoff.
- Input overlap at a cutoff (a held-out window's 16-day input reaching into training dates) is not label leakage.
- The asset draw, including the test asset, is fixed for the whole study; it is not redrawn per seed, which allows batched seeds.
- Reference counts (snapshot Oct 2026, public S&P 500 constituent data): 36 financials with index tenure of about 20 years or more; other sectors 6 to 34.

**Left open**
- None currently.

**Flags**: C4, C5, C6, C7.

**Gaps**: G2.

## S4. The network

**Ascertained**
- A plain MLP: the 19 inputs (S2), a number of hidden layers (depth) of one constant width, and one linear output. Depth and width are factors (S6).
- Activation: leaky ReLU at the PyTorch default slope of 0.01 (not tuned). No normalization layers, skip connections, or dropout (S6).
- Initialization: PyTorch default for linear layers, fixed and observed (option A); the evaluation weights start at the initialization. Weights and biases are uniform in +/-1/sqrt(fan_in), so weight variance is 1/(3*fan_in) [from memory]. Options B (fixed small init), C (width-aware scaling) and D (init scale as a factor) were not adopted.

**Left open**
- None currently.

**Flags**
- The PyTorch default is not He initialization (variance 2/fan_in, about 6 times larger) [from memory].
- See also C2, C9.

**Gaps**: G7.

## S5. How it learns

**Ascertained**
- Optimizer: schedule-free AdamW [paper: Defazio]. It keeps three sequences: the base iterate z, where the Adam step is taken; the gradient point y = (1 - beta)*z + beta*x, where gradients are evaluated; and the evaluation weights x, a weighted running average of z. The model is x [paper: Defazio, Sec. 2, Alg. 1].
- Fixed optimizer settings: beta = 0.9 [paper: Defazio: works on all problems but one, and its best value does not change with training length, Sec. 4.1, Sec. 5]; beta2 = 0.999 [paper: Defazio, App. G: used in most tasks]; eps = 1e-8 [from memory: implementation default]; averaging weights proportional to the squared learning rate, so warmup steps get little weight (c_{t+1} = gamma_t^2 / sum(gamma_i^2)) [paper: Defazio, eq. 23].
- Weight decay, when on, applies to all weight matrices, biases excluded, at the gradient point y as in schedule-free AdamW (z <- z - gamma_t*lambda*y) [paper: Defazio, Alg. 1, Sec. 4.4].
- Warmup: linear, 5% of steps (625 steps, 25 epochs [calc]), part of the optimizer; required [paper: Defazio, Sec. 4.4: "Learning rate warmup is still necessary for our method"]. Defazio et al. used 5% in their deep-learning experiments and 2% in their AlgoPerf entry [paper: Defazio, App. G]. The longer value was chosen because the smallest batches raise the risk of early instability [inference]. Warmup steps carry about 1.7% of the averaging weight in the final evaluation weights, and the warmup costs about 400 full-rate steps, about 3% of training [calc].
- No schedule beyond the warmup: after warmup the base learning rate is constant. The evaluation weights behave like a model trained with a linear decay ending at the current step, so the method applies a schedule implicitly [paper: Defazio, Sec. 2.3, Fig. 4].
- Training length is fixed in epochs (about 500, heuristic); not a factor and not tuned. This also removes any need to choose an epoch count from the validation window before the final-stage retrain.
- Batch size is coupled to dataset size: K = 25 steps per epoch, so B ~ N/25 and every recipe gets about 12,500 steps. Each epoch splits the N windows into 25 near-equal batches (sizes differ by at most 1). B runs from about 30 to about 4,300 in the final stage [calc].
- Implementation: a batched port of the reference implementation (facebookresearch/schedule_free, pinned version), verified against it member by member. Like the reference, it applies Adam's bias correction in the denominator, sqrt(v/(1 - beta2^(k+1))) + eps, rather than folding it into the learning rate as the paper describes [paper: Defazio, Sec. 4.4]; the averaging weights then use the warmup-scaled learning rate only.
- No batch-statistics normalization, so schedule-free AdamW needs no batch-norm handling [paper: Defazio, Sec. 4.4].
- The later project must train with schedule-free AdamW and use the evaluation weights for the characteristics found here to apply.

**Left open**
- None currently.

**Flags**
- Where the bias correction sits may change the effect of weight decay in early steps [paper: Defazio, Sec. 4.4].
- eps is fixed while gradients are small (C2) (G10).
- See also C2, C6.

**Gaps**: G6, G10.

## S6. The recipe space

**Ascertained** (all factors are scientific)

| Factor | Options |
|---|---|
| Depth (number of hidden layers, S4) | 1, 2, 4, 8 |
| Width (constant hidden width, S4) | 8, 16, 32, 64 |
| Regularization (S5) | none; weight decay |
| Data tier (S3) | 1, 4, 12, 24, 24 (diverse) assets |
| History (S3) | 3, 7, 18 years |

- **Fixed:** loss MSE and a one-step target (S1); activation leaky ReLU at slope 0.01 (S4); optimizer schedule-free AdamW (S5). Findings hold for MSE, schedule-free AdamW, this leaky ReLU, and a one-step target only; loss shape, optimizer type, activation type, and horizon are out of scope.
- Excluded from the space: a horizon factor (multi-step targets); early dropout; regularization "both" (it only adds the interaction); MAE loss.
- Excluded from scope: Huber and other losses, other optimizers (including AdamW with a schedule and SGD with momentum), other activations (SiLU and gated), dropout in any form, L1 regularization, normalization layers (the window normalization of S2 has no parameters; it is part of the input and output mapping, not a layer), gradient clipping, network shape, skip connections, explicit learning-rate schedules (schedule-free AdamW applies one implicitly; its fixed warmup is part of the optimizer, S5).
- Size: 10 categorical combinations (regularization x data tier); full grid 480 with depth, width, and history [calc]. Every recipe is run once (S13); the space is not changed based on results.

**Left open**
- None currently.

**Flags**: C6, C9.

**Gaps**: G4, G5.

## S7. Hyperparameter tuning

**Ascertained**
- Performed per recipe, in the tuning stage, so each recipe is on fair footing; a means, not the subject of study. Trial bookkeeping and storage are handed to Optuna.
- Tuned (nuisance): learning rate, and weight-decay strength lambda when weight decay is on. Recipes without weight decay tune the learning rate only. beta, beta2, eps and warmup are fixed (S5). Basis: Defazio et al. tune learning rate and weight decay and name this sweep the method's main practical need, because the best values differ from those under a schedule [paper: Defazio, Sec. 4.1, Sec. 6].
- Method: quasi-random search over scrambled Halton points with a fixed seed [paper: Playbook], the same points for every recipe of the same dimension. The search is non-adaptive, so all trials of a recipe are drawn up front, with scipy's Halton generator (the engine of Optuna's quasi-Monte Carlo sampler, which draws the first trial of an ask-ahead batch at random). Every trial runs the full 500 epochs; no pruning.
- Each trial is trained with 4 seeds, and the objective of a trial is the mean over its seeds, so that selection does not reward a setting that was lucky on one seed. Seed j has the same initialization and data order in every trial of a recipe, so comparisons between trials are paired seed by seed [paper: Bouthillier]. Mean over seeds was chosen over single-seed trials because, with one seed, seed noise in the objective is larger than the differences between learning rates, so the best trial is largely the luckiest and the maximum favours settings with high seed variance [measured on a project run].
- A recipe's trials and their seeds are trained as one batched model, one member per (trial, seed), each with its own hyperparameters. Members do not interact: each has its own weights and optimizer state, and the averaging weights c_t do not depend on the learning-rate scale (it cancels in gamma_t^2 / sum(gamma_i^2)), so they are the same for every member [calc]. The models are small, so extra batched trials cost little memory.
- One tuning protocol for every analyzed recipe: the shared ranges below and 16 trials x 4 seeds per recipe, the same for recipes with and without weight decay and the same as in the calibration run.
- Objective (maximized): validation R^2 on t+1 of the evaluation weights, test asset, mean of the last 25 epochs, averaged over the trial's seeds.
- Wide ranges (calibration only): learning rate log-uniform 1e-5 to 1e-1 [from memory]; weight decay log-uniform 1e-5 to 1 [from memory]. The upper end leaves room for schedule-free learning rates, which ran about 1-10 times the base optimizer's in Defazio et al.'s experiments (for example 0.005 against 0.0005 on NanoGPT) [paper: Defazio, Sec. 4.1, App. G].
- Main-run ranges: one shared range per tuned hyperparameter for all recipes, not one per tier x history cell. The best learning rate moves with batch size (for Adam-style optimizers it rises with about sqrt(B), peaks, then falls) [paper: Li et al., eq. 4, Sec. 6; transfer to schedule-free AdamW is inference] and with width [from memory]; per-cell ranges would follow batch size only and would give each data cell its own search space, so the data-size factor would carry tuning quality (C6).
- Range rule: each main-run range spans from the lowest to the highest best value found in the calibration run, widened by 0.75 decade on each side. Because of the margin, the exact best values found in calibration matter little.
- Edge hits: a main-run recipe whose chosen value lies within one trial spacing of a range end is flagged, reported, and marked on the lattices (S12). It is not re-tuned, which would break the single protocol. The share of edge hits is reported [paper: Choi; Playbook: edge check].
- Tuning-to-final transfer: tuned values are reused unchanged in the final stage. The step count is the same (25 steps per epoch, 500 epochs) and the batch is about 1.25 times larger [calc]. No project paper covers this transfer for schedule-free AdamW (G8); Defazio et al. cover training length only, where the best learning rate is stable except for short runs [paper: Defazio, Sec. 5].
- Weight-decay convention: schedule-free AdamW at y (S5); lr*lambda is reported alongside lambda.
- Leakage in tuning splits: no label crosses a cutoff (S3 boundary convention); input overlap only. No separate handling.
- Infeasible trials (any seed diverged or NaN) are recorded as failed with the worst objective; the infeasible fraction per recipe is reported.
- Tuned hyperparameters are used in the final stage, with the validation window merged into training.

**Left open**
- None currently.

**Flags**
- Tuning-effort asymmetry: with one trial count, weight-decay recipes spread their trials over two dimensions and no-decay recipes over one. If the best learning rate depends on lambda, weight-decay recipes are tuned less precisely, and the regularization factor partly carries tuning quality [inference].
- Batched trials cost little memory, but compute still grows with the number of members once the hardware is saturated [inference].
- A calibration best value at an end of the wide range puts the main-run range beyond what calibration tested, by the margin.
- Paired trials may lose most of their pairing by epoch 500: a one-bit change at initialization produces about as much variability as any other source [paper: Summers], and trials differ in hyperparameters from the first step [inference].
- Four seeds halve the seed noise of the objective but do not remove it; it is largest at tier 1 with 3 years (C4).
- See also C2, C6.

**Gaps**: G8, G9.

## S8. Seeds and error rates

**Ascertained**
- Every metric's variance is accounted for, with multiple seeds per recipe.
- Seed count: 32. The relative error of a seed std is about 13% at 32 seeds [calc]. Bouthillier et al. recommend about 29 runs for their comparison test [paper: Bouthillier, App. C.3].
- One seed sets initialization and data order. Bundling them is supported: all sources of nondeterminism had similar effects on run-to-run variability, within about 20% of each other [paper: Summers]. Each seed draws from its own random stream, independent of which other seeds share the batch. Seed streams are independent across recipes: seed *i* of one recipe shares nothing with seed *i* of another, so comparisons between recipes are unpaired (S12). In tuning, seed j of a recipe is shared by all its trials (S7).
- Seeds run batched and share the same data (the draw is fixed). Each seed gets its own minibatches from its own shuffle [inference: arithmetic is unchanged; memory is small].
- Error rates: alpha = 0.05, beta = 0.2 (provisional).
- Margins: dimensionless, derived from alpha, beta and n only, fixed for every comparison. delta_mu = (t_(1-alpha) + t_(1-beta/2))*sqrt(2/n) in units of seed sigma, with 2(n - 1) degrees of freedom; this is the equivalence-test sample-size equation solved for the bound, and setting the bound to the smallest effect detectable with the available sample follows Lakens's recommendation when no theory sets it [paper: Lakens, eq. 5]. delta_sigma = (z_(1-alpha) + z_(1-beta/2))/sqrt(n - 1) on a log std ratio. At n = 32: delta_mu ~ 0.74 sigma, delta_sigma ~ 0.53 (ratio about 1.7) [calc].
- Hypothesis tests are used only inside regionalization, to decide merges (S12); no test result is reported as a finding. Status compares point estimates with fixed thresholds, without a test (S10).

**Left open**
- None currently.

**Flags**
- Reproducibility holds for the distribution across seeds, not for individual seeds. A different batching of seeds can change floating-point order, and under training instability any such change produces seed-level divergence [paper: Summers; the batching mechanism is inference].
- Data order is about as large a source of variance as initialization; data sampling is larger than both [paper: Bouthillier] (C7).
- See also C7.

## S9. Measurements

Every quantity is defined once, here. All use the t+1 output, the final stage, and the evaluation weights (S5). Every quantity is reported for every recipe, together with its seed std. T1 is itself computed across seeds; its uncertainty is a jackknife over seeds (S12). Which quantities are regionalized is defined in S12.

| Quantity | Definition | Read on | Feeds status |
|---|---|---|---|
| V1a slope | Slope of the training curve over the last 10% of epochs (about 50 of 500): mean of the last 25 epochs minus mean of the preceding 25, in absolute R^2 units; not divided by 25. Reported as the magnitude of the seed-mean slope and the seed std of the slope | Test-asset training windows | Yes |
| V1b slope | Same construction on the test curve | Test window, test asset | Yes |
| T1 | Mean over windows of the prediction variance across seeds, on each side. Reported per side and as the log ratio 0.5*log(T1_test / T1_train) (the log of sqrt(T1_test) / sqrt(T1_train)), positive in the overfitting direction like the gap. Seed variance (initialization, data order) on fixed data | Training side: test-asset training windows. Test side: test window | Yes (the log ratio) |
| Training R^2 | R^2 on t+1, mean of the last 25 epochs of the training curve | Test-asset training windows | No |
| Test R^2 | R^2 on t+1, mean of the last 25 epochs of the test curve | Test window | Only through V1b |
| Generalization gap | Training R^2 minus test R^2, per seed | Both | No (never a filter) |
| Prediction spread | Std over windows of the network output in window units, per seed; zero for a network that ignores its input (C1) | Test window | No |
| Relative weight change | \|\|x - x_1\|\| / \|\|x_1\|\| over all weight matrices at the final epoch, where x_1 is the initialization, per seed; indicates lazy vs. rich training and networks that barely moved (C1) | Parameters | No |

- Meaning of V1 under schedule-free AdamW: the evaluation weights closely track the frontier of loss against training time within one run [paper: Defazio, Fig. 1, Sec. 2.3], so a flat curve means that stopping later, even with an ideal decay, would not help.
- T1 is the variance term of a bias-variance decomposition across seeds, the counterpart of Neal et al.'s variance due to initialization (here also data order) [paper: Neal].

**Left open**
- None currently.

**Flags**
- V1a reads the test asset only, not the pooled training objective; a flat training curve on the test asset does not guarantee that the pooled objective is flat.
- The evaluation weights average all steps since warmup, so a steady drift of the base iterate appears in them at reduced speed (half for a linear drift) and step-to-step noise is averaged away [calc, idealized]. The flatness threshold (S10) applies on this scale.
- The same decomposition gives a bias term (error of the seed-mean prediction), which is what the glossary's bias regime refers to; it is not among the quantities above, and status uses the T1 log ratio instead (G12).
- Variance due to initialization is large when parameters are fewer than training points and falls with width once they are more [paper: Neal]; the design crosses that boundary (about 170 parameters against 108,000 windows at one end, about 30,000 against 750 at the other [calc]), so T1 moves with width and data size for reasons other than overfitting (C6, C9).
- The gap is in R^2 units; the T1 log ratio is dimensionless. They share no scale.
- See also C1, C4, C6, C7, C8.

**Gaps**: G11, G12, G13.

## S10. Recipe status

**Ascertained**
- Status is based on V1 and the T1 log ratio: valid, failure-unsaturated, failure-overfit.
- Mapping: if the V1a or V1b slope magnitude is above the flatness threshold, the recipe is failure-unsaturated. Otherwise, if the T1 log ratio is above delta_sigma (S8), it is failure-overfit. Otherwise it is valid. Unsaturated takes precedence.
- Flatness threshold: delta_mu (S8) times the median, over all recipes, of the seed std of test R^2. One value, shared by V1a and V1b (both in R^2 units). The rule is fixed in advance; the number is computed from the population after the runs, with no free choice. Meaning: the change between the last two 25-epoch blocks is below the test-R^2 difference the maps resolve for a typical recipe [inference]. It says the end of training is flat at map resolution, not that further training could not help.
- The T1 threshold is delta_sigma, so it adds no new number; at n = 32 it corresponds to T1_test / T1_train ~ 2.9 [calc]. Status compares point estimates with the thresholds, without a test (S8).
- Failures are recorded with rates per option and per pair of options (S11).
- All recipes are reported and enter every map (S12). Status is a categorical characteristic, regionalized like the others; findings about the bias regime are read inside valid-status regions.

**Left open**
- None currently.

**Flags**
- delta_sigma is the regionalization's resolution, not a theory-set boundary between regimes (G12).
- The training curve's seed std may differ from the test curve's, so one shared threshold is stricter on one side than the other [inference].
- See also C1, C6, C8.

**Gaps**: G11, G12.

## S11. Characteristics

**Ascertained**
- Per recipe: the quantities in S9. Per region: S12.
- Over all recipes: failure rates (unsaturated, overfit) per option and per pair of options (failures are expected to come from combinations, C9), and data saturation at population level. No failure-rate cutoff is applied; the space is not changed based on results.
- Data saturation is measured along history within each tier, and along the nested financial tiers (1 -> 2 -> 3 -> 4) within each history. Tier 5 is not on the size axis, because composition changes there (C5). It uses all recipes, with status against data size reported alongside (C8).
- Variance enters the analysis at two levels: seed std within a recipe, and the spread between recipes within an option. The functional ANOVA decomposition [paper: Hutter] is computed directly from cell means over the full grid, for both the seed mean and the seed std; no surrogate model is needed [inference].
- Tuned hyperparameter values are reported as a secondary output, including the tuned learning rate against batch size (a diagnostic for C6; Li et al.'s Surge result covers Adam-style optimizers, and its transfer to schedule-free AdamW is [inference]). Tuned values, edge hits, and the infeasible fraction are shown in the map layout, not regionalized (S12).
- Which DWT bands a network relies on: deferred (confounded by the volatility features, C2).
- Feature learning, feature similarity, and test-curve shape: out of scope; relative weight change is logged (S9).

**Left open**
- None currently.

**Flags**: C2, C6, C8.

**Gaps**: G14.

## S12. Regionalization and visual outputs

**Ascertained**
- Each mapped quantity gets its own regionalization, over all recipes. A region must be connected; recipes that are far apart but have similar results stay in separate regions. Borders are not predefined. The procedure is a contiguity-constrained agglomerative clustering, with contiguity defined by the user, here in factor space [paper: Guo].
- Maps:

| Map | Quantity mapped | Rule |
|---|---|---|
| Test R^2 | per-seed values | mean |
| Generalization gap | per-seed values | mean |
| T1, test vs. train | T1 log ratio | log ratio |
| Status | label | connected areas with the same label; no test |

- Not mapped: training R^2 (equal to test R^2 plus the gap, per seed); T1 per side, prediction spread, and relative weight change (per-recipe diagnostics, S9); seed std (reported per region as sigma_R, not tested); failure rate (becomes the status composition of a region); data saturation (a change along history and tier, S11).
- Not regionalized, shown in the map layout: tuned hyperparameters, edge hits (S7), and the infeasible fraction (one value per recipe, no seeds, so no SE).
- Adjacency: nodes are all recipes of the grid. Depth, width and history: neighbouring levels. Tier: 1-2, 2-3, 3-4, 4-5 (C5). Regularization: the two options. About 6.9 neighbours per recipe on average [calc]. A merged region's neighbours are the union of its members' neighbours (Guo's dynamic contiguity update) [paper: Guo].
- Recipe summary per quantity: mean xbar and sd s over n seeds.
- Region summary: mu_R = mean of the members' means; sigma_R^2 = mean of the members' seed variances; heterogeneity (the spread of the members' means) is reported separately. SE(mu_R) = sqrt(sum(s_j^2)/n) / n_R.
- Error rates and margins: S8.
- Mean rule (equivalence). H0: |mu_A - mu_B| >= delta_mu*sigma_p; H1: within. D = (|mu_A - mu_B| + t_(1-alpha)*SE_d) / (delta_mu*sigma_p), with SE_d = sqrt(SE_A^2 + SE_B^2), sigma_p = sqrt((sigma_A^2 + sigma_B^2)/2), Welch degrees of freedom; the pair passes if D < 1, which is exactly the two one-sided tests rejecting H0 at alpha [paper: Lakens: two one-sided tests, Welch by default, equivalent to the 90% CI lying inside the bounds].
- Log-ratio rule (the T1 map): a recipe's value is its T1 log ratio, with its SE from a jackknife over seeds; a region's value is the mean of its members' values, with SE sqrt(sum(SE_j^2)) / n_R. D = (|difference| + z_(1-alpha)*SE) / delta_sigma; the pair passes if D < 1.
- Procedure: every recipe starts as its own region; compute D for every adjacent pair; merge the pair with the smallest D (exact ties by recipe ID); recompute the merged region's summary and the D values of its adjacent pairs; stop when the smallest D is >= 1. Deterministic, with no starting point.
- Output per region: members, size, mu_R, sigma_R, heterogeneity, and which factors the region spans.
- **Visual outputs.** The set is fixed in advance and produced in full for every mapped quantity, so no view is chosen after seeing results. Recipe value means the seed mean for test R^2 and the gap, the T1 log ratio for the T1 map, and the label for status. Test R^2, the gap, and the T1 log ratio use a diverging color scale centred at 0 (persistence, no gap, equal seed variance on both sides); status uses one fixed color per label. Each quantity keeps one color scale across all its figures.
  - **Lattice** (where values and borders sit). Per mapped quantity, one lattice per regularization option. Outer grid: tier as rows (1 to 5, top to bottom), history as columns (3, 7, 18 years). Each outer cell is a 4 x 4 heatmap: depth as rows (1, 2, 4, 8), width as columns (8, 16, 32, 64). Each recipe square shows its value as color and its region ID as a label. Region borders are drawn as outlines between depth and width neighbours inside each outer cell; borders along tier, history, and regularization are read from the region IDs. Edge hits are marked with a symbol on every lattice.
  - **Region graph** (how regions are structured). Per mapped quantity, one graph. A node is a region, labeled with its region ID, sized by its number of recipes, and colored by mu_R (status: by label). An edge joins two regions that contain at least one adjacent pair of recipes, with width proportional to the number of such pairs. Layout is deterministic.
  - **EDF panels** (how each option's recipes are distributed). For test R^2, the gap, and the T1 log ratio, one panel per factor, with one curve per option: the fraction of that option's recipes whose recipe value is below a given level [paper: RegNet, eq. 1]. Status has no EDF; its per-option composition is the failure-rate table of S11.
  - **Tuning lattices** (not regionalized). Same layout as the map lattices, without region outlines: tuned learning rate (both regularization options), tuned lambda (weight decay only), and the infeasible fraction (both options). Boundaries that coincide with jumps in tuned learning rate are read by comparing a map lattice with the tuned-learning-rate lattice in the same layout.
  - Count: 8 map lattices, 4 region graphs, 15 EDF panels, 5 tuning lattices [calc].

**Left open**
- None currently.

**Flags**
- alpha holds per merge, not per map; later merges reuse data, so the nominal alpha is approximate [inference].
- The margins assume near-normal per-seed values; no normality check and no tail adjustment are planned [inference].
- The plug-in sigma_p in the mean margin approximates the exact noncentral-t test [from memory].
- The tolerated absolute mean difference scales with sigma, so noisy pairs merge at larger absolute differences (distributional view, chosen consciously). Lakens notes the same property of standardized bounds [paper: Lakens].
- Merges compare region summaries (centroid-like). Guo's evidence favours complete linkage (every cross pair similar) for within-region homogeneity, and single linkage chains [paper: Guo]; a drifting summary can admit a recipe that is equivalent to the summary but not to the member it borders. Heterogeneity is reported for this reason. Centroid merges can also produce inversions in the merge tree [from memory].
- Guo partitions a merge tree into a chosen number of regions; here merging stops by test, with no counterpart in Guo (G15).
- Tuning variance is not in the seed SEs, so differences between neighbours partly reflect tuning luck [paper: Bouthillier]; the tuning lattices and edge-hit marks check for this.
- EDF panels pool over all other factors, so effects that exist only in some combinations are averaged out; the lattices show those combinations [inference].
- A region graph with many small regions becomes hard to read; the lattice remains readable [inference].
- See also C1, C4, C6, C7, C8.

**Gaps**: G15.

## S13. Running it

**Ascertained**
- Every recipe of the grid is run once; there is no sampling and no iteration. Changing the recipe space (adding or removing options, tightening bounds) based on results is out of scope.
- Run order:
  1. Calibration run: 24 recipes at the extremes of what moves the best hyperparameters. Batch size: tier 1 x 3 years (B ~ 30), tier 3 x 7 years (middle, because the best Adam learning rate peaks at an intermediate batch size [paper: Li et al.]), tier 4 x 18 years (B ~ 4,300). Capacity: the four depth x width corners (1, 8), (1, 64), (8, 8), (8, 64). Regularization: both options. Tier 5 is left out because it has tier 4's batch size [inference]. Wide ranges, 16 trials x 4 seeds per recipe, tuning stage only. Its best values set the main-run ranges (S7); its results are otherwise discarded.
  2. Main run: all 480 recipes, 16 trials x 4 seeds each in the tuning stage, then 32 seeds each in the final stage.
  3. Analysis: the quantities of S9, status (S10), characteristics and the functional ANOVA decomposition (S11), and regionalization and visual outputs (S12).
- Cost [measured, at 16 trials x 4 seeds in tuning and 32 seeds in the final stage]: calibration about 17 minutes and the main run about 4.6 hours of wall time on two GPUs, with trials and seeds batched per recipe pair.

**Left open**
- A batch-size control (rerunning a subset at more steps per epoch to separate batch-size from data-size effects, C6): deferred by the owner; may be investigated later.

**Flags**
- Record the ranges and computed thresholds used, so the later project can reproduce the protocol (S5).

## S14. Confound register

Each entry is one root cause with all its consequences. Sections reference entries by ID.

**C1. Persistence baseline** (S1, S9, S10, S12). Next-day returns carry little predictable signal, so R^2 near or below zero is expected: a linear model on the same window information scores below zero on the test window [measured]. A network that ignores its input outputs a constant in window units and forecasts the window mean shifted by a fixed multiple of sigma; prediction spread (S9) is zero for it. Near-constant outputs also have small variance across seeds, so a trivial network can look like a low-variance one and pass V1 trivially [speculation]. Prediction spread and relative weight change (S9) are reported to separate valid recipes from such networks; check how they separate in the failure rates.

**C2. Input representation** (S2, S4, S5, S11).
- Level: every input except mu is invariant to the price level. mu leaves its training range when prices trend [measured: every JPM test window's mu lies above its training range at all three history lengths], and ReLU MLPs extrapolate linearly outside the training range [paper: Xu]. The forecast takes its level from the de-normalization, so this risk enters only through the mu input.
- Prices are back-adjusted for dividends and splits, so a window's mu partly reflects corporate actions after its date [inference]. Differences within a window, and therefore every other input, are unaffected.
- Scale: the normalized details and volatility features are of unit scale; mu (about 1 to 7) and log sigma (about -6 to 0) are not centred [measured]. Input scale acts like an extra init scale, shifting the lazy/rich position (S4) [inference from Chizat, Kunin; not tested in them; Kunin: relative layer scale also moves the regime, and in nonlinear networks faster-learning early layers speed up rich learning].
- Gradients: the loss is on the forecast, so gradients scale with sigma times the forecast error, both small; with eps fixed (S5), eps may dominate the Adam denominator [paper: Choi, for Adam; transfer to schedule-free AdamW is inference; the size of the effect is speculation].
- The volatility features are exact functions of the details under the orthonormal Haar convention: the population std over m days equals sqrt(sum of squared details in the matching dyadic block / m), with 15 details for 16 days, 7 for 8, 3 for 4 [calc]. They add no information, only the square-root-of-sum-of-squares; the std of log prices mixes drift and volatility. Attribution to DWT bands is therefore confounded by the volatility features (S11). The orthonormal convention sets these identities; revisiting it changes them.

**C4. History length** (S3, S7, S9, S12). History length is confounded with:
- Regime content: if the data end near the present, the origin is near late 2024; an 18-year history then includes 2008, a 3-year one starts around 2021 [inference].
- Tuning noise: the validation window scales with history; at tier 1 with 3 years it is about 150 heavily autocorrelated windows.
- Training-side noise: test-asset training windows are few at short history (about 750 at 3 years).
- Drift: the test window can degrade for regime reasons, not only overfitting. The T1 log ratio mixes seen vs. unseen windows with the period difference (3, 7, or 18 years of training vs. a 2-year test window), so drift can mimic overfitting.
- Gap regions that span history options mix different training-side scoring periods (S12).

**C5. Data pool and tiers** (S3).
- Tier 4 vs. 5 isolates composition at equal size.
- Survivorship: the pool is current index members; index tenure is a lower bound on listing history; sector labels are today's GICS; dual share classes are not deduplicated (G17).

**C6. Data-size couplings** (S3, S5, S6, S7, S9, S10, S11, S12). N is set by tier and history, and B ~ N/25, so gradient noise (lr/B) changes systematically with tier and history; batch-size and data-size effects cannot be separated.
- What the coupling buys: equal steps and epochs for every recipe, so every epoch-based setting (warmup, V1 windows) means the same number of steps everywhere. Tuning-to-final transfer keeps the step count and enlarges the batch about 1.25 times (S7, G8). Per-recipe tuning absorbs part of the batch effect: the same final performance is usually attainable at any batch size when lr and regularization are tuned and training is long enough [paper: Playbook; Shallue]. Optimal metaparameters follow no simple rule in batch size, which favors per-recipe tuning over scaling heuristics [paper: Shallue]. When training saturates, the effect of batch size on solution quality was very small or nonexistent in Shallue et al.'s small workloads [paper: Shallue]. Shallue et al. studied SGD with and without momentum, not Adam-type optimizers, so these points are [inference] here.
- What it can hide: that guidance concerns the best achievable performance, not characteristics. Small batches add noise with a regularizing effect, and large batches can overfit more [paper: Playbook]; recipes without weight decay have no tuned regularizer to compensate. The base iterate runs at a constant learning rate after warmup, so per-step noise scales with lr/B; the evaluation weights average much of it out, so V1 and T1 depend on B less than a constant-lr last iterate would, but not necessarily not at all [inference]. Data-saturation curves (S11) mix more data with less gradient noise.
- Not affected: comparisons within one tier and history (depth, width, regularization share the same B), and tier 4 vs. 5 (same N).
- Diagnostic: tuned lr against B (S11). A batch-size control is deferred (S13).

**C7. Single draw, small effective sample** (S1, S3, S8, S9, S12). Conclusions are conditional on one fixed asset draw, one test asset, and one test period (about 2 years, about 500 windows). T1 is seed variance (initialization, data order) on one fixed dataset; data-sampling variance is not measured [paper: Bouthillier: data sampling is the largest source; initialization is generally under half of it, on par with data order]. Overlapping windows are heavily autocorrelated; the effective sample size is far below the window count (G17).

**C8. Test-window reuse and status** (S9, S10, S11, S12).
- The test window is read by V1b, test R^2, and the test side of T1, and also decides status; hyperparameters are fixed before it is read.
- The flatness threshold's value comes from the population after the runs, but by a rule fixed in advance, so it leaves no analyst freedom; it still depends on the same results it classifies.
- Quantities that feed status (slopes, T1 log ratio) act as filters, so findings about them are conditional on validity.

**C9. Capacity vs. data** (S4, S6).
- With 19 inputs, widths 8 and 16 are bottlenecks and 32 and 64 are expansions.
- Depth 8 at width 64 has about 30,000 parameters against about 750 correlated windows at tier 1 with 3 years [calc]; many overfit failures are expected [speculation; to be seen in failure rates].
- Depths 4 and 8 straddle the hypothesized overfit boundary at 6; it shows as a jump in failure rate, not at a specific depth.

## S15. Evidence gaps and literature

**Ascertained**
- Methods are grounded in the papers listed in `literature/index.md` and the Tuning Playbook; the owner may add more and wants to be warned about omissions. The literature is kept small: a new source is added only when a decision needs it, and one source should back a whole mechanism rather than one detail. Where the Playbook and Defazio et al. conflict on optimizer configuration (schedule, warmup, which Adam settings to tune), Defazio et al. applies.
- Anything that is not part of the experiment is handed to established tools.
- The brief fixes definitions, not implementation; tools for anything not named here are chosen at implementation.

**In the project (used):** Defazio et al. 2024 (The Road Less Scheduled: schedule-free AdamW; optimizer, warmup, tuning set, meaning of V1; S5, S7, S9); Tuning Playbook (quasi-random search; range-edge check; batch-size points in C6); Choi et al. (range-edge check; eps, C2); Chizat (lazy training); Kunin et al. 2024 (Get Rich Quick: overall and relative init scale set lazy vs. rich; C2); Bouthillier (variance, pairing, run count); Summers and Dinneen 2021 (nondeterminism and instability; S7, S8, S9); fANOVA (functional ANOVA decomposition; S11); Shallue et al. 2018 (data parallelism and batch size; SGD family only); Neal et al. 2018 (variance due to initialization vs. sampling, by width; T1, S9); Lakens 2017 (equivalence tests; S8 margins, S12 mean rule); Guo 2008 (REDCAP: contiguity-constrained agglomerative clustering; S12); RegNet (distributions over a design space; S12 EDF panels); Li et al. 2024 (Surge Phenomenon: optimal lr vs. batch size for Adam-style optimizers; S7 shared ranges, S13 calibration set, S11 diagnostic; inference for schedule-free AdamW); Liu et al. 2022 (Non-stationary Transformers: window normalization, de-normalization, re-injected statistics; S2, C2); Xu et al. 2021 (How Neural Networks Extrapolate: ReLU MLPs become linear outside the training range; C2).

**In the project, not yet cited in a section:** Gu, Kelly and Xiu 2020 (US equity return prediction; R^2 against a zero forecast; shallow nets beat deep; C1, C9).

**Guides (kept, not cited):** Rhif et al. 2019 (wavelet transform for non-stationary time series: review); Tawakuli et al. 2024 (time-series data preprocessing: survey).

**Tools:** Optuna (trial storage); scipy (Halton points); the schedule-free reference implementation (facebookresearch/schedule_free, pinned), against which the batched port is verified.

**Cite only, no upload:** Akiba et al. 2019 (Optuna); Schuirmann 1987 (two one-sided tests) and Berger and Hsu 1996 (intersection-union tests), both as cited in Lakens; Percival and Walden (wavelet methods), if a DWT reference is needed.

**Candidate additions** [from memory, not in the project]: LeCun et al. 1998 (Efficient BackProp: input scaling), only if C2 needs a citation rather than an estimate; Jiang et al. 2022 (Assessing Generalization of SGD via Disagreement; seed disagreement tracks test error, classification only; G12); D'Angelo, Andriushchenko et al. 2024 (Why Do We Need Weight Decay in Modern Deep Learning?; weight decay acts through the effective learning rate in under-training; G5); Hestness et al. 2017 (data-scaling learning curves; G14); Kelly, Malamud and Zhou 2024 (The Virtue of Complexity in Return Prediction; C1, C9).

**Considered, not added:** Franceschi et al. 2024 (overview; primaries cited instead); McCandlish et al. 2018 (SGD rule; superseded by Li et al. 2024 for Adam); Yang et al. (muP; no transfer is used); Cohen et al. (edge of stability; out of scope); Belkin / Nakkiran (double descent; owner: not related); NAS Evaluation Is Frustratingly Hard and PDPs for HPO (covered by RegNet and Bouthillier); Lin and Dobriban 2021 (covered by Bouthillier and Neal); Kornblith et al. 2019 (CKA; feature similarity out of scope, S11); Woodworth (superseded by Kunin); CompleteP and DeepCAVE (no use with schedule-free AdamW and full-length trials); Liu et al. 2023 (Dropout Reduces Underfitting; dropout out of scope); Kim et al. 2022 (RevIN; Liu's parameter-free normalization performs the same); Ogasawara et al. 2010 (adaptive normalization; covered by Liu); Passalis et al. (DAIN; learned normalization adds parameters outside the capacity grid and needs its own learning rates per sublayer); Fan et al. 2023 (Dish-TS; at a one-step horizon its output-side coefficient net is itself a linear forecaster); Lopez de Prado (fractional differencing needs a lookback far beyond 16 days, and the level is handled by window normalization; no embargo is needed, since with a one-step target no label crosses a cutoff); Gencay, Selcuk and Whitcher; Assunção et al. 2006 (SKATER; equal to Guo's First-Order-SLK); Morales-Brotons et al. 2024 (EMA of weights; averaging is inside the optimizer instead); Hägele et al. 2024 (constant lr with cooldown; the V1 window would fall inside the cooldown); Wellek (equivalence testing; no use).

**Evidence-gap register.** Each entry names what lacks backing, the nearest support, and what would close the gap. Sections reference entries by ID.

- **G1. Input representation** (S2): Haar DWT of the window-normalized log prices, re-injected window statistics appended to the input, volatility features, 16-day lookback. Nearest: Liu (window normalization and re-injection, for Transformers on non-financial and exchange-rate series); Rhif (wavelets, review only). Closing: studies of normalized wavelet inputs for neural return forecasting.
- **G2. Data design** (S3): tier construction, single test asset, test-asset-only evaluation, history lengths, 80/20 split, validation merged into final training, no embargo. Nearest: Gu, Kelly and Xiu (pooling across stocks). Closing: evaluation-protocol literature for financial ML.
- **G4. Architecture ranges** (S6, C9): depth and width ranges; the hypothesized overfit boundary at depth 6. Nearest: Neal, Summers (variance only); Gu, Kelly and Xiu (shallow beats deep).
- **G5. Regularization in the bias regime** (S6): weight decay's role under schedule-free AdamW. Nearest: Playbook (general). Closing: D'Angelo et al. 2024 (candidate).
- **G6. Training length and steps per epoch** (S5): 500 epochs, K = 25. Nearest: consequences only (Shallue, Playbook).
- **G7. Initialization** (S4): PyTorch default [from memory]. Nearest: Chizat, Kunin (init scale moves lazy vs. rich).
- **G8. Tuning-to-final transfer** (S7, C6): reuse of tuned values at the same steps and a 1.25 times larger batch, under schedule-free AdamW. Nearest: Defazio (training length only).
- **G9. Tuning objective, ranges, trial count, and seeds per trial** (S7): last-25-epoch mean as objective; shared ranges from the best values at a few extreme recipes plus a 0.75-decade margin; 16 trials x 4 seeds for all recipes, objective averaged over seeds. Nearest: Choi (ranges, edge check); Defazio (learning-rate shift); Playbook (search-space refinement); Li et al. (batch-size dependence).
- **G10. Fixed eps with small gradients** (S5, C2). Nearest: Choi (Adam; eps matters); Defazio (eps not tuned).
- **G11. Saturation criterion** (S9, S10): V1 slope construction on the test asset; one shared flatness threshold derived from the regionalization margin delta_mu and the median seed std of test R^2. Nearest: Defazio (the evaluation weights track the loss-time frontier, which gives flatness its meaning); Lakens (the margin); no paper on the criterion itself.
- **G12. Bias regime and overfitting criterion** (S9, S10): status by the T1 log ratio with delta_sigma as threshold; no direct measure of the bias share. Nearest: Neal (decomposition), Summers (diversity). Closing: Jiang et al. 2022 (candidate; classification only).
- **G13. Prediction spread** (S9). Nearest: none (relative weight change: Chizat, partial).
- **G14. Data-saturation measurement** (S11). Nearest: none. Closing: Hestness et al. 2017 (candidate).
- **G15. Regionalization details** (S12): jackknife SE and the log-ratio rule for T1; centroid-like merging; stopping by test; error control over sequential merges. Nearest: Lakens (means only), Guo (linkage, fixed number of regions), Berger and Hsu (cite only).
- **G17. Pool and sample** (C5, C7): survivorship; effective sample size of overlapping windows. Nearest: none.
