# Project Guide: Characteristics of Training-Saturated MLPs in the Bias Regime

A reader's companion to `PROJECT_BRIEF.md`.

- **The brief is the source of truth.** This guide explains it; it adds no decisions. If the two disagree, the brief wins, and the guide is out of date.
- **Synced to:** the brief as of 2026-11-02 (consolidation: full grid run once, one-step target, one tuning protocol, 32 trials for every recipe).
- **Register IDs are unchanged.** Confounds keep C1–C9 and evidence gaps keep G1–G17, so anything that cites the brief also resolves here (Appendix E).

---

## How to read this guide

Chapters build from the bottom up. Each one introduces only what it needs and relies only on earlier chapters. Every chapter has the same blocks, in the same order:

| Block | What it gives you |
|---|---|
| **In one line** | What the chapter settles |
| **Builds on** | The earlier chapters you need; reread their *Carry forward* boxes to reload context |
| **Concept** | The idea in general terms, before any project choice. Findings from papers are explained, not assumed |
| **In this study** | What the owner has decided (the brief's *Ascertained*) |
| **Still open** | Undecided items, each with the working reading in use. None of them is a decision |
| **Worked example** | One fixed recipe followed through every chapter, with pseudocode for what is computed. The pseudocode shows *what*, not *how*; tools are chosen at implementation |
| **Watch out** | The chapter's own warnings, plus pointers to the cross-cutting confounds (C) and evidence gaps (G) |
| **Further reading** | Where to go deeper, with the section or figure |
| **Carry forward** | The few points later chapters assume you remember |

**Evidence markers.** Every claim that does not come directly from the owner carries one of these:

| Marker | Meaning | Brief tag |
|---|---|---|
| 📄 | Established guidance from a project paper (named) | [paper: X] |
| 🧮 | Own arithmetic | [calc] |
| 📏 | Estimate, not computed on project data | [estimate] |
| 💭 | Reasoned, not tested | [inference] |
| ❔ | Speculation | [speculation] |
| 🧠 | From memory, to be verified | [from memory] |

Status markers: ✅ decided · 🔓 left open · 🅿️ parked (open, but evaluated only when the owner asks; its working reading is used as is) · ⚠️ warning.

**The worked example recipe**, used in every chapter:

> **Recipe E**: depth 2, width 32, weight decay on, data tier 3 (12 financial assets), history 7 years.

---

## Chapter 0. Orientation

**In one line.** The study measures how MLP design and data choices *behave* once training has run to its end. It does not ask which recipe wins.

**Builds on.** Nothing.

**Concept.** Most model studies ask which configuration performs best. This one asks a different question: for every combination of choices in a fixed design space, what does the trained network look like? How well does it fit, how much does it change from seed to seed, does it overfit, and did it actually finish learning? Where in the design space do these properties stay the same, and where do they change? The answer is a set of *maps* over the design space, not a leaderboard.

Two terms frame the question:
- **Training saturation.** Training has run long enough that further training no longer changes the result in a measurable way.
- **Bias regime.** The remaining error comes mostly from the network being too simple or too constrained (underfitting), not from memorizing the training data (overfitting).

The study wants to describe networks that are in both states, and to describe how often, and where, networks fail to reach them.

**In this study** ✅
- Data universe and task: US equities, with a one-step return forecast from price windows processed by a discrete wavelet transform (DWT) (Chapters 1–3).
- Unit of study: the full grid of 480 **recipes**. Each recipe is tuned on its own and then trained with 32 seeds.
- Every recipe is analyzed for training and test R², the generalization gap, and how much predictions vary across seeds. Each of these comes with its spread across seeds.
- **Recipe status** (valid, failure-unsaturated, failure-overfit) is one categorical characteristic among the others.
- The main quantities are **regionalized**: recipes are grouped into connected regions with equivalent results, and no borders are drawn in advance (Chapter 12).
- Findings about the bias regime are read inside regions whose status is valid. Failure rates and data saturation are described over the whole population.
- The recipe space is fixed. Changing it in response to results is out of scope.
- The characterized recipes are meant for a later project. That project must use the same training protocol (Chapter 5) for these characteristics to carry over.

**Still open.** Nothing at this level.

**Worked example.** Recipe E is one of the 480 recipes. By Chapter 12 it will have 32 trained seeds, a status, and a region ID on each of four maps.

**Watch out.** The study characterizes; it does not rank. A statement such as "recipe A has a higher test R² than recipe B" is never a finding by itself here.

**Further reading.** RegNet (Radosavovic et al. 2020), §3.1: the case for studying *distributions* of models over a design space rather than the best model found. 📄

**Carry forward**
- The output is characteristics of choices, shown as maps, not a ranking.
- The two target conditions are training saturation and the bias regime.
- The grid is fixed: 480 recipes, each run once, 32 seeds each.

---

## Chapter 1. The forecasting problem

**In one line.** Predict tomorrow's log return of a stock, and score the prediction against forecasting a return of zero.

**Builds on.** Chapter 0.

**Concept.**
- **Log price and log return.** With closing price P(t), the one-step log return is log P(t+1) − log P(t). Log returns add up over time and do not depend on the price level.
- **Persistence.** The naive "no change" forecast. For prices it predicts P(t+1) = P(t); for returns this is the same as predicting a return of zero.
- **R² against persistence.** R² = 1 − Σ(r − r̂)² / Σ(r − 0)². Here r is the actual return and r̂ the prediction. R² = 0 means no better than predicting zero, and R² < 0 means worse.
- **Why benchmark against zero rather than the historical mean.** Daily stock returns are dominated by noise. Gu, Kelly & Xiu use exactly this zero benchmark for individual stock returns, because the historical mean is so noisy that it lowers the bar. Benchmarking against the mean instead raises every method's out-of-sample R² by roughly three percentage points. 📄 (Gu et al. §1.8, eq. 19 and fn. 34)

R² values near zero, or even below it, are therefore normal in this setting and not a sign of a bug.

**In this study** ✅
- **Target.** The one-step return from the last input date t, i.e. k = 1. Multi-step targets were removed on 2026-11-02.
- **Comparison metric.** R² on the t+1 prediction against persistence, not the training loss.
- **Where R² is computed.** On the **test asset** only (Chapter 3), with the persistence baseline computed on the same windows.
- Time-invariance of MLPs is not a research question.

**Still open**
- 🅿️ **Target scale** (part of the parked item on input scale, output scale and normalization; see Chapter 2). *Working reading:* cumulative log return, no rescaling (for example, no conversion to percent).

**Worked example**
```text
# target for one window ending at date t (working reading)
r[t+1] = log(P[t+1]) - log(P[t])

# R² against persistence on a set of test-asset windows W
R2(W) = 1 - sum_{w in W} (r_w - r_hat_w)^2  /  sum_{w in W} r_w^2
```
For Recipe E, as for every recipe, R² is always computed on the test asset's windows, even though the network trains on 12 assets.

**Watch out**
- ⚠️ A network that outputs almost exactly zero scores R² near zero and looks stable. Chapter 9 adds diagnostics for this case. → **C1**
- Gap: the design choices behind this setup lack direct literature backing. → **G2**

**Further reading**
- Gu, Kelly & Xiu 2020, *Empirical Asset Pricing via Machine Learning*, §1.8 (performance evaluation, the zero benchmark). 📄

**Carry forward**
- Target: the next-day log return. Persistence means predicting zero.
- R² is measured against predicting zero, on the test asset only. Values near zero are normal.

---

## Chapter 2. From prices to inputs

**In one line.** Each sample is a 16-day window of log prices, transformed into 16 wavelet coefficients plus 3 volatility features: 19 inputs.

**Builds on.** Chapter 1.

**Concept.**
- **Window.** One sample: a lookback of past inputs ending at the last input date t, together with its target at t+1.
- **Haar discrete wavelet transform (DWT).** Repeatedly take pairs of neighbouring values (a, b). Keep their scaled sum (a+b)/√2 as the "approximation" and their scaled difference (a−b)/√2 as the "detail". Then repeat on the approximations only. For 16 values this gives details at 4 scales (8 + 4 + 2 + 1 = 15 numbers) and 1 final approximation, 16 numbers in all. The √2 scaling makes the transform *orthonormal*: it preserves sums of squares, so energy is neither gained nor lost. Detail coefficients describe movement at each time scale; the approximation carries the overall level. 📄 (Rhif et al. §2.3, multi-resolution analysis)
- **Computing the DWT inside each window** uses only that window's prices, so it never sees the future. It is causal by construction.
- **Extrapolation.** ReLU-type MLPs become linear outside the range of their training data, along any direction away from the origin. So inputs that drift outside the training range get a linear, usually wrong, response. 📄 (Xu et al. 2021, Theorem 1, Fig. 1)

**In this study** ✅
- Lookback: 16 trading days.
- Input: the closing price series over the lookback, transformed by the orthonormal Haar DWT ((a±b)/√2 at each stage), at all scales: 15 details and 1 approximation.
- Three volatility features are added: the population standard deviation (dividing by n) of the log price over the last 4, 8 and 16 days. They are there to spare small networks a nonlinearity they may not be able to approximate, not to add information.
- Input dimension: 19.
- The DWT is applied to the inputs only, never to the target.
- Fixed regardless of the parked item below: the Haar DWT and a cumulative-return target.

**Still open**
- 🅿️ **Input scale, output scale and normalization.** This covers the price scale (log or other), treatment of the level (the approximation coefficient), rescaling of inputs or targets, and fixed or learned input normalization. *Working reading:* log closing price; the level is kept through the approximation coefficient; cumulative log return as target; no scaling or standardization of inputs or outputs. (Parked: not raised unless the owner asks.)

**Worked example**
```text
# p: 16 log closing prices of one asset, oldest -> newest, ending at date t
a = p
details = []
while length(a) > 1:
    pairs     = group a into consecutive pairs (a_i, b_i)
    details  += [(a_i - b_i) / sqrt(2) for each pair]   # sign convention not fixed by the brief
    a         = [(a_i + b_i) / sqrt(2) for each pair]
# details: 8 + 4 + 2 + 1 = 15 values; a: 1 approximation value

vol4  = pstd(p[-4:])     # population std (divide by n)
vol8  = pstd(p[-8:])
vol16 = pstd(p[-16:])

x = concat(details, a, [vol4, vol8, vol16])   # 19 inputs
```
For a stock trading near $100, log P ≈ 4.6, so the approximation coefficient is about 4 × 4.6 ≈ 18. The other 18 inputs are around 0.01 to 0.1. 📏

**Watch out**
- ⚠️ **Scale gap.** One input (the approximation coefficient) is about 100 to 1,000 times larger than the others and than the target. It is also the only input that can drift outside the training range when prices trend upward; the details, the volatility features and the target do not depend on the price level. → **C2**
- ⚠️ **The volatility features are redundant by construction.** Under the orthonormal convention, the population std over m days equals √(sum of squared details in the matching block / m). That uses 15 details for 16 days, 7 for 8, and 3 for 4. 🧮 So they add no information, only a convenient nonlinearity. This also means you cannot cleanly attribute reliance to particular DWT bands. → **C2**
- The orthonormal convention causes both the scale gap and these identities; changing the convention would change both.
- Gap: this input representation has no direct literature backing. → **G1**

**Further reading**
- Rhif et al. 2019, *Wavelet Transform Application for/in Non-Stationary Time-Series Analysis: A Review*, §2.3 (multi-resolution analysis, approximation and detail). 📄
- Xu et al. 2021, *How Neural Networks Extrapolate*, §3.1, Theorem 1 and Fig. 1. 📄
- Tawakuli et al. 2024 (time-series preprocessing survey): general background, kept as a guide.

**Carry forward**
- 19 inputs: 16 Haar coefficients (1 of them the level) and 3 redundant volatility features.
- One input, the level, is about 18 while everything else is about 0.01 to 0.1. This scale gap matters for initialization (Ch. 4), the optimizer (Ch. 5) and tuning ranges (Ch. 7).

---

## Chapter 3. The data universe

**In one line.** Which assets and dates a network sees, how time is split into tuning, training and test periods, and why no label crosses a split.

**Builds on.** Chapters 1–2.

**Concept.**
- **Pooling.** A single network trains on windows from several assets at once, which gives it more data than one asset's history. Gu et al. pool across stocks in the same way. 📄
- **Splitting by time.** For forecasting, training data must come before validation data, which must come before test data. Splitting at random dates would let the network learn from the future.
- **Label leakage.** Leakage happens when a training sample's target lies inside a held-out period. *Input overlap* is different and harmless: a held-out window's inputs reaching back into training dates is not leakage, because no target crosses the boundary.

**In this study** ✅

*Assets*
- US equities, main sector financials, assets pooled by sector, history up to 20 years.
- Five **data tiers**:

| Tier | Assets | Composition |
|---|---|---|
| 1 | 1 | financial (the **test asset**) |
| 2 | 4 | financials |
| 3 | 12 | financials |
| 4 | 24 | financials (similar) |
| 5 | 24 | diverse: 4 each from financials, industrials, health care, information technology, consumer discretionary, utilities |

- The financial tiers are nested: tier 1 ⊂ 2 ⊂ 3 ⊂ 4. Tier 5's four financials are tier 2's four.
- Tiers 4 and 5 have the same size, so comparing them isolates *composition*.
- The asset draw, including the test asset, is fixed for the whole study.
- "Consumer" means the consumer discretionary sector in the Global Industry Classification Standard (GICS). Days are trading days.
- Removed (2026-11-02): a 44-asset tier. It would only matter if every smaller tier left models unsaturated, and that would call for redesigning the study around larger datasets anyway.
- Reference counts (snapshot October 2026, public S&P 500 constituent data): 36 financials with index tenure of about 20 years or more; 6 to 34 in other sectors.

*Time*

```
history start ── tuning-stage training ──┤tuning cutoff├── validation window ──┤origin├── test window
              └──────────────── final-stage training data ───────────────────┘
```

- **History length:** 3, 7 or 18 years, counted back from the origin. History includes the validation window, so it equals the whole final-stage training data.
- **Tuning stage:** trains on the first 80% of the final-stage training data and tunes hyperparameters on the last 20% (the **validation window**).
- **Final stage:** retrains with the tuned values on all of it (validation window merged in), so training ends at the origin. The **test window**, the last 2 years of data, follows immediately. Why merge: otherwise the test window would sit too far from the end of training.
- Splits are by calendar date, across all assets at once.

*Boundaries*
- **Cutoff** means the tuning cutoff or the origin: the last *target* date of a stage. A window belongs to a stage if its target date is on or before that stage's cutoff, so its last input date is at least one trading day earlier. This holds for every asset in every tier. No label crosses a cutoff.
- There is no embargo at either cutoff and no in-period purged holdout. Input overlap at a cutoff is accepted, since it is not leakage.

*Evaluation*
- Every train-vs-test comparison (R², the generalization gap, and T1 from Chapter 9) uses **the test asset only, on both sides**. The network still trains on all assets of its tier. This keeps training-set diversity from leaking into the comparisons.
- **Test-asset training windows** are the test asset's windows inside the final-stage training data.

*Sizes* 🧮
- About 250 windows per asset per year.
- Data per recipe ranges from 1 asset × 3 years to 24 assets × 18 years, a factor of 144: roughly 750 to 108,000 windows.
- Test-asset training windows: about 750 (3 years) to about 4,500 (18 years).
- Validation window: about 0.6, 1.4 or 3.6 years (about 150 windows at tier 1 with 3 years).
- Test window: about 500 windows.

**Still open.** Nothing in this chapter.

**Worked example**
```text
# Recipe E: tier 3 (12 assets), history 7 years
N_final        ≈ 12 assets × 7 years × 250  ≈ 21,000 windows
N_tuning_train ≈ 0.8 × N_final              ≈ 16,800 windows
validation     ≈ 1.4 years                  (≈ 350 test-asset windows)
test-asset training windows ≈ 7 × 250       ≈ 1,750
test window                                 ≈ 500 test-asset windows

# stage membership of a window, applied to every asset
stage(window) = "tuning-train" if target_date <= tuning_cutoff
                "validation"   if tuning_cutoff < target_date <= origin
                "test"         if target_date > origin
```
🧮 for all figures.

**Watch out**
- ⚠️ **History length is not a clean "more data" knob.** Longer histories also contain different market regimes (18 years reaches back to 2008), less noisy tuning, more training-side windows, and a larger time distance to the test window. → **C4**
- ⚠️ **Survivorship.** The pool consists of current index members; sector labels are today's; dual share classes are not deduplicated. → **C5**, **G17**
- ⚠️ **One draw, one test asset, one test period.** All conclusions are conditional on them, and overlapping windows are strongly autocorrelated, so the effective sample is far below the window count. → **C7**, **G17**
- Gap: tier design, a single test asset, the split ratios and no embargo lack direct backing. → **G2**

**Further reading**
- Gu, Kelly & Xiu 2020 (pooling across stocks; their sample-splitting scheme). 📄

**Carry forward**
- Tiers 1 to 5 run from 1 to 24 assets; 1 to 4 are nested financials, 5 is diverse at tier 4's size.
- History is 3, 7 or 18 years. Tuning uses an 80/20 split; the final stage merges the validation window and then tests on the next 2 years.
- Splits are by target date, so no label crosses a cutoff.
- All comparisons are made on the test asset only.

---

## Chapter 4. The network

**In one line.** A plain MLP of 1 to 8 hidden layers and 8 to 64 units per layer, with leaky ReLU and the default PyTorch initialization, which decides whether training starts "lazy" or "rich".

**Builds on.** Chapters 2–3.

**Concept.**
- **MLP.** A stack of fully connected layers, each followed by a nonlinearity. *Depth* is the number of hidden layers; *width* is the number of units per hidden layer. The output is a single number, the predicted return.
- **Leaky ReLU.** max(x, 0.01·x). It behaves like ReLU but keeps a small slope for negative inputs, so units never go completely silent.
- **Initialization scale and the lazy vs. rich regimes.**
  - *Lazy* training: the network barely moves from its initial weights and behaves almost like a fixed linear model on random features.
  - *Rich* (feature-learning) training: the internal representation changes substantially during training.
  - Chizat et al. show that lazy training occurs when the model's scale is large relative to how far it needs to move. Scaling up the output, or the initial weights of homogeneous networks, pushes training toward lazy. 📄 (Chizat et al. §1.2, eq. 1, Fig. 1)
  - Kunin et al. add that the overall scale is not all that matters. The *relative* scale between layers matters too, and in nonlinear networks an initialization where early layers learn faster ("upstream") can speed up rich learning. 📄 (Kunin et al. 2024, Fig. 1, §5)
- **Width and seed-to-seed variance.** Neal et al. split prediction variance into a part due to initialization and a part due to sampling the training data. The part due to initialization is large when the network has fewer parameters than training points. Once it has more, it falls as width grows. 📄 (Neal et al. 2018, §2.3 eq. 4, §3.4 Fig. 2)

**In this study** ✅
- Depth (hidden layers): 1, 2, 4, 8. Width (constant across hidden layers): 8, 16, 32, 64. Both are recipe factors (Chapter 6).
- Activation: leaky ReLU with the PyTorch default slope of 0.01. Fixed, not tuned.
- Initialization: the PyTorch default for linear layers, fixed and observed ("option A"). Weights and biases are uniform in ±1/√fan_in, so the weight variance is 1/(3·fan_in). 🧠 The alternatives were not adopted: B (fixed small initialization), C (width-aware scaling) and D (initialization scale as a factor).
- The evaluation weights (Chapter 5) start at the initialization.
- No normalization layers, no skip connections, no dropout.

**Still open.** Nothing in this chapter.

**Worked example**
```text
# Recipe E: depth 2, width 32, 19 inputs, 1 output
params = (19·32 + 32) + (32·32 + 32) + (32·1 + 1) = 640 + 1056 + 33 = 1,729
# ≈ 1 parameter per 12 windows (N_final ≈ 21,000): fewer parameters than points
init: W, b ~ Uniform(-1/sqrt(fan_in), +1/sqrt(fan_in))  for every linear layer
```
At the extremes of the grid 🧮:
- depth 1, width 8: about 170 parameters, against up to 108,000 windows;
- depth 8, width 64: about 30,000 parameters, against as few as 750 windows.

**Watch out**
- ⚠️ The PyTorch default is *not* He initialization. He initialization uses variance 2/fan_in, about 6 times larger. An earlier project note calling the default "He-style" was wrong. 🧠
- ⚠️ With the approximation coefficient around 18, the default initialization probably produces initial outputs of order 1, while targets are about 0.01. 📏 Early training then mostly shrinks the random initial output. 💭 (the brief derives this from Chizat's criterion) Not adopted: a small initialization of the output layer only.
- ⚠️ The large input acts like an extra initialization scale and can shift where a recipe sits between lazy and rich. 💭 → **C2**
- ⚠️ Widths 8 and 16 are narrower than the 19 inputs (bottlenecks); widths 32 and 64 are wider (expansions). The grid crosses the boundary between fewer and more parameters than training points, so seed variance moves with width and data size for reasons other than overfitting. → **C9**, **C6**
- Gaps: the architecture ranges (**G4**) and initialization (**G7**) lack direct backing.

**Further reading**
- Chizat, Oyallon & Bach, *On Lazy Training in Differentiable Programming*, §1.2 (the criterion, eq. 1) and Fig. 1. 📄
- Kunin et al. 2024, *Get Rich Quick*, Fig. 1 (overall vs. relative scale) and §5 (nonlinear networks). 📄
- Neal et al. 2018, *A Modern Take on the Bias-Variance Tradeoff*, §2.3 and §3.4. 📄
- Gu, Kelly & Xiu 2020: shallow networks outperform deep ones in return prediction (in the project, not yet cited in a brief section). 📄

**Carry forward**
- Depth {1, 2, 4, 8} × width {8, 16, 32, 64}, leaky ReLU, default PyTorch initialization (not He).
- Initialization scale, and input scale with it, moves training between lazy and rich.
- Seed variance from initialization depends on width and on parameters vs. data, not only on overfitting.

---

## Chapter 5. How it learns

**In one line.** Mean squared error (MSE), minimized by schedule-free AdamW: a fixed 5% warmup, then a constant learning rate, with an internal running average used as the model, for 500 epochs of exactly 25 steps each.

**Builds on.** Chapters 3–4.

**Concept.**
- **Adam.** Adam scales each parameter's step by running estimates of the gradient's mean (rate β₁) and squared magnitude (rate β₂). A small ε in the denominator prevents division by zero. **AdamW** adds *decoupled* weight decay: each step shrinks the weights toward zero by lr·λ, separately from the gradient step.
- **Why schedules exist.** At a constant learning rate, the iterate keeps bouncing around in gradient noise. Decaying the learning rate toward the end removes that noise and usually gives a better final model, but a schedule needs the stopping time in advance.
- **Schedule-free AdamW** (Defazio et al. 2024) removes the need to know the stopping time. It keeps three sequences: 📄 (§2, Alg. 1)
  - **z**, the base iterate, where the Adam step is taken;
  - **y** = (1 − β)·z + β·x, the point where gradients are evaluated;
  - **x**, a weighted running average of z. **x is the model you evaluate.**

  The averaging weights are proportional to the squared learning rate, c₍t+1₎ = γ_t² / Σγ_i², so steps taken at a small learning rate get little weight. 📄 (eq. 23) The average x behaves like a model trained with a linear decay ending *at the current step*. The method therefore applies a schedule implicitly and can be read at any time. 📄 (§2.3, Fig. 4) Defazio et al. also show that x closely tracks the best loss reachable at each point in training. 📄 (Fig. 1)
- **Practical findings from Defazio et al.**
  - Warmup is still required. 📄 (§4.4)
  - β = 0.9 works on all their problems but one, and its best value does not change with training length. 📄 (§4.1, §5)
  - The main tuning need is learning rate and weight decay, whose best values differ from those under a schedule; schedule-free learning rates ran about 1 to 10 times higher. 📄 (§4.1, §6, App. G)
  - The best learning rate is stable across training lengths, except for short runs. 📄 (§5)
- **Batch size and learning rate.**
  - Shallue et al.: as batch size grows, the steps needed to reach a goal first fall in proportion ("perfect scaling"), then show diminishing returns, then stop improving. The best metaparameters follow no simple rule in batch size, so each batch size needs its own tuning. Their study covers SGD with and without momentum, not Adam. 📄 (Shallue et al. §4.1, Fig. 1)
  - Li et al.: for Adam-style optimizers, the best learning rate *rises then falls* as batch size grows. It peaks near a noise scale, and the peak shifts to larger batches as training progresses. 📄 (Li et al. 2024, Fig. 1, eq. 4, §6) Whether this carries over to schedule-free AdamW is 💭.

**In this study** ✅
- Loss: MSE. Fixed.
- Optimizer: schedule-free AdamW, fixed settings:
  - β = 0.9 📄; β₂ = 0.999 📄 (App. G: used in most tasks); ε = 1e-8 🧠 (implementation default);
  - averaging weights c₍t+1₎ = γ_t² / Σγ_i² 📄;
  - weight decay on all weight matrices, biases excluded, applied at y: z ← z − γ_t·λ·y. 📄 (Alg. 1, §4.4)
- **Warmup:** linear, 5% of steps, which is 625 steps or 25 epochs. 🧮 It counts as part of the optimizer.
  - Defazio et al. used 5% in their deep-learning experiments and 2% in their AlgoPerf entry. 📄 (App. G)
  - 5% was chosen because the smallest batches and the scale gap raise the risk of early instability. 💭
  - Warmup steps carry about 1.7% of the final averaging weight and cost about 400 full-rate steps, about 3% of training. 🧮
- **No schedule after warmup:** the base learning rate is constant from then on.
- **Training length:** 500 epochs, fixed by heuristic. Not a factor and not tuned. A fixed length also means no epoch count has to be chosen before the final-stage retrain.
- **Batch size coupled to data size:** K = 25 steps per epoch, so B ≈ N/25 and every recipe gets about 12,500 steps. Each epoch splits the N windows into 25 near-equal batches (sizes differ by at most 1). In the final stage B runs from about 30 to about 4,300. 🧮
- No batch-statistics normalization, so no batch-norm handling is needed. 📄 (§4.4)
- **Every measurement reads the evaluation weights x.**
- The later project must use schedule-free AdamW and read the evaluation weights for these characteristics to apply.

**Still open.** Nothing in this chapter.

**Worked example**
```text
# Recipe E, final stage
N = 21,000;  K = 25;  B ≈ N / K = 840;  epochs = 500;  steps = 12,500
warmup_steps = 0.05 × 12,500 = 625      # linear ramp of γ from ~0 to lr

per step:
    y = (1 - β)·z + β·x
    g = ∇MSE(y, minibatch)               # gradient at y
    z = AdamStep(z, g, γ_t, β2, ε)       # Adam step on z (bias correction folded into γ_t)
    z = z - γ_t·λ·y   (weight matrices only; λ > 0 because weight decay is on)
    c = γ_t² / Σ_{i≤t} γ_i²
    x = (1 - c)·x + c·z                  # evaluation weights
after each epoch: evaluate x on the test-asset training windows and on the test window
```
This shows the computations the brief fixes, not an implementation. The reference implementation is facebookresearch/schedule_free.

**Watch out**
- ⚠️ Adam's bias correction is folded into the learning rate, which may change how weight decay acts in early steps. 📄 (§4.4)
- ⚠️ ε is fixed while targets are about 0.01, so gradients are tiny and ε may dominate Adam's denominator. 📄 (Choi et al., for Adam) / 💭 (for schedule-free AdamW) → **C2**, **G10**
- ⚠️ **Batch size is tied to data size.** Tier and history set N, and B ≈ N/25, so gradient noise changes systematically along the data axes and cannot be separated from data-size effects. Comparisons within one tier × history cell, and tier 4 vs. 5, are not affected. → **C6**
- Gap: 500 epochs and K = 25 lack direct backing. → **G6**

**Further reading**
- Defazio et al. 2024, *The Road Less Scheduled*: §2 and Alg. 1 (method), §2.3 and Fig. 4 (implicit schedule), Fig. 1 (tracking the loss frontier), §4.4 (warmup, weight decay, batch norm), §5 (β and training length), App. G (settings). 📄
- Shallue et al. 2018, *Measuring the Effects of Data Parallelism*, §4.1 and Fig. 1. 📄
- Li et al. 2024, *Surge Phenomenon in Optimal Learning Rate and Batch Size Scaling*, Fig. 1. 📄

**Carry forward**
- Schedule-free AdamW tracks z, y and x; **x is the model**, and it behaves as if the learning rate decayed to "now".
- 5% warmup (25 epochs), then a constant learning rate. 500 epochs × 25 steps = 12,500 steps for every recipe.
- B ≈ N/25 runs from about 30 to 4,300. The batch-size and data-size confound is C6.

---

## Chapter 6. The recipe space

**In one line.** Five factors (depth, width, regularization, data tier, history) crossed into 480 recipes; everything else is fixed or out of scope.

**Builds on.** Chapters 3–5.

**Concept.**
- **Factor, option, recipe.** A factor is a design dimension, an option is one of its values, and a recipe is one option per factor.
- **Scientific, nuisance and fixed hyperparameters.** The Tuning Playbook sorts settings by role. 📄
  - *Scientific* ones are what you want to learn about.
  - *Nuisance* ones must be optimized away for each scientific setting so that comparisons are fair.
  - *Fixed* ones are held constant, which limits how far the conclusions reach.
- **Design spaces.** RegNet argues that comparing *distributions* of models from a design space is more robust and informative than comparing the best models found. 📄 (§3.1)

**In this study** ✅

| Factor (all scientific) | Options |
|---|---|
| Depth (hidden layers) | 1, 2, 4, 8 |
| Width (constant hidden width) | 8, 16, 32, 64 |
| Regularization | none; weight decay |
| Data tier | 1, 4, 12, 24, 24 (diverse) assets |
| History | 3, 7, 18 years |

- **Fixed:** MSE loss, schedule-free AdamW, leaky ReLU (0.01), one-step target. Findings hold only under these.
- **Cut from the space:**
  - the horizon factor (multi-step targets: step weighting and label overlap loaded onto the factor, and only t+1 was reported);
  - early dropout (fixed schedule, overlap with warmup, width dependence);
  - regularization "both" (it only adds the interaction);
  - MAE loss (the loss set was not comprehensive anyway; a fuller loss study may come later).
- **Excluded from scope:** Huber and other losses; other optimizers (AdamW with a schedule, SGD with momentum); other activations (SiLU, gated); dropout in any form; L1 regularization; normalization layers; gradient clipping; network shape; skip connections; explicit learning-rate schedules.
- **Size:** 10 categorical combinations (regularization × tier); full grid 4 × 4 × 2 × 5 × 3 = **480**. 🧮
- Every recipe is run once. The space is not changed based on results.

**Still open.** Nothing in this chapter.

**Worked example**
```text
recipe_E = {depth: 2, width: 32, regularization: "weight_decay", tier: 3, history: 7}
grid     = product(depth[4], width[4], regularization[2], tier[5], history[3])   # 480 recipes
```

**Watch out**
- ⚠️ Findings do not extend to other losses, optimizers, activations or horizons.
- ⚠️ Depth 8 × width 64 at tier 1 × 3 years means about 30,000 parameters against about 750 correlated windows, so many overfit failures are expected. ❔ Depths 4 and 8 straddle the hypothesized overfit boundary at depth 6; if it exists, it shows as a jump in failure rate, not at a specific depth. → **C9**
- Gaps: architecture ranges (**G4**); the role of weight decay in the bias regime under schedule-free AdamW (**G5**).

**Further reading**
- Tuning Playbook, "Identifying scientific, nuisance, and fixed hyperparameters". 📄
- RegNet, §3.1 (tools for design-space design). 📄

**Carry forward**
- Five scientific factors, 480 recipes, all run once. The space is closed.
- Anything outside the fixed set (MSE, schedule-free AdamW, leaky ReLU, t+1) is out of scope.

---

## Chapter 7. Fair footing: hyperparameter tuning

**In one line.** Every recipe tunes its learning rate (and λ, if it uses weight decay) with the same 32-trial quasi-random search over ranges shared by all recipes; a small calibration run sets those ranges.

**Builds on.** Chapters 3, 5 and 6.

**Concept.**
- **Why tune each recipe.** If one recipe gets a good learning rate and another a poor one, their difference reflects tuning luck, not the design choice. Nuisance hyperparameters must be optimized for each scientific setting. 📄 (Playbook, "Creating a set of studies")
- **Quasi-random search.** Instead of random points, use a low-discrepancy sequence such as scrambled Halton, which covers the space evenly. It does not adapt to results, so all trials can be drawn up front and run in parallel. The Playbook prefers it during exploration. 📄
- **Search spaces matter.** Choi et al. show that optimizer comparisons can flip when the search space changes. The search space may be the single most important factor behind published rankings. 📄 (Choi et al., abstract and §1)
- **Edge hits.** If the best value lands at the edge of the range, the true best may lie outside it. Shallue et al. reran searches in that case; the Playbook lists it as a standard check. 📄
- **Paired trials.** Giving all trials the same initialization and data order reduces the noise in comparisons *between* trials. 📄 (Bouthillier et al.)

**In this study** ✅
- Tuning happens per recipe, in the tuning stage. It is a means, not the subject of study. Everything about tuning that is not part of the experiment is handed to Optuna.
- **Tuned (nuisance):** learning rate, and λ when weight decay is on. β, β₂, ε and warmup stay fixed. This follows Defazio et al.'s practical need. 📄
- **Method:** Optuna's quasi-Monte Carlo sampler (scrambled Halton, fixed seed). All trials are drawn up front. Every trial runs the full 500 epochs, with no pruning.
- **Batched trials:** a recipe's trials train as one batched model, each member with its own hyperparameters, weights and optimizer state. All members share the initialization and data order, so comparisons between trials are paired. The averaging weights c_t do not depend on the learning-rate scale (it cancels), so they are the same for every member. 🧮
- **One protocol:** 32 trials per recipe, the same with or without weight decay, and the same in the calibration run.
- **Objective (maximized):** validation R² on t+1, using the evaluation weights, on the test asset, averaged over the last 25 epochs.
- **Calibration run:** 24 recipes at the extremes (Chapter 13), with wide ranges:
  - learning rate log-uniform in 1e-5 to 1e-1 🧠;
  - λ log-uniform in 1e-5 to 1 🧠.
  - The upper end leaves room for schedule-free learning rates, which ran about 1 to 10 times higher (for example 0.005 against 0.0005 on NanoGPT). 📄 (§4.1, App. G)
  - The lower end stays low because the approximation coefficient (about 18) multiplies first-layer steps. 💭
- **Main-run ranges:** one shared range per tuned hyperparameter, for *all* recipes. Why not one range per tier × history cell:
  - the best learning rate moves with batch size 📄 (Li et al., transfer 💭) and with width 🧠;
  - per-cell ranges would follow batch size only;
  - and each data cell would get its own search space, so the data-size factor would carry tuning quality. → **C6**
- **Range rule:** each main-run range spans from the lowest to the highest best value found in calibration, widened by 0.75 decade on each side. Because of the margin, the exact calibration values matter little.
- **Edge hits:** a recipe whose chosen value lies within one trial spacing of a range end is flagged, reported and marked on the lattices (Chapter 12). It is *not* re-tuned, since that would break the single protocol. The share of edge hits is reported. 📄 (Choi; Playbook)
- **Tuning-to-final transfer:** tuned values are reused unchanged in the final stage. The step count is the same and the batch is about 1.25 times larger. 🧮 No project paper covers this transfer for schedule-free AdamW. → **G8**
- λ is reported together with lr·λ.
- **Infeasible trials** (diverged or NaN) are recorded as failed with the worst objective; the infeasible fraction per recipe is reported.
- Leakage: no label crosses the tuning cutoff (Chapter 3), so no separate handling is needed.

**Still open.** Nothing in this chapter.

**Worked example**
```text
# Recipe E, tuning stage: N ≈ 16,800, B ≈ 672
space   = {lr: logUniform(main_lr_range), λ: logUniform(main_wd_range)}   # 2-D, since weight decay is on
trials  = QMC_Halton(scrambled, seed=fixed).sample(space, n=32)          # drawn up front
model   = batched(32 members, shared init and data order, member i uses trials[i])
train model for 500 epochs × 25 steps
score_i = mean(validation_R2_testasset(x_i, epochs 476..500))
best    = argmax_i score_i
edge_hit = best value within one trial spacing of either range end, for any tuned hyperparameter
infeasible_fraction = (# diverged or NaN members) / 32
# final stage: reuse best (lr, λ) unchanged; N ≈ 21,000, B ≈ 840 (≈ 1.25× larger)
```
Recipe E's data cell (tier 3 × 7 years) is one of the three calibration cells. Its capacity (2, 32) is not a calibration corner, so Recipe E itself is not calibrated.

**Watch out**
- ⚠️ **Unequal tuning effort.** With 32 trials in every recipe, weight-decay recipes spread their trials over two dimensions while no-decay recipes use one. If the best learning rate depends on λ, weight-decay recipes are tuned less precisely, so the regularization factor partly carries tuning quality. 💭
- ⚠️ Batched trials cost little memory, but compute still grows with the number of members once the hardware is saturated. 💭
- ⚠️ **Pairing may fade.** A one-bit change at initialization produces about as much variability as any other source 📄 (Summers & Dinneen, Table 3), and trials differ in hyperparameters from the first step. 💭
- ⚠️ Each trial uses one seed, so the objective is noisy, especially at tier 1 with 3 years (about 150 validation windows). → **C4**
- See **C2**, **C6**. Gaps: tuning-to-final transfer (**G8**); objective, ranges and trial count (**G9**).

**Further reading**
- Tuning Playbook: "Creating a set of studies"; "Striking a balance between informative and affordable experiments"; "Why use quasi-random search…"; "Extracting insight from experimental results" (edge check). 📄
- Choi et al. 2019, *On Empirical Comparisons of Optimizers for Deep Learning*, abstract and §1. 📄
- Shallue et al. 2018, §4 (per-batch-size retuning, boundary reruns). 📄

**Carry forward**
- 32 quasi-random trials per recipe, over ranges shared by all recipes and set by a 24-recipe calibration run plus a 0.75-decade margin.
- The objective is validation R² on the test asset, averaged over the last 25 epochs. Edge hits are flagged, not fixed.
- Tuned values are reused unchanged in the final stage (G8).

---

## Chapter 8. Randomness and judging differences

**In one line.** Each recipe is trained with 32 independent seeds. Differences between recipes are judged by *equivalence* tests with margins fixed by α, β and n alone.

**Builds on.** Chapters 5 and 7.

**Concept.**
- **Sources of run-to-run variation.**
  - Bouthillier et al. measured each source separately. Resampling the data (bootstrap) is by far the largest. Initialization is generally under half of it, on a par with data order. 📄 (Fig. 1)
  - Summers & Dinneen find that every source of nondeterminism they studied has a similar effect on variability. Even a one-bit change in one initial weight produces about as much, because training is unstable. 📄 (§3, Table 3)
  - Bouthillier et al. recommend about 29 runs for their comparison test. 📄 (App. C.3)
- **Equivalence testing.** A standard test can show that two things *differ*. It cannot show that they are the same, since a non-significant result may simply lack power.
  - An equivalence test turns the question around. Pick a margin Δ, then test whether you can *reject* that the difference is at least Δ.
  - Lakens describes the "two one-sided tests" (TOST) procedure: one test against the lower bound, one against the upper bound. If both reject, the difference lies within ±Δ. With α = 0.05 this is the same as the 90% confidence interval lying inside the bounds. 📄
  - When no theory sets Δ, a defensible choice is the smallest effect the available sample can detect. 📄 (Lakens, eq. 5, solved for the bound)

**In this study** ✅
- Every metric's variance is accounted for, using multiple seeds per recipe.
- **Seed count: 32.** The relative error of a seed standard deviation is about 13% at 32 seeds. 🧮
- **What a seed is.** One seed sets both initialization and data order; bundling them is supported by Summers & Dinneen (effects within about 20% of each other). 📄
  - Each seed draws from its own random stream, independent of which other seeds share its batch.
  - Streams are independent *across* recipes: seed i of one recipe shares nothing with seed i of another, so comparisons between recipes are unpaired.
  - Tuning trials within a recipe still share initialization and data order (Chapter 7).
- Seeds run batched and share the same data (the draw is fixed). Each seed shuffles its own minibatches. 💭
- **Error rates:** α = 0.05, β = 0.2, marked provisional by the owner.
- **Margins** are dimensionless, derived from α, β and n only, and fixed for every comparison:
  - **δ_μ** = (t₁₋α + t₁₋β/2) · √(2/n), in units of seed σ, with 2(n − 1) degrees of freedom. At n = 32: **δ_μ ≈ 0.74 σ**. 🧮 📄 (Lakens eq. 5)
  - **δ_σ** = (z₁₋α + z₁₋β/2) / √(n − 1), on a log standard-deviation ratio. At n = 32: **δ_σ ≈ 0.53**, a ratio of about 1.7. 🧮
- **Two levels of variance:** the seed standard deviation within a recipe, and the spread between recipes within an option. The functional ANOVA decomposition (Chapter 11) covers the second.
- **Hypothesis tests are used only inside regionalization, to decide merges** (Chapter 12). No test result is reported as a finding. Status compares point estimates with fixed thresholds, without a test (Chapter 10).

**Still open.** Nothing in this chapter.

**Worked example**
```text
for s in 1..32:
    rng_s = independent_stream(recipe_E, s)     # unrelated to seed s of any other recipe
    init_s, order_s = draw(rng_s)               # one seed sets both
train all 32 seeds as one batched model on the same data, each with its own shuffle
# per quantity q: mean_q = mean over seeds, sd_q = std over seeds
```

**Watch out**
- ⚠️ Reproducibility holds for the *distribution* across seeds, not for individual seeds. A different batching of seeds can change floating-point order, and under unstable training any such change makes individual seeds diverge. 📄 (Summers) / 💭 (the batching mechanism)
- ⚠️ T1 (Chapter 9) captures only initialization and data-order variance on one fixed dataset. Variance from sampling the data, the largest source in Bouthillier et al., is not measured. → **C7**

**Further reading**
- Bouthillier et al. 2021, *Accounting for Variance in Machine Learning Benchmarks*, Fig. 1 (sources) and App. C.3 (number of runs). 📄
- Summers & Dinneen 2021, *Nondeterminism and Instability in Neural Network Optimization*, §3 and Table 3. 📄
- Lakens 2017, *Equivalence Tests: A Practical Primer*, the TOST section and eq. 5. 📄

**Carry forward**
- 32 independent seeds per recipe; a seed sets initialization and data order; comparisons between recipes are unpaired.
- δ_μ ≈ 0.74 σ for means and δ_σ ≈ 0.53 for log standard-deviation ratios. Both are fixed in advance from α, β and n.
- Tests decide merges only; they are never findings.

---

## Chapter 9. What is measured

**In one line.** Per recipe and per seed: training and test R², the gap between them, how much predictions vary across seeds (T1), whether the curves are flat (V1), and two diagnostics. All are read from the evaluation weights at the end of the final stage.

**Builds on.** Chapters 1, 3, 5 and 8.

**Concept.**
- **Training and test curves.** A metric recorded after every epoch, on training and on held-out data.
- **Generalization gap.** Training performance minus test performance.
- **Bias–variance decomposition across seeds.** For a given input, train many seeds:
  - the *variance* term is how much their predictions disagree;
  - the *bias* term is the error of their average prediction.

  Neal et al. use the law of total variance to separate variance due to initialization (seeds on fixed data) from variance due to sampling the training data. 📄 (§2.3, eq. 4)
- **Flatness under schedule-free AdamW.** The evaluation weights x track the frontier of loss against training time. So a flat curve for x means that training longer, even with an ideal decay, would not help. 📄 (Defazio et al., Fig. 1, §2.3)

**In this study** ✅

All quantities use the t+1 output, the final stage and the evaluation weights. Every quantity is reported for every recipe together with its seed standard deviation.

- **Training curve:** R² per epoch on the test-asset training windows.
- **Test curve:** R² per epoch on the test window.

| Quantity | Definition | Read on | Feeds status? |
|---|---|---|---|
| **V1a slope** | Training curve: mean of the last 25 epochs minus mean of the preceding 25, in absolute R² units (not divided by 25). Reported as the magnitude of the seed-mean slope and the seed std of the slope | Test-asset training windows | Yes |
| **V1b slope** | Same construction on the test curve | Test window | Yes |
| **T1** | Mean over windows of the prediction variance across seeds, on each side. Reported per side and as the log ratio ½·log(T1_test / T1_train). This is the log of the ratio of root-mean variances, and it is positive in the overfitting direction, like the gap | Train side: test-asset training windows. Test side: test window | Yes (the log ratio) |
| **Training R²** | Mean of the last 25 epochs of the training curve | Test-asset training windows | No |
| **Test R²** | Mean of the last 25 epochs of the test curve | Test window | Only through V1b |
| **Generalization gap** | Training R² minus test R², per seed | Both | No (never a filter) |
| **Prediction spread** | Std over windows of the t+1 prediction, per seed; picks out near-constant predictors | Test window | No |
| **Relative weight change** | ‖x − x₁‖ / ‖x₁‖ over all weight matrices at the final epoch, x₁ being the initialization, per seed; indicates lazy vs. rich training and networks that barely moved | Parameters | No |

- T1 is the variance term of a bias–variance decomposition across seeds. It is the counterpart of Neal et al.'s variance due to initialization, here also including data order. 📄 Seeds share the target, so T1 equals the variance of the t+1 error.
- T1 is itself computed across seeds; its uncertainty is a jackknife over seeds (Chapter 12).

**Still open.** Nothing in this chapter.

**Worked example**
```text
# Recipe E, final stage; preds[s, w] = t+1 prediction of seed s on window w, from x after epoch 500
train_curve[s, e] = R2(test-asset training windows, seed s, epoch e)   # ≈ 1,750 windows
test_curve[s, e]  = R2(test window, seed s, epoch e)                   # ≈ 500 windows

V1a[s] = mean(train_curve[s, 476..500]) - mean(train_curve[s, 451..475])
V1b[s] = same on test_curve
report |mean_s V1a|, sd_s V1a   (same for V1b)

trainR2[s] = mean(train_curve[s, 476..500]);   testR2[s] = mean(test_curve[s, 476..500])
gap[s]     = trainR2[s] - testR2[s]

T1_side       = mean_w var_s(preds[s, w])          # computed on each side
T1_logratio   = 0.5 · log(T1_test / T1_train)
SE(T1_logratio) = jackknife over the 32 seeds

spread[s]     = std_w(preds_test[s, w])
rel_change[s] = ||x_s - x_init_s|| / ||x_init_s||  (weight matrices only)
```
The brief does not fix the variance estimator convention (dividing by n or by n − 1).

**Watch out**
- ⚠️ V1a reads the test asset only, not the pooled training objective. A flat curve on the test asset does not guarantee that the pooled objective is flat.
- ⚠️ Because x averages all steps since warmup, a steady drift of the base iterate shows up in x at reduced speed (half for a linear drift), and step-to-step noise is averaged away. 🧮 (idealized) The flatness threshold (Chapter 10) applies on this smoothed scale.
- ⚠️ The bias term itself (the error of the seed-mean prediction) is not among the quantities above; status uses the T1 log ratio instead. → **G12**
- ⚠️ Variance due to initialization depends on width and on parameters vs. data 📄 (Neal), and this grid crosses that boundary. So T1 moves for reasons other than overfitting. → **C6**, **C9**
- ⚠️ The gap is in R² units and the T1 log ratio is dimensionless; they share no scale.
- See **C1** (near-constant predictors), **C4** (history), **C7**, **C8** (test-window reuse). Gaps: **G11** (saturation criterion), **G13** (prediction spread).

**Further reading**
- Neal et al. 2018, §2.3 (decomposition, eq. 4) and §3.4 (variance due to initialization vs. width). 📄
- Defazio et al. 2024, Fig. 1 and §2.3. 📄
- Chizat et al.: parameter movement as a signal of lazy training (partial support for relative weight change). 📄

**Carry forward**
- Main quantities: test R², gap, T1 log ratio (positive means more seed disagreement on the test side), and V1a/V1b slopes over the last 50 epochs.
- Every quantity is read on the test asset, from the evaluation weights, and reported with its seed standard deviation.
- Prediction spread and relative weight change are diagnostics for trivial or unmoved networks.

---

## Chapter 10. Classifying recipes

**In one line.** A recipe is *failure-unsaturated* if either curve is still moving, otherwise *failure-overfit* if seeds disagree much more on test than on training windows, and otherwise *valid*.

**Builds on.** Chapters 8–9.

**Concept.**
- **Bias regime.** A network is in the bias regime when its error is dominated by underfitting rather than overfitting.
- **Variance as a proxy.** An overfitting network fits idiosyncrasies of its particular training run, so different seeds tend to agree on training data but disagree on unseen data. A large rise in seed disagreement from the training side to the test side is therefore used as the overfitting signal. This proxy is the study's own construction; no project paper establishes it. → **G12**

**In this study** ✅
- Status is based on V1 and the T1 log ratio.
- **Mapping, in order of precedence:**
  1. If the V1a or V1b slope magnitude exceeds the **flatness threshold**, the recipe is **failure-unsaturated**.
  2. Otherwise, if the T1 log ratio exceeds **δ_σ**, it is **failure-overfit**.
  3. Otherwise it is **valid**.
- **Flatness threshold** = δ_μ × (the median, over all recipes, of the seed standard deviation of test R²). It is one value, shared by V1a and V1b, both in R² units.
  - The rule is fixed in advance. The number is computed from the population after the runs, leaving no free choice.
  - Meaning: the change between the last two 25-epoch blocks is smaller than the test-R² difference the maps can resolve for a typical recipe. 💭 It says the end of training is flat *at map resolution*, not that further training could not help.
- **Overfit threshold** = δ_σ, so it adds no new number. At n = 32 it corresponds to T1_test / T1_train ≈ 2.9. 🧮
- Status compares point estimates with the thresholds, without a test.
- Failures are recorded with rates per option and per pair of options (Chapter 11).
- All recipes are reported and enter every map. Status is a categorical characteristic, regionalized like the others. Findings about the bias regime are read inside valid-status regions.

**Still open.** Nothing in this chapter.

**Worked example**
```text
flat_thr = δ_μ × median_over_480_recipes( sd_s(testR2) )      # one number, after all runs

status(E) =
    "failure-unsaturated" if |mean_s V1a| > flat_thr or |mean_s V1b| > flat_thr
    else "failure-overfit" if T1_logratio > δ_σ              # ≈ 0.53, i.e. T1 ratio ≈ 2.9
    else "valid"
```

**Watch out**
- ⚠️ δ_σ is the regionalization's resolution, not a theory-set boundary between regimes. → **G12**
- ⚠️ The training curve's seed std may differ from the test curve's, so one shared threshold is stricter on one side than the other. 💭
- ⚠️ The test window decides status *and* supplies test R², V1b and T1. Hyperparameters are fixed before it is read, but the flatness threshold depends on the same results it classifies, and quantities that feed status act as filters. → **C8**
- ⚠️ A near-zero-output network can pass V1 trivially. → **C1**
- See **C6**. Gaps: **G11** (saturation criterion), **G12**.

**Further reading**
- Neal et al. 2018, §2.3 (the decomposition behind T1). 📄
- Lakens 2017 (the margin δ_μ behind the threshold). 📄
- Candidate, not in the project: Jiang et al. 2022, *Assessing Generalization of SGD via Disagreement* (seed disagreement tracks test error; classification only). 🧠

**Carry forward**
- Status: unsaturated (a slope above threshold) takes precedence over overfit (T1 log ratio > δ_σ), and everything else is valid.
- Both thresholds come from the regionalization margins, so status resolution equals map resolution.
- Bias-regime findings are read only inside valid regions.

---

## Chapter 11. From recipes to characteristics

**In one line.** Beyond per-recipe numbers, the study reports failure rates by option and option pair, how results change with more data (data saturation), and a variance decomposition over factors.

**Builds on.** Chapters 6, 8, 9 and 10.

**Concept.**
- **Data saturation.** The point where adding data stops helping. It is a population-level trend, distinct from *training* saturation (Chapter 10).
- **Functional ANOVA** (Hutter et al. 2014). Split the variation of a performance function over its whole domain into a constant, *main effects* (one factor varied, all others averaged out) and *interaction effects* (what pairs or sets of factors add beyond their main effects). Each component's share of the total variance measures its importance. 📄 (§3.1, eqs. 4–6) Hutter et al. fit random forests because they observe only samples of configurations. Here the full grid is observed, so the decomposition can be computed directly from cell means. 💭

**In this study** ✅
- **Per recipe:** the quantities of Chapter 9. **Per region:** Chapter 12.
- **Over all recipes:**
  - Failure rates (unsaturated, overfit) per option and per pair of options. Failures are expected to come from combinations (C9). No failure-rate cutoff is applied; the space is not changed.
  - Data saturation, measured along history within each tier and along the nested financial tiers (1 → 2 → 3 → 4) within each history. Tier 5 is not on the size axis, because composition changes there (C5). It uses all recipes, with status against data size reported alongside (C8).
  - The functional ANOVA decomposition from cell means over the full grid, for both the seed mean and the seed standard deviation, with no surrogate model.
- **Secondary outputs:** tuned hyperparameter values, including the tuned learning rate against batch size as a diagnostic for C6 (Li et al.'s result covers Adam-style optimizers; transfer is 💭). Tuned values, edge hits and the infeasible fraction are shown in the map layout, not regionalized.
- **Deferred:** which DWT bands a network relies on (confounded by the volatility features, C2).
- **Out of scope:** feature learning, feature similarity and test-curve shape. Relative weight change is logged.

**Still open.** Nothing in this chapter.

**Worked example**
```text
# Recipe E contributes to:
failure_rate[option]           for each of its 5 options (e.g. depth=2, tier=3, …)
failure_rate[option_a, option_b] for each of its 10 option pairs
data_saturation along history:  tier 3 × {3, 7, 18} years   (E sits at 7)
data_saturation along tiers:    {1, 2, 3, 4} × 7 years      (E sits at tier 3)
fANOVA:                         cell mean of E (and its seed sd) enters the 480-cell table
lr_vs_B diagnostic:             (B ≈ 840, tuned lr of E)
```

**Watch out**
- ⚠️ Data-saturation curves mix more data with less gradient noise, because B ≈ N/25. → **C6**
- ⚠️ Status shares the test window with everything else. → **C8**
- ⚠️ Band attribution is confounded by the volatility features. → **C2**
- Gap: data-saturation measurement has no backing. → **G14** (candidate: Hestness et al. 2017 🧠)

**Further reading**
- Hutter, Hoos & Leyton-Brown 2014, *An Efficient Approach for Assessing Hyperparameter Importance*, §3.1. 📄
- Li et al. 2024 (for the learning rate vs. batch-size diagnostic). 📄

**Carry forward**
- Failure rates come per option and per option pair. Data saturation runs along history and the nested financial tiers, never tier 5.
- Functional ANOVA is computed directly over the full grid, for the mean and the seed standard deviation.

---

## Chapter 12. Maps: regionalization and visual outputs

**In one line.** For each of four quantities, recipes are merged step by step into connected regions of equivalent results; the regions are drawn as lattices, region graphs and distribution plots fixed in advance.

**Builds on.** Chapters 6, 8, 9 and 10.

**Concept.**
- **Regionalization** comes from geography: dividing spatial units, such as counties, into contiguous regions that are internally similar. Guo's REDCAP family does this by agglomerative clustering constrained by contiguity. Start with every unit as its own region and repeatedly merge the most similar *adjacent* pair. 📄 (Guo 2008, §1, §3)
  - **Dynamic contiguity:** after a merge, the new region's neighbours are the union of its members' neighbours. 📄
  - **Linkage** defines similarity between regions: single (closest pair), average, or complete (farthest pair). Single linkage tends to "chain"; Guo finds full-order complete and average linkage best for internal homogeneity. 📄 (§2, §5)
  - Guo cuts the merge tree into a chosen number of regions.
- Here the "space" is the factor grid, with adjacency meaning a one-step change in one factor.
- **Empirical distribution function (EDF).** The fraction of models whose value lies below a given level. Plotting one curve per option compares whole distributions, not best cases. 📄 (RegNet eq. 1, Fig. 2)

**In this study** ✅

*What is mapped*

| Map | Quantity | Merge rule |
|---|---|---|
| Test R² | per-seed values | mean rule |
| Generalization gap | per-seed values | mean rule |
| T1, test vs. train | T1 log ratio | log-ratio rule |
| Status | label | connected areas with the same label; no test |

- **Not mapped:**
  - training R² (it equals test R² plus the gap, per seed);
  - T1 per side, prediction spread and relative weight change (per-recipe diagnostics);
  - seed std (reported per region as σ_R, not tested);
  - failure rate (it becomes the status composition of a region);
  - data saturation (a change along history and tier).
- **Not regionalized, but shown in the map layout:** tuned hyperparameters, edge hits, and the infeasible fraction (one value per recipe, no seeds, so no standard error).

*Adjacency*
- Nodes are all 480 recipes.
- Depth, width and history: neighbouring levels.
- Tier: 1–2, 2–3, 3–4, 4–5 (C5).
- Regularization: the two options.
- About 6.9 neighbours per recipe on average. 🧮
- Merged regions take the union of their members' neighbours. 📄 (Guo)

*Summaries*
- Recipe: mean x̄ and sd s over n seeds.
- Region:
  - μ_R = mean of the members' means;
  - σ_R² = mean of the members' seed variances;
  - heterogeneity (the spread of members' means), reported separately;
  - SE(μ_R) = √(Σ sⱼ² / n) / n_R.

*Merge rules* (margins from Chapter 8)
- **Mean rule (equivalence).** H₀: |μ_A − μ_B| ≥ δ_μ·σ_p.
  - D = (|μ_A − μ_B| + t₁₋α·SE_d) / (δ_μ·σ_p), with SE_d = √(SE_A² + SE_B²), σ_p = √((σ_A² + σ_B²)/2), and Welch degrees of freedom.
  - The pair passes if D < 1. This is exactly TOST rejecting H₀ at α. 📄 (Lakens: Welch by default; equivalent to the 90% confidence interval lying inside the bounds)
- **Log-ratio rule (T1 map).**
  - A recipe's value is its T1 log ratio, with a jackknife standard error. A region's value is the mean of its members' values, with SE √(Σ SEⱼ²) / n_R.
  - D = (|difference| + z₁₋α·SE) / δ_σ; the pair passes if D < 1.

*Procedure*
1. Every recipe starts as its own region.
2. Compute D for every adjacent pair.
3. Merge the pair with the smallest D (exact ties broken by recipe ID).
4. Recompute the merged region's summary and the D values of its adjacent pairs.
5. Stop when the smallest D is ≥ 1.

The procedure is deterministic, with no starting point to choose.

*Output per region:* members, size, μ_R, σ_R, heterogeneity, and which factors the region spans.

*Visual outputs* (fixed in advance, produced in full for every mapped quantity)
- **Recipe value:** the seed mean (test R², gap), the T1 log ratio, or the label (status).
- **Colors:** a diverging scale centred at 0 for test R², gap and T1 (0 means persistence, no gap, or equal seed variance); one fixed color per status label. Each quantity keeps one scale across all its figures.
- **Lattice:**
  - one per mapped quantity per regularization option;
  - outer grid: tier as rows (1 to 5), history as columns (3, 7, 18);
  - each outer cell is a 4 × 4 heatmap, depth as rows (1, 2, 4, 8) by width as columns (8, 16, 32, 64);
  - each square shows its value as color and its region ID as a label;
  - region borders are drawn between depth and width neighbours; borders along tier, history and regularization are read from the region IDs;
  - edge hits are marked with a symbol.
- **Region graph:**
  - one per mapped quantity;
  - nodes are regions, labeled by ID, sized by recipe count, colored by μ_R (status: by label);
  - edges join regions that contain an adjacent pair, with width proportional to the number of such pairs;
  - layout is deterministic.
- **EDF panels:** for test R², gap and T1, one panel per factor and one curve per option. 📄 (RegNet eq. 1) Status has no EDF; its per-option composition is the failure-rate table.
- **Tuning lattices** (not regionalized): same layout, without outlines. They show the tuned learning rate (both regularization options), tuned λ (weight decay only) and the infeasible fraction (both options). Borders that coincide with jumps in tuned learning rate are read by comparing lattices side by side.
- **Count:** 8 map lattices, 4 region graphs, 15 EDF panels, 5 tuning lattices. 🧮

**Still open.** Nothing in this chapter.

**Worked example**
```text
neighbours(E) = {
  depth 1 and depth 4                     (width 32, wd, tier 3, 7y)
  width 16 and width 64                   (depth 2, wd, tier 3, 7y)
  tier 2 and tier 4                       (depth 2, width 32, wd, 7y)
  history 3y and 18y                      (depth 2, width 32, wd, tier 3)
  no weight decay                         (depth 2, width 32, tier 3, 7y)
}   # 9 neighbours (E is interior on every ordered axis)

# regionalization of one quantity q with the mean rule
regions = {each recipe: own region}
loop:
    D_min, (A, B) = min over adjacent region pairs of D(A, B)
    if D_min >= 1: stop
    merge A and B; neighbours(AB) = neighbours(A) ∪ neighbours(B) − {A, B}
    recompute μ_R, σ_R, SE for AB and D for its adjacent pairs
```
Recipe E appears in the "weight decay" lattice, in the outer cell (tier 3, 7 years), in the square (depth 2, width 32).

**Watch out**
- ⚠️ α holds per merge, not per map. Later merges reuse data, so the nominal α is approximate. 💭
- ⚠️ The margins assume near-normal per-seed values; no normality check or tail adjustment is planned. 💭
- ⚠️ The plug-in σ_p approximates the exact noncentral-t test. 🧠
- ⚠️ The tolerated absolute difference scales with σ, so noisy pairs merge at larger differences. This distributional view was chosen consciously; Lakens notes the same property of standardized bounds. 📄
- ⚠️ **Centroid-like merging.** Merges compare region summaries, whereas Guo's evidence favours complete linkage. A drifting summary can admit a recipe that is equivalent to the summary but not to the member it borders; heterogeneity is reported for this reason. Centroid merges can also produce inversions in the merge tree. 📄 / 🧠
- ⚠️ Stopping by test has no counterpart in Guo, who cuts to a chosen number of regions. → **G15**
- ⚠️ Tuning variance is not in the seed standard errors, so differences between neighbours partly reflect tuning luck. The tuning lattices and edge-hit marks are the check. 📄 (Bouthillier)
- ⚠️ EDF panels pool over all other factors, hiding effects that exist only in some combinations; the lattices show those. 💭
- ⚠️ Region graphs with many small regions become hard to read; the lattice stays readable. 💭
- See **C1**, **C4** (gap regions spanning history options mix training-side periods), **C6**, **C7**, **C8**. Gap: **G15**.

**Further reading**
- Guo 2008, *Regionalization with Dynamically Constrained Agglomerative Clustering and Partitioning (REDCAP)*: §1 (problem), §2 (SKATER and chaining), §3 (contiguity-constrained clustering), §5 (evaluation). 📄
- Lakens 2017 (TOST, Welch). 📄
- RegNet, §3.1, eq. 1 and Fig. 2 (EDFs). 📄

**Carry forward**
- Four maps: test R², gap, T1 log ratio, status. Regions are connected and merged by equivalence until no adjacent pair passes.
- Adjacency is a one-step change in one factor (about 6.9 neighbours per recipe on average).
- The output set is fixed: 8 map lattices, 4 region graphs, 15 EDF panels and 5 tuning lattices.

---

## Chapter 13. Running it

**In one line.** Calibrate 24 extreme recipes, run all 480 recipes (32 trials, then 32 seeds each), then analyze: about 31,500 training runs in all.

**Builds on.** Chapters 7, 8 and 12.

**Concept.** The order matters. Search ranges must be set before the main run, so that every analyzed recipe sees the same protocol, and the analysis must not feed back into the design.

**In this study** ✅
- Every recipe is run once: no sampling, no iteration. Changing the recipe space based on results is out of scope.
- **Run order**
  1. **Calibration run:** 24 recipes at the extremes of what moves the best hyperparameters:
     - three batch-size cells: tier 1 × 3 years (B ≈ 30); tier 3 × 7 years (the middle, because the best Adam learning rate peaks at an intermediate batch size 📄 Li et al.); tier 4 × 18 years (B ≈ 4,300);
     - four capacity corners: (depth, width) = (1, 8), (1, 64), (8, 8), (8, 64);
     - both regularization options.

     Tier 5 is left out because it has tier 4's batch size. 💭 The run uses wide ranges, 32 trials each, tuning stage only. Its best values set the main-run ranges; its results are otherwise discarded.
  2. **Main run:** all 480 recipes, 32 trials each in the tuning stage, then 32 seeds each in the final stage.
  3. **Analysis:** quantities (Ch. 9), status (Ch. 10), characteristics and functional ANOVA (Ch. 11), regionalization and visual outputs (Ch. 12).
- **Cost** 🧮 (rough). This uses the rate of a prior experiment, 300 runs in about 2 hours with batched seeds and parallel small models; that rate may not hold across model sizes or batch widths.
  - Calibration: about 770 runs (about 5 h).
  - Main run: about 15,400 tuning runs plus 15,400 final runs, about 30,700 (about 205 h).
  - Wall time depends on how many batched members fit before the hardware saturates.

**Still open**
- 🔓 **Batch-size control.** Rerun a subset at more steps per epoch to separate batch-size from data-size effects (C6). Deferred by the owner; may be investigated later. *Working reading:* not run.

**Worked example**
```text
calibration_cells   = [(tier 1, 3y), (tier 3, 7y), (tier 4, 18y)]
calibration_corners = [(1, 8), (1, 64), (8, 8), (8, 64)]
calibration = cells × corners × {none, wd}           # 24 recipes × 32 trials = 768 runs
ranges      = widen([min best, max best], 0.75 decade each side)   per tuned hyperparameter
main        = 480 recipes × (32 trials + 32 seeds)   # 30,720 runs
# Recipe E: its cell (tier 3, 7y) is calibrated; E itself is not. It enters the main run only.
```

**Watch out**
- ⚠️ Record the ranges and computed thresholds actually used, so the later project can reproduce the protocol.

**Further reading**
- Li et al. 2024, Fig. 1 (why a middle batch-size cell is included). 📄

**Carry forward**
- Calibration (24 recipes) → main run (480 × 32 trials + 480 × 32 seeds) → analysis. Nothing loops back.

---

## Chapter 14. Known caveats (confound register)

**In one line.** Nine root causes (one now removed) that can make an effect look like something it is not. Each is listed with where it bites and what the study does about it.

**Builds on.** All earlier chapters.

**Concept.** A *confound* is a second cause that moves together with the factor you are studying, so an observed effect cannot be attributed to the factor alone. A confound is not a mistake; it is a limit on interpretation, and listing it is what makes the findings readable.

| ID | Root cause | Where it bites | What the study does |
|---|---|---|---|
| **C1** | **Zero-return baseline** (Ch. 1). The target is near zero and persistence predicts zero | A trivial, near-constant network can score R² ≈ 0, look low-variance (small T1) and pass V1 trivially ❔ | Reports prediction spread and relative weight change (Ch. 9); check how they separate within failure rates |
| **C2** | **Input representation** (Ch. 2, under the parked working reading) | The approximation coefficient (about 18) dwarfs the other inputs (about 0.01–0.1) 📏 and alone can leave the training range under drift. Input scale acts like an extra initialization scale, shifting lazy vs. rich 💭 (from Chizat, Kunin). The learning-rate range must span the gap. ε may dominate Adam's denominator 📄 (Choi, for Adam) / ❔ (size of the effect). The volatility features are exact functions of the details 🧮, so band attribution is confounded | Wide calibration ranges; band attribution deferred. Not grounds to reopen the parked item |
| **C3** | *Removed 2026-11-02* (horizon factor cut) | n/a | n/a |
| **C4** | **History length** (Ch. 3) | It also changes regime content (2008 is inside 18 years) 💭, tuning noise (about 150 validation windows at tier 1, 3 years), training-side noise (about 750 windows), and time distance to the test window. Drift can mimic overfitting in the T1 log ratio. Gap regions spanning history options mix scoring periods | Named for interpretation; no correction |
| **C5** | **Data pool and tiers** (Ch. 3) | Survivorship: current members, tenure is a lower bound on listing history, today's sector labels, dual share classes not deduplicated | Tier 4 vs. 5 isolates composition at equal size; tier 5 kept off the data-size axis |
| **C6** | **Data-size couplings** (Ch. 5, 7, 11). B ≈ N/25, so gradient noise moves with tier and history | *Gains:* equal steps and epochs everywhere, so warmup and the V1 windows mean the same thing in every recipe. Per-recipe tuning absorbs part of the batch effect, and at saturation batch size barely affects solution quality in small workloads 📄 (Playbook; Shallue — SGD family, so 💭 here). *Hides:* that guidance concerns best performance, not characteristics. Small batches regularize and large ones can overfit more 📄 (Playbook), and recipes without weight decay have no tuned regularizer to compensate. Per-step noise scales with lr/B; x averages much of it, so V1 and T1 depend less on B, but not necessarily not at all 💭. Data-saturation curves mix more data with less noise | Shared tuning ranges; tuned lr vs. B diagnostic; batch-size control deferred (Ch. 13). Unaffected: comparisons within one tier × history cell, and tier 4 vs. 5 |
| **C7** | **Single draw, small effective sample** (Ch. 3, 8) | Conclusions are conditional on one asset draw, one test asset and one test period (about 500 windows). T1 omits data-sampling variance, the largest source 📄 (Bouthillier). Overlapping windows are autocorrelated | Stated as conditional; → G17 |
| **C8** | **Test-window reuse and status** (Ch. 9–12) | The test window feeds V1b, test R² and test-side T1, *and* decides status. The flatness threshold's value comes from the same results it classifies (by a fixed rule). Status-feeding quantities act as filters | Hyperparameters fixed before the test window is read; rule fixed in advance; findings on filter quantities read as conditional on validity |
| **C9** | **Capacity vs. data** (Ch. 4, 6) | Widths 8 and 16 are bottlenecks against 19 inputs. Depth 8 × width 64 has about 30,000 parameters vs. about 750 correlated windows 🧮, so many overfit failures are expected ❔. Depths 4 and 8 straddle the hypothesized boundary at 6 | Visible as jumps in failure rate (Ch. 11) |

**In this study / Still open / Worked example.** See the table above; there are no separate items. For Recipe E, C6 applies to any comparison with other tiers or histories, and C4 applies along history. C9 is mild: about 1,729 parameters against about 21,000 windows. 🧮

**Further reading.** Shallue et al. 2018; Tuning Playbook (batch size); Bouthillier et al. 2021. 📄

**Carry forward**
- C6 (batch size moves with data size) and C8 (one test window does everything) touch nearly every finding.
- C1 and C2 are why the diagnostics and the wide tuning ranges exist.

---

## Chapter 15. Where evidence is thin (evidence-gap register)

**In one line.** Seventeen design choices (two now removed) that no project paper directly supports, with the nearest support and what would close each gap.

**Builds on.** All earlier chapters.

**Concept.** A gap is not a warning that a choice is wrong. It marks where the study rests on its own reasoning rather than on literature, and therefore where further reading or a small check would add the most.

| ID | Choice lacking backing | Chapter | Nearest support | Would close it |
|---|---|---|---|---|
| G1 | Input representation (Haar inside each window, level kept, volatility features, 16-day lookback, no scaling) | 2 | Rhif (review only) | Studies of wavelet-coefficient inputs and their scaling for neural return forecasting |
| G2 | Data design (tiers, single test asset, test-asset-only evaluation, histories, 80/20 split, merged validation, no embargo) | 1, 3 | Gu, Kelly & Xiu (pooling) | Evaluation-protocol literature for financial ML |
| G3 | *Removed 2026-11-02* (horizon factor cut) | n/a | n/a | n/a |
| G4 | Depth and width ranges; hypothesized overfit boundary at depth 6 | 4, 6 | Neal, Summers (variance only); Gu et al. (shallow beats deep) | n/a |
| G5 | Weight decay's role in the bias regime under schedule-free AdamW | 6 | Playbook (general) | D'Angelo et al. 2024 (candidate 🧠) |
| G6 | 500 epochs, K = 25 | 5 | Consequences only (Shallue, Playbook) | n/a |
| G7 | PyTorch default initialization 🧠 | 4 | Chizat, Kunin (initialization scale moves lazy vs. rich) | n/a |
| G8 | Reusing tuned values at the same steps and a 1.25× larger batch, under schedule-free AdamW | 7 | Defazio (training length only) | n/a |
| G9 | Tuning objective (last-25-epoch mean); shared ranges from extreme recipes plus 0.75 decade; 32 trials | 7 | Choi (ranges, edge check); Defazio (learning-rate shift); Playbook (search-space refinement); Li et al. (batch-size dependence) | n/a |
| G10 | Fixed ε with small gradients | 5 | Choi (Adam: ε matters); Defazio (ε not tuned) | n/a |
| G11 | Saturation criterion (V1 construction on the test asset; shared flatness threshold from δ_μ and the median seed std) | 9, 10 | Defazio (meaning of flatness); Lakens (margin) | No paper on the criterion itself |
| G12 | Overfitting criterion via the T1 log ratio with δ_σ; no direct measure of the bias share | 9, 10 | Neal (decomposition); Summers (diversity) | Jiang et al. 2022 (candidate 🧠; classification only) |
| G13 | Prediction spread | 9 | None (relative weight change: Chizat, partial) | n/a |
| G14 | Data-saturation measurement | 11 | None | Hestness et al. 2017 (candidate 🧠) |
| G15 | Regionalization details (jackknife SE and the log-ratio rule; centroid-like merging; stopping by test; error control over sequential merges) | 12 | Lakens (means only); Guo (linkage, fixed number of regions); Berger & Hsu (cite only) | n/a |
| G16 | *Removed 2026-11-02* (full grid run once; no sampling or iterations) | n/a | n/a | n/a |
| G17 | Survivorship; effective sample size of overlapping windows | 3 | None | n/a |

**Still open / Worked example.** Not applicable to this chapter.

**Further reading.** Candidate additions (not in the project, 🧠):
- LeCun et al. 1998 (*Efficient BackProp*, input scaling), only if C2 needs a citation rather than an estimate;
- Jiang et al. 2022 (G12);
- D'Angelo, Andriushchenko et al. 2024 (G5);
- Hestness et al. 2017 (G14);
- Kelly, Malamud & Zhou 2024 (*The Virtue of Complexity in Return Prediction*; C1, C9).

**Carry forward**
- The least-supported parts are the status criteria (G11, G12), the regionalization rules (G15) and the input design (G1).

---

## Appendix A. Glossary

Alphabetical. Each entry points to the chapter that explains it.

| Term | Meaning | Ch. |
|---|---|---|
| Adjacent | Two recipes differing in one factor by one step (any two options for unordered factors) | 12 |
| Bias regime | Error of a training-saturated network dominated by underfitting rather than overfitting | 0, 10 |
| Calibration run | 24 extreme recipes tuned with wide ranges before the main run, to set the main-run ranges; otherwise discarded | 7, 13 |
| Characteristic | A measured property of a recipe, choice or region; not a ranking | 11 |
| Cutoff | The tuning cutoff or the origin: the last target date of a stage | 3 |
| Data saturation | Where more data stops helping; population-level only | 11 |
| Data tier | The subset of assets a network trains on; one of five | 3 |
| Data universe | The pool of assets and dates the study draws from | 3 |
| Edge hit | A tuned value within one trial spacing of a range end | 7 |
| Evaluation weights (x) | The running weighted average that schedule-free AdamW returns as the model; every measurement reads them | 5 |
| Factor / option | A recipe dimension / one of its values | 6 |
| Final stage | Retraining with tuned values on all final-stage training data; its results are analyzed | 3 |
| Generalization gap | Training R² minus test R² | 9 |
| Main run | All 480 recipes under one tuning protocol; its results are analyzed | 13 |
| Prediction variance across seeds | For one window, the variance across seeds of the t+1 prediction (= variance of the t+1 error) | 9 |
| R² | R² of the t+1 prediction relative to persistence (a zero-return forecast), on the test asset | 1 |
| Recipe | One option per factor | 6 |
| Recipe status | Valid, failure-unsaturated or failure-overfit | 10 |
| Region | A connected set of adjacent recipes with equivalent results for one quantity | 12 |
| Regionalization | The procedure that partitions recipes into regions, separately per mapped quantity | 12 |
| Seed | One training run of a recipe; sets initialization and data order | 8 |
| Seed std | Standard deviation across seeds of a scalar quantity | 8 |
| Target | The one-step return from the last input date | 1 |
| Task | Predicting a window's target from its input | 1 |
| Test asset | The single asset every recipe is evaluated on; tier 1's asset, in every tier | 3 |
| Test curve | R² per epoch on the test window, final stage | 9 |
| Test-asset training windows | The test asset's windows in the final-stage training data | 3 |
| Timeline | History start → tuning-stage training → tuning cutoff → validation window → origin → test window | 3 |
| Training curve | R² per epoch on the test-asset training windows, final stage | 9 |
| Training saturation | End of training where both the training curve and the test curve are flat (measured by V1) | 9, 10 |
| Trial | One hyperparameter configuration of a recipe, trained in the tuning stage | 7 |
| Tuning stage | Trains on tuning-stage data and tunes on the validation window | 3 |
| Window | One sample: a lookback input ending at date t, with its target | 2 |

## Appendix B. Symbols

| Symbol | Meaning | Ch. |
|---|---|---|
| P(t), r | Closing price; one-step log return | 1 |
| z, y, x | Schedule-free sequences: base iterate, gradient point, evaluation weights | 5 |
| x₁ | Initialization (start of x) | 9 |
| β, β₂, ε | Schedule-free interpolation; Adam second-moment rate; Adam denominator constant | 5 |
| γ_t, c₍t+1₎ | Learning rate at step t; averaging weight γ_t² / Σγ_i² | 5 |
| λ | Weight-decay strength | 5, 7 |
| N, B, K | Windows per stage; batch size (≈ N/K); steps per epoch (25) | 5 |
| n | Seeds per recipe (32) | 8 |
| α, β (tests) | Type I error (0.05); Type II error (0.2). Not the optimizer β | 8 |
| δ_μ, δ_σ | Equivalence margins for means (in σ units) and for log std ratios | 8 |
| V1a, V1b | Training- and test-curve slopes over the last 50 epochs | 9 |
| T1 | Mean over windows of the prediction variance across seeds | 9 |
| μ_R, σ_R, n_R | Region mean, pooled seed std, member count | 12 |
| D | Merge statistic; a pair passes if D < 1 | 12 |

## Appendix C. Numbers at a glance

| Item | Value | Ch. |
|---|---|---|
| Lookback | 16 trading days | 2 |
| Inputs | 19 (15 details + 1 approximation + 3 volatility features) | 2 |
| Approximation coefficient at about $100 | ≈ 18; other inputs ≈ 0.01–0.1 📏 | 2 |
| Tiers (assets) | 1, 4, 12, 24, 24 (diverse) | 3 |
| History | 3, 7, 18 years | 3 |
| Windows per asset-year | ≈ 250 🧮 | 3 |
| Windows per recipe (final stage) | ≈ 750 to 108,000 (×144) 🧮 | 3 |
| Test-asset training windows | ≈ 750 to 4,500 🧮 | 3 |
| Validation window | ≈ 0.6 / 1.4 / 3.6 years; ≈ 150 windows at tier 1, 3 years 🧮 | 3 |
| Test window | 2 years, ≈ 500 windows | 3 |
| Tuning / validation split | 80 / 20 | 3 |
| Depth / width | {1, 2, 4, 8} / {8, 16, 32, 64} | 4, 6 |
| Parameters | ≈ 170 (depth 1, width 8) to ≈ 30,000 (depth 8, width 64) 🧮 | 4 |
| Leaky ReLU slope | 0.01 | 4 |
| β / β₂ / ε | 0.9 / 0.999 / 1e-8 | 5 |
| Epochs × steps per epoch | 500 × 25 = 12,500 steps | 5 |
| Warmup | 5% = 625 steps = 25 epochs; ≈ 1.7% of averaging weight; ≈ 3% of training 🧮 | 5 |
| Batch size (final stage) | ≈ 30 to ≈ 4,300 🧮 | 5 |
| Tuning → final batch | ≈ 1.25× larger 🧮 | 7 |
| Recipes | 480 (10 categorical combinations) | 6 |
| Trials per recipe | 32 | 7 |
| Calibration ranges | lr 1e-5–1e-1; λ 1e-5–1 🧠 | 7 |
| Range margin | 0.75 decade each side | 7 |
| Objective / R² window | Mean of the last 25 epochs | 7, 9 |
| Seeds per recipe | 32; ≈ 13% relative error of a seed std 🧮 | 8 |
| α / β | 0.05 / 0.2 (provisional) | 8 |
| δ_μ at n = 32 | ≈ 0.74 σ 🧮 | 8 |
| δ_σ at n = 32 | ≈ 0.53 (std ratio ≈ 1.7) 🧮 | 8 |
| V1 window | Last 50 epochs (two 25-epoch blocks) | 9 |
| Overfit threshold | T1 ratio ≈ 2.9 🧮 | 10 |
| Neighbours per recipe | ≈ 6.9 on average 🧮 | 12 |
| Visual outputs | 8 lattices, 4 region graphs, 15 EDF panels, 5 tuning lattices 🧮 | 12 |
| Calibration | 24 recipes, ≈ 770 runs, ≈ 5 h 🧮 | 13 |
| Main run | ≈ 30,700 runs, ≈ 205 h 🧮 | 13 |

## Appendix D. Literature map

| Source | In project | Used in chapters |
|---|---|---|
| Defazio et al. 2024, *The Road Less Scheduled* | ✓ | 5, 7, 9, 10 |
| Tuning Playbook | ✓ | 6, 7, 14 |
| Choi et al. 2019, *On Empirical Comparisons of Optimizers* | ✓ | 5, 7 |
| Chizat, Oyallon & Bach, *On Lazy Training* | ✓ | 4, 9 |
| Kunin et al. 2024, *Get Rich Quick* | ✓ | 4, 14 |
| Bouthillier et al. 2021, *Accounting for Variance in ML Benchmarks* | ✓ | 7, 8, 12 |
| Summers & Dinneen 2021, *Nondeterminism and Instability* | ✓ | 7, 8 |
| Hutter et al. 2014 (fANOVA) | ✓ | 11 |
| Shallue et al. 2018, *Measuring the Effects of Data Parallelism* | ✓ | 5, 7, 14 |
| Neal et al. 2018, *A Modern Take on the Bias-Variance Tradeoff* | ✓ | 4, 9, 10 |
| Lakens 2017, *Equivalence Tests* | ✓ | 8, 10, 12 |
| Guo 2008, REDCAP | ✓ | 12 |
| Radosavovic et al. 2020 (RegNet) | ✓ | 0, 6, 12 |
| Li et al. 2024, *Surge Phenomenon* | ✓ | 5, 7, 11, 13 |
| Gu, Kelly & Xiu 2020 | ✓ (not yet cited in a brief section) | 1, 3, 4 |
| Xu et al. 2021, *How Neural Networks Extrapolate* | ✓ (not yet cited in a brief section) | 2 |
| Rhif et al. 2019 (wavelet review) | ✓ (guide) | 2 |
| Tawakuli et al. 2024 (preprocessing survey) | ✓ (guide) | 2 |
| Optuna; facebookresearch/schedule_free | Tools | 5, 7 |
| Akiba et al. 2019 (Optuna) | Cite only | 7 |
| Schuirmann 1987 (TOST); Berger & Hsu 1996 (intersection–union tests), both as cited in Lakens | Cite only | 8, 15 |
| Percival & Walden (wavelet methods) | Cite only, if a DWT reference is needed | 2 |

The brief's §12 also records removed and considered-but-not-added sources, with reasons.

## Appendix E. Brief ↔ guide cross-reference

| Brief | Guide |
|---|---|
| Purpose | Ch. 0 |
| Glossary | Appendix A (and the chapter where each term is introduced) |
| §1 Data universe and task | Ch. 1 (target, metric), Ch. 2 (inputs), Ch. 3 (assets, timeline, boundaries) |
| §2 Recipe space | Ch. 4 (architecture), Ch. 6 (factors, grid, scope) |
| §3 Training protocol | Ch. 4 (initialization), Ch. 5 |
| §4 Hyperparameter tuning | Ch. 7 |
| §5 Seeds and error rates | Ch. 8 (functional ANOVA in Ch. 11) |
| §6 Measurements | Ch. 9 |
| §7 Recipe status | Ch. 10 |
| §8 Characteristics | Ch. 11 |
| §9 Regionalization | Ch. 12 |
| §10 Scale and run | Ch. 13 |
| §11 Confound register (C1–C9) | Ch. 14; first introduced in C1 → 1, C2 → 2, C4/C5/C7 → 3, C9 → 4, C6 → 5, C8 → 10 |
| §12 Literature and gap register (G1–G17) | Ch. 15, Appendix D |
