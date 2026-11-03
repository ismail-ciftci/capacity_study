# Project Guide: Characteristics of Training-Saturated MLPs in the Bias Regime

A lecture-style companion to `PROJECT_BRIEF.md`, for anyone joining the project.

- **Chapter n explains brief section Sn.** The brief is the specification and the source of truth; this guide teaches it. The guide adds no decisions. If the two disagree, the brief wins.
- **Numbers live in the brief.** The guide uses the numbers that carry the story and works one recipe through in full (the worked example). Complete lists of settings, thresholds and warnings are in the brief section of the same number.
- **Each chapter is checked against its brief section.** Under every chapter title, the line `Synced: Sn <hash>` records the version of brief section Sn the chapter was last reviewed against. When the brief changes, `tests/test_docs.py` names the chapters to review.

---

## How to read this guide

Chapters build from the bottom up: each introduces only what it needs and relies only on earlier chapters. Every chapter uses the same blocks, in this order, and leaves out a block that has nothing to say.

| Block | What it gives you |
|---|---|
| **In one line** | What the chapter settles |
| **Builds on** | The earlier chapters you need; their *Carry forward* points reload the context |
| **Concept** | The idea in general terms, before any project choice. Findings from papers are explained, not assumed |
| **In this study** | What the study does and why, including what is still open |
| **Worked example** | One fixed recipe followed through every chapter, with pseudocode for what is computed. Pseudocode shows *what*, not *how* |
| **Watch out** | The warnings that matter for understanding, with pointers to the confound register (C, Chapter 14) and the evidence-gap register (G, Chapter 15) |
| **Further reading and carry forward** | Where to go deeper, and the few points later chapters assume you remember |

**Evidence tags.** A claim that does not come from the project owner carries one of the brief's tags:

| Tag | Meaning |
|---|---|
| [paper: X] | From a project paper; the key is listed in `literature/index.md` |
| [calc] | Own arithmetic |
| [measured] | Computed on project data |
| [estimate] | An estimate not computed on project data |
| [inference] | Reasoned, not tested |
| [speculation] | Speculation |
| [from memory] | From memory, to be verified |

**Notation** is plain ASCII, the same as in the brief: `mu`, `sigma`, `lambda`, `sqrt(x)`, `x^2`, `~` for "about". Brief sections are written S0 to S15; sections of papers are written Sec.

**The worked example recipe**, used in every chapter:

> **Recipe E**: depth 2, width 32, weight decay on, data tier 3 (12 financial assets), history 7 years.

---

## Chapter 0. Orientation

Synced: S0 d917ea65

**In one line.** The study measures how MLP design and data choices *behave* once training has run to its end. It does not ask which recipe wins.

**Concept.** Most model studies ask which configuration performs best. This one asks a different question: for every combination of choices in a fixed design space, what does the trained network look like? How well does it fit, how much does it change from seed to seed, does it overfit, and did it actually finish learning? Where in the design space do these properties stay the same, and where do they change? The answer is a set of *maps* over the design space, not a leaderboard.

Two terms frame the question:
- **Training saturation.** Training has run long enough that further training no longer changes the result in a measurable way.
- **Bias regime.** The remaining error comes mostly from the network being too simple or too constrained (underfitting), not from memorizing the training data (overfitting).

The study describes networks that are in both states, and how often, and where, networks fail to reach them.

**In this study**
- The task is a one-day-ahead price forecast for US stocks, from price windows processed by a discrete wavelet transform (Chapters 1 to 3).
- The unit of study is the full grid of 480 **recipes**. Each recipe is tuned on its own and then trained with 32 seeds.
- Every recipe is analyzed for training and test R^2, the generalization gap, and how much predictions vary across seeds, each with its spread across seeds.
- **Recipe status** (valid, failure-unsaturated, failure-overfit) is one categorical characteristic among the others.
- The main quantities are **regionalized**: recipes are grouped into connected regions with equivalent results, and no borders are drawn in advance (Chapter 12).
- Findings about the bias regime are read inside regions whose status is valid. Failure rates and data saturation are described over the whole population.
- The recipe space is fixed. Changing it in response to results is out of scope.
- The characterized recipes are meant for a later project, which must use the same training protocol (Chapter 5) for these characteristics to carry over.

**Worked example.** Recipe E is one of the 480 recipes. By Chapter 12 it will have 32 trained seeds, a status, and a region ID on each of four maps.

**Watch out.** The study characterizes; it does not rank. "Recipe A has a higher test R^2 than recipe B" is never a finding by itself here.

**Further reading and carry forward**
- RegNet (Radosavovic et al. 2020), Sec. 3.1: the case for studying *distributions* of models over a design space rather than the best model found [paper: RegNet].
- The output is characteristics of choices, shown as maps, not a ranking.
- The two target conditions are training saturation and the bias regime.
- The grid is fixed: 480 recipes, each run once, 32 seeds each.

---

## Chapter 1. The forecasting problem

Synced: S1 f1c89dff

**In one line.** Predict tomorrow's log closing price of a stock, and score the return that forecast implies against forecasting no change.

**Builds on.** Chapter 0.

**Concept.**
- **Log price and log return.** With closing price P(t), the one-step log return is log P(t+1) - log P(t). Log returns add up over time and do not depend on the price level.
- **Persistence.** The naive "no change" forecast: tomorrow's price equals today's. In return terms it predicts a return of zero.
- **R^2 against persistence.** With r the realized return and r_hat the forecast return, R^2 = 1 - sum(r - r_hat)^2 / sum(r^2). R^2 = 0 means no better than persistence, and R^2 < 0 means worse.
- **Why benchmark against zero rather than the historical mean.** Daily stock returns are dominated by noise. Gu, Kelly and Xiu use exactly this zero benchmark for individual stock returns, because the historical mean is so noisy that it lowers the bar; benchmarking against the mean instead raises every method's out-of-sample R^2 by roughly three percentage points [paper: Gu, Kelly and Xiu, Sec. 1.8, eq. 19 and fn. 34].

R^2 values near zero, or even below it, are therefore normal in this setting and not a sign of a bug.

**In this study**
- **Target.** The log closing price one trading day ahead, log P(t+1). One step only.
- **Forecast and implied return.** The network's forecast is a log price. Subtracting today's log price gives the return the forecast implies. How the network's output becomes the forecast is the subject of Chapter 2.
- **Loss.** Mean squared error between forecast and target. Because both sides share today's log price, the error of the forecast is exactly the error of the implied return, so training on prices and scoring on returns measure the same thing.
- **Comparison metric.** R^2 of the implied return against persistence, not the training loss.
- **Where R^2 is computed.** On the **test asset** only, on both the training side and the test side, with the persistence baseline computed on the same windows. The network still trains on every asset of its tier (Chapter 3); evaluating on one asset keeps training-set diversity out of the comparisons.

**Worked example**
```text
# one window of the test asset, last input date t
forecast       = network forecast of log P[t+1]          # a log price (Chapter 2)
implied_return = forecast - log P[t]
realized       = log P[t+1] - log P[t]

# forecast error == implied-return error, so the MSE loss is a return MSE
forecast - log P[t+1] == implied_return - realized

# R^2 against persistence on a set of test-asset windows W
R2(W) = 1 - sum_{w in W} (realized_w - implied_return_w)^2 / sum_{w in W} realized_w^2
```
Recipe E trains on 12 assets, but its R^2 is always computed on the test asset's windows.

**Watch out**
- Next-day returns carry little predictable signal: a linear model on the same window information scores below zero on the test window [measured]. A network that has learned nothing useful can still look calm and stable; Chapters 2 and 9 explain how such networks are recognized. -> **C1**
- Conclusions rest on one test asset and one test period. -> **C7**

**Further reading and carry forward**
- Gu, Kelly and Xiu 2020, *Empirical Asset Pricing via Machine Learning*, Sec. 1.8 (performance evaluation, the zero benchmark) [paper: Gu, Kelly and Xiu].
- The target is tomorrow's log price; the forecast's implied return is what gets scored.
- R^2 is measured against persistence, on the test asset only. Values near zero are normal.

---

## Chapter 2. From prices to inputs

Synced: S2 445750bc

**In one line.** Each 16-day window is normalized by its own mean and spread, turned into 15 wavelet details plus 2 volatility features, and given back its mean and spread as 2 extra inputs: 19 inputs. The network's output is mapped back with the same mean and spread.

**Builds on.** Chapter 1.

**Concept.**
- **Window.** One sample: a lookback of past log prices ending at the last input date t, together with its target at t+1.
- **Haar discrete wavelet transform (DWT).** Take neighbouring pairs of values (a, b). Keep their scaled sum (a+b)/sqrt(2) as the "approximation" and their scaled difference (a-b)/sqrt(2) as the "detail", then repeat on the approximations only. For 16 values this gives details at 4 scales (8 + 4 + 2 + 1 = 15 numbers) and 1 final approximation. The sqrt(2) scaling makes the transform *orthonormal*: it preserves sums of squares. Details describe movement at each time scale; the approximation carries the overall level [paper: Rhif, Sec. 2.3]. Computed inside each window, the DWT uses only that window's prices, so it never sees the future.
- **Non-stationarity and extrapolation.** A stock's price level drifts over years. A network that takes the level as a raw input meets, on the test window, levels it never saw in training. ReLU-type MLPs become linear outside the range of their training data, so such inputs get a linear, usually wrong, response [paper: Xu, Theorem 1, Fig. 1].
- **Window normalization.** Liu et al. subtract each window's mean and divide by its standard deviation before the model, then apply the reverse to the model's output: forecast = mean + std x output [paper: Liu, Sec. 3.1, eqs. 1 and 2]. Every window the model sees then has the same mean and spread, and the model becomes *equivariant* to shifting and scaling a series: shift the prices, and the forecast shifts with them.
- **Over-stationarization.** Normalization has a cost. Two windows that differ only by a constant shift become identical inputs, so the model can no longer tell them apart. Liu et al. call this over-stationarization and restore the lost information by feeding the removed statistics back into the model; doing so improved forecasts, most on the least stationary data, including daily exchange rates [paper: Liu, Sec. 3.2, Table 6].

**In this study**
- **Lookback:** 16 trading days of log closing prices, p_1 ... p_16, with p_16 = log P(t).
- **Window statistics:** mu, the mean of the 16 log prices, and sigma, their population standard deviation (dividing by n).
- **Normalized window:** (p - mu)/sigma, with mean 0 and standard deviation 1.
- **Haar DWT of the normalized window:** 15 details. The approximation of a centred window is 0, so it is not an input; the level it would have carried comes back through mu.
- **Volatility features:** the population standard deviation of the normalized prices over the last 4 and 8 days. Over 16 days it is 1 by construction, so it is not an input. These features spare small networks a nonlinearity they may not approximate; they add no information.
- **Re-injected statistics:** mu and log sigma are appended to the input, following Liu et al.'s remedy for over-stationarization. Liu et al. re-inject inside a Transformer; appending to an MLP's input is this study's adaptation [inference].
- **Input dimension:** 15 + 2 + 2 = 19, with no further scaling.
- **Output:** the network predicts the target in window units, (log P(t+1) - mu)/sigma, and the forecast is mu + sigma x output.

Why the output is mapped this way: an output of zero forecasts the window mean, which is a poor forecast for a trending or moving price. To reproduce persistence, the network must output (p_16 - mu)/sigma, a linear function of the normalized details, so it has to use its input. A network cannot score well by switching itself off, for example when weight decay drives its weights toward zero.

- **Still open:** a window whose 16 closes are all equal has sigma = 0 and cannot be normalized. The working reading drops it, like a window with a missing day; the current data have none [measured].

**Worked example**
```text
# p: 16 log closing prices of one test-asset window, oldest -> newest, ending at date t
mu    = mean(p)                          # e.g. ~5.0 for a stock near $150
sigma = pstd(p)                          # e.g. ~0.02 in a calm month
q     = (p - mu) / sigma                 # normalized window: mean 0, std 1

a = q;  details = []
while length(a) > 1:
    pairs    = group a into consecutive pairs (a_i, b_i)
    details += [(a_i - b_i) / sqrt(2) for each pair]      # finest level first
    a        = [(a_i + b_i) / sqrt(2) for each pair]
# details: 15 values; a == 0 because q is centred

x = concat(details, [pstd(q[-4:]), pstd(q[-8:])], [mu, log(sigma)])   # 19 inputs

output   = network(x)                    # in window units
forecast = mu + sigma * output           # a log price
persistence_output = (q[-1])             # = (p_16 - mu)/sigma: what "no change" requires
```
Because the transform is linear, the details of the normalized window are the details of the raw window divided by sigma [calc].

**Watch out**
- **mu is the one input that still drifts.** Every other input is unchanged when the whole window is shifted, but mu is the level itself. On the test window, every JPM window's mu lies above its training range at all three history lengths [measured]. The forecast takes its level from the de-normalization, so the extrapolation risk enters only through this one input. -> **C2**
- **Adjusted prices.** Prices are back-adjusted for dividends and splits, so a window's mu partly reflects corporate actions after its date [inference]. Differences within a window, and so every other input, are unaffected. -> **C2**
- **mu and log sigma are not centred** (about 1 to 7 and about -6 to 0 [measured]), while the other inputs are of unit scale. -> **C2**
- **The volatility features are redundant by construction.** Under the orthonormal convention, the population standard deviation over m days equals sqrt(sum of squared details in the matching block / m) [calc]. They add a convenient nonlinearity and no information, which also means reliance on particular DWT bands cannot be cleanly attributed. -> **C2**
- Gap: this input representation has no direct backing for return forecasting with MLPs. -> **G1**

**Further reading and carry forward**
- Liu et al. 2022, *Non-stationary Transformers*: Sec. 3.1 (eqs. 1 and 2, normalization and de-normalization), Sec. 3.2 (over-stationarization), Table 6 (re-injecting the statistics) [paper: Liu].
- Xu et al. 2021, *How Neural Networks Extrapolate*, Sec. 3.1, Theorem 1 and Fig. 1 [paper: Xu].
- Rhif et al. 2019, *Wavelet Transform Application for/in Non-Stationary Time-Series Analysis: A Review*, Sec. 2.3 [paper: Rhif]. Tawakuli et al. 2024 (preprocessing survey) as general background.
- 19 inputs: 15 normalized Haar details, 2 normalized volatility features, mu and log sigma.
- The network predicts in window units; the forecast is mu + sigma x output. A zero output forecasts the window mean, so persistence needs the input.
- mu is the only input that can leave its training range.

---

## Chapter 3. The data universe

Synced: S3 e077cebc

**In one line.** Which assets and dates a network sees, how time is split into tuning, training and test periods, and why no label crosses a split.

**Builds on.** Chapters 1 and 2.

**Concept.**
- **Pooling.** A single network trains on windows from several assets at once, which gives it more data than one asset's history. Gu, Kelly and Xiu pool across stocks in the same way [paper: Gu, Kelly and Xiu].
- **Splitting by time.** For forecasting, training data must come before validation data, which must come before test data. Splitting at random dates would let the network learn from the future.
- **Label leakage.** Leakage happens when a training sample's target lies inside a held-out period. *Input overlap* is different and harmless: a held-out window's inputs reaching back into training dates is not leakage, because no target crosses the boundary.

**In this study**

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

- The financial tiers are nested (tier 1 within 2 within 3 within 4), and tier 5's four financials are tier 2's four. Tiers 4 and 5 have the same size, so comparing them isolates *composition*.
- The asset draw, including the test asset, is fixed for the whole study; it is not redrawn per seed, which lets seeds be trained together.
- "Consumer" means the consumer discretionary sector in the Global Industry Classification Standard (GICS). Days are trading days.

*Time*

```
history start -- tuning-stage training --|tuning cutoff|-- validation window --|origin|-- test window
              +---------------- final-stage training data -------------------+
```

- **History length:** 3, 7 or 18 years, counted back from the origin. History includes the validation window, so it equals the whole final-stage training data.
- **Tuning stage:** trains on the first 80% of the final-stage training data and tunes hyperparameters on the last 20%, the **validation window**.
- **Final stage:** retrains with the tuned values on all of it, so training ends at the origin. The **test window**, the last 2 years of data, follows immediately; merging the validation window keeps the test window close to the end of training.
- Splits are by calendar date, across all assets at once.

*Boundaries*
- A **cutoff** (the tuning cutoff or the origin) is the last *target* date of a stage. A window belongs to a stage if its target date is on or before that stage's cutoff, so its last input date is at least one trading day earlier. No label crosses a cutoff.
- There is no embargo at either cutoff. Input overlap at a cutoff is accepted, since it is not leakage.
- **Test-asset training windows** are the test asset's windows inside the final-stage training data; they are the training side of every comparison (Chapter 1).

*Sizes* [calc]
- About 250 windows per asset per year; data per recipe runs from 1 asset x 3 years to 24 assets x 18 years, a factor of 144, roughly 750 to 108,000 windows.
- Test-asset training windows: about 750 (3 years) to about 4,500 (18 years). Validation window: about 0.6, 1.4 or 3.6 years (about 150 windows at tier 1 with 3 years). Test window: about 500 windows.

**Worked example**
```text
# Recipe E: tier 3 (12 assets), history 7 years
N_final        ~ 12 assets x 7 years x 250 ~ 21,000 windows
N_tuning_train ~ 0.8 x N_final             ~ 16,800 windows
validation     ~ 1.4 years                 (~350 test-asset windows)
test-asset training windows ~ 7 x 250      ~ 1,750
test window                                ~ 500 test-asset windows

# stage membership of a window, applied to every asset
stage(window) = "tuning-train" if target_date <= tuning_cutoff
                "validation"   if tuning_cutoff < target_date <= origin
                "test"         if target_date > origin
```
All figures [calc].

**Watch out**
- **History length is not a clean "more data" knob.** Longer histories also contain different market regimes (18 years reaches back to 2008), less noisy tuning, more training-side windows, and a larger time distance to the test window. -> **C4**
- **Survivorship.** The pool consists of current index members with today's sector labels; dual share classes are not deduplicated. -> **C5**, **G17**
- **One draw, one test asset, one test period.** Overlapping windows are strongly autocorrelated, so the effective sample is far below the window count. -> **C7**, **G17**
- **Batch size follows data size** (Chapter 5), so tier and history also change gradient noise. -> **C6**
- Gap: tier design, a single test asset, the split ratios and no embargo lack direct backing. -> **G2**

**Further reading and carry forward**
- Gu, Kelly and Xiu 2020 (pooling across stocks; their sample-splitting scheme) [paper: Gu, Kelly and Xiu].
- Tiers 1 to 5 run from 1 to 24 assets; 1 to 4 are nested financials, 5 is diverse at tier 4's size.
- History is 3, 7 or 18 years. Tuning uses an 80/20 split; the final stage merges the validation window and tests on the next 2 years.
- Splits are by target date, so no label crosses a cutoff.

---

## Chapter 4. The network

Synced: S4 812555c9

**In one line.** A plain MLP with 1 to 8 hidden layers of 8 to 64 units, leaky ReLU, and the default PyTorch initialization, which decides whether training starts "lazy" or "rich".

**Builds on.** Chapters 2 and 3.

**Concept.**
- **MLP.** A stack of fully connected layers, each followed by a nonlinearity. *Depth* is the number of hidden layers; *width* is the number of units per hidden layer. The last layer is linear and produces one number.
- **Leaky ReLU.** max(x, 0.01 x). It behaves like ReLU but keeps a small slope for negative inputs, so units never go completely silent.
- **Initialization scale and the lazy vs. rich regimes.**
  - *Lazy* training: the network barely moves from its initial weights and behaves almost like a fixed linear model on random features.
  - *Rich* (feature-learning) training: the internal representation changes substantially during training.
  - Chizat et al. show that lazy training occurs when the model's scale is large relative to how far it needs to move. Scaling up the output, or the initial weights of homogeneous networks, pushes training toward lazy [paper: Chizat, Sec. 1.2, eq. 1, Fig. 1].
  - Kunin et al. add that the overall scale is not all that matters. The *relative* scale between layers matters too, and in nonlinear networks an initialization where early layers learn faster can speed up rich learning [paper: Kunin, Fig. 1, Sec. 5].
- **Width and seed-to-seed variance.** Neal et al. split prediction variance into a part due to initialization and a part due to sampling the training data. The part due to initialization is large when the network has fewer parameters than training points; once it has more, it falls as width grows [paper: Neal, Sec. 2.3 eq. 4, Sec. 3.4 Fig. 2].

**In this study**
- A plain MLP: the 19 inputs of Chapter 2, `depth` hidden layers of one constant `width`, and one linear output, the prediction in window units. Depth and width are recipe factors (Chapter 6).
- Activation: leaky ReLU with the PyTorch default slope of 0.01, fixed and not tuned. No normalization layers, no skip connections, no dropout.
- Initialization: the PyTorch default for linear layers, fixed and observed. Weights and biases are uniform in +/-1/sqrt(fan_in), so the weight variance is 1/(3 fan_in) [from memory]. Fixed small initialization, width-aware scaling, and initialization scale as a factor were considered and not adopted.
- The evaluation weights (Chapter 5) start at the initialization.

**Worked example**
```text
# Recipe E: depth 2, width 32, 19 inputs, 1 output
params = (19*32 + 32) + (32*32 + 32) + (32*1 + 1) = 640 + 1056 + 33 = 1,729
# ~1 parameter per 12 windows (N_final ~ 21,000): fewer parameters than points
init: W, b ~ Uniform(-1/sqrt(fan_in), +1/sqrt(fan_in))  for every linear layer
```
At the extremes of the grid [calc]: depth 1, width 8 has about 170 parameters against up to 108,000 windows; depth 8, width 64 has about 30,000 parameters against as few as 750 windows.

**Watch out**
- The PyTorch default is *not* He initialization; He initialization uses variance 2/fan_in, about 6 times larger [from memory].
- Input scale acts like an extra initialization scale and can shift where a recipe sits between lazy and rich. Most inputs are of unit scale, but mu and log sigma are not centred (Chapter 2) [inference]. -> **C2**
- Widths 8 and 16 are narrower than the 19 inputs (bottlenecks); 32 and 64 are wider (expansions). The grid crosses the boundary between fewer and more parameters than training points, so seed variance moves with width and data size for reasons other than overfitting. -> **C9**, **C6**
- Gap: the initialization lacks direct backing. -> **G7**

**Further reading and carry forward**
- Chizat, Oyallon and Bach, *On Lazy Training in Differentiable Programming*, Sec. 1.2 (eq. 1) and Fig. 1 [paper: Chizat].
- Kunin et al. 2024, *Get Rich Quick*, Fig. 1 and Sec. 5 [paper: Kunin].
- Neal et al. 2018, *A Modern Take on the Bias-Variance Tradeoff*, Sec. 2.3 and Sec. 3.4 [paper: Neal].
- Gu, Kelly and Xiu 2020 find that shallow networks outperform deep ones in return prediction [paper: Gu, Kelly and Xiu].
- A plain MLP with leaky ReLU and the default PyTorch initialization (not He).
- Initialization scale, and input scale with it, moves training between lazy and rich.
- Seed variance from initialization depends on width and on parameters versus data, not only on overfitting.

---

## Chapter 5. How it learns

Synced: S5 b66770d0

**In one line.** Mean squared error minimized by schedule-free AdamW: a 5% warmup, then a constant learning rate, with an internal running average used as the model, for 500 epochs of exactly 25 steps each.

**Builds on.** Chapters 1 and 4.

**Concept.**
- **Adam.** Adam scales each parameter's step by running estimates of the gradient's mean and squared magnitude (the latter at rate beta2). A small eps in the denominator prevents division by zero. **AdamW** adds *decoupled* weight decay: each step shrinks the weights toward zero by lr x lambda, separately from the gradient step.
- **Why schedules exist.** At a constant learning rate, the iterate keeps bouncing around in gradient noise. Decaying the learning rate toward the end removes that noise and usually gives a better final model, but a schedule needs the stopping time in advance.
- **Schedule-free AdamW** (Defazio et al. 2024) removes the need to know the stopping time. It keeps three sequences [paper: Defazio, Sec. 2, Alg. 1]:
  - **z**, the base iterate, where the Adam step is taken;
  - **y** = (1 - beta) z + beta x, the point where gradients are evaluated;
  - **x**, a weighted running average of z. **x is the model you evaluate.**

  The averaging weights are proportional to the squared learning rate, c_(t+1) = gamma_t^2 / sum(gamma_i^2), so steps taken at a small learning rate get little weight [paper: Defazio, eq. 23]. The average x behaves like a model trained with a linear decay ending *at the current step*, so the method applies a schedule implicitly and can be read at any time [paper: Defazio, Sec. 2.3, Fig. 4]. Defazio et al. also show that x closely tracks the best loss reachable at each point in training [paper: Defazio, Fig. 1].
- **Practical findings from Defazio et al.** Warmup is still required (Sec. 4.4). beta = 0.9 works on all their problems but one, and its best value does not change with training length (Sec. 4.1, Sec. 5). The main tuning need is the learning rate and weight decay, whose best values differ from those under a schedule; schedule-free learning rates ran about 1 to 10 times higher (Sec. 4.1, Sec. 6, App. G). The best learning rate is stable across training lengths, except for short runs (Sec. 5) [paper: Defazio].
- **Batch size and learning rate.**
  - Shallue et al.: as batch size grows, the steps needed to reach a goal first fall in proportion, then show diminishing returns, then stop improving. The best metaparameters follow no simple rule in batch size, so each batch size needs its own tuning. Their study covers SGD with and without momentum, not Adam [paper: Shallue, Sec. 4.1, Fig. 1].
  - Li et al.: for Adam-style optimizers, the best learning rate *rises then falls* as batch size grows, peaking near a noise scale that shifts to larger batches as training progresses [paper: Li et al., Fig. 1, eq. 4, Sec. 6]. Whether this carries over to schedule-free AdamW is [inference].

**In this study**
- Loss: MSE on the forecast (Chapter 1).
- Optimizer: schedule-free AdamW with beta = 0.9, beta2 = 0.999, eps = 1e-8 [from memory: implementation default], and averaging weights proportional to the squared learning rate.
- **Weight decay**, when a recipe uses it, applies to all weight matrices, never to biases, at the gradient point y: z <- z - gamma_t lambda y [paper: Defazio, Alg. 1, Sec. 4.4].
- **Warmup:** linear over the first 5% of steps (625 steps, 25 epochs [calc]), as part of the optimizer. Defazio et al. used 5% in their deep-learning experiments and 2% in their AlgoPerf entry; the longer value was chosen because the smallest batches raise the risk of early instability [inference]. Warmup steps carry about 1.7% of the final averaging weight [calc].
- **No schedule after warmup:** the base learning rate is constant from then on.
- **Training length:** 500 epochs, fixed by heuristic, not a factor and not tuned. A fixed length also means no epoch count has to be chosen before the final-stage retrain.
- **Batch size follows data size:** 25 steps per epoch, so the batch is B ~ N/25 and every recipe gets about 12,500 steps. Each epoch splits the N windows into 25 near-equal batches. In the final stage B runs from about 30 to about 4,300 [calc].
- **Implementation:** a batched copy of the reference implementation (facebookresearch/schedule_free), checked against it member by member. Like the reference, it applies Adam's bias correction in the denominator, sqrt(v / (1 - beta2^(k+1))) + eps, instead of folding it into the learning rate as the paper describes; the averaging weights then use the warmup-scaled learning rate only.
- **Every measurement reads the evaluation weights x.** The later project must use schedule-free AdamW and read x for these characteristics to apply.

**Worked example**
```text
# Recipe E, final stage
N = 21,000;  K = 25;  B ~ N / K = 840;  epochs = 500;  steps = 12,500
warmup_steps = 0.05 x 12,500 = 625      # gamma ramps linearly from ~0 to lr

per step k:
    y  = (1 - beta) * z + beta * x
    g  = grad of MSE(forecast(y), target) on the minibatch       # gradient at y
    v  = beta2 * v + (1 - beta2) * g^2
    d  = g / (sqrt(v / (1 - beta2^(k+1))) + eps)                   # bias correction in the denominator
    d += lambda * y        (weight matrices only; Recipe E uses weight decay)
    z  = z - gamma_k * d
    c  = gamma_k^2 / sum_{i<=k} gamma_i^2
    x  = (1 - c) * x + c * z                                       # evaluation weights
after each epoch: evaluate x on the test-asset training windows and on the test window
```

**Watch out**
- Where the bias correction sits may change how weight decay acts in early steps [paper: Defazio, Sec. 4.4].
- The loss is on the forecast, so gradients scale with sigma times the forecast error, both small, while eps is fixed; eps may then dominate Adam's denominator [paper: Choi, for Adam; inference for schedule-free AdamW]. -> **C2**, **G10**
- **Batch size is tied to data size.** Tier and history set N, and B ~ N/25, so gradient noise changes systematically along the data axes and cannot be separated from data-size effects. Comparisons within one tier x history cell, and tier 4 vs. 5, are not affected. -> **C6**
- Gap: 500 epochs and 25 steps per epoch lack direct backing. -> **G6**

**Further reading and carry forward**
- Defazio et al. 2024, *The Road Less Scheduled*: Sec. 2 and Alg. 1 (method), Sec. 2.3 and Fig. 4 (implicit schedule), Fig. 1 (tracking the loss frontier), Sec. 4.4 (warmup, weight decay, batch norm), Sec. 5 (beta and training length), App. G (settings) [paper: Defazio].
- Shallue et al. 2018, Sec. 4.1 and Fig. 1 [paper: Shallue]; Li et al. 2024, *Surge Phenomenon*, Fig. 1 [paper: Li et al.].
- Schedule-free AdamW tracks z, y and x; **x is the model**, and it behaves as if the learning rate decayed to "now".
- 5% warmup, then a constant learning rate; 500 epochs x 25 steps = 12,500 steps for every recipe.
- B ~ N/25, so batch size and data size move together (C6).

---

## Chapter 6. The recipe space

Synced: S6 e19269a4

**In one line.** Five factors (depth, width, regularization, data tier, history) crossed into 480 recipes; everything else is fixed or out of scope.

**Builds on.** Chapters 3 to 5.

**Concept.**
- **Factor, option, recipe.** A factor is a design dimension, an option is one of its values, and a recipe is one option per factor.
- **Scientific, nuisance and fixed hyperparameters.** The Tuning Playbook sorts settings by role [paper: Playbook]: *scientific* ones are what you want to learn about; *nuisance* ones must be optimized away for each scientific setting so that comparisons are fair; *fixed* ones are held constant, which limits how far the conclusions reach.
- **Design spaces.** RegNet argues that comparing *distributions* of models from a design space is more robust and informative than comparing the best models found [paper: RegNet, Sec. 3.1].

**In this study**

| Factor (all scientific) | Options | Introduced in |
|---|---|---|
| Depth (hidden layers) | 1, 2, 4, 8 | Chapter 4 |
| Width (constant hidden width) | 8, 16, 32, 64 | Chapter 4 |
| Regularization | none; weight decay | Chapter 5 |
| Data tier | 1, 4, 12, 24, 24 (diverse) assets | Chapter 3 |
| History | 3, 7, 18 years | Chapter 3 |

- **Fixed:** MSE loss and a one-step target (Chapter 1), leaky ReLU (Chapter 4), schedule-free AdamW (Chapter 5). Findings hold only under these.
- **Not in the space:** a horizon factor (multi-step targets), early dropout, regularization "both" (it would only add the interaction), and an MAE loss.
- **Out of scope:** other losses, other optimizers, other activations, dropout in any form, L1 regularization, normalization layers, gradient clipping, network shape, skip connections, and explicit learning-rate schedules. The window normalization of Chapter 2 is not a normalization layer: it has no parameters and belongs to the input and output mapping.
- **Size:** 4 x 4 x 2 x 5 x 3 = **480** recipes [calc]. Every recipe is run once, and the space is not changed based on results.

**Worked example**
```text
recipe_E = {depth: 2, width: 32, regularization: "weight_decay", tier: 3, history: 7}
grid     = product(depth[4], width[4], regularization[2], tier[5], history[3])   # 480 recipes
```

**Watch out**
- Findings do not extend to other losses, optimizers, activations or horizons.
- Depth 8 x width 64 at tier 1 x 3 years means about 30,000 parameters against about 750 correlated windows, so many overfit failures are expected [speculation]. Depths 4 and 8 straddle the hypothesized overfit boundary at depth 6; if it exists, it shows as a jump in failure rate, not at a specific depth. -> **C9**
- Gaps: the architecture ranges (**G4**) and the role of weight decay in the bias regime under schedule-free AdamW (**G5**).

**Further reading and carry forward**
- Tuning Playbook, "Identifying scientific, nuisance, and fixed hyperparameters" [paper: Playbook]; RegNet, Sec. 3.1 [paper: RegNet].
- Five scientific factors, 480 recipes, all run once. The space is closed.
- Anything outside the fixed set (MSE, schedule-free AdamW, leaky ReLU, one step ahead) is out of scope.

---

## Chapter 7. Fair footing: hyperparameter tuning

Synced: S7 84c0fc28

**In one line.** Every recipe tunes its learning rate (and lambda, if it uses weight decay) with the same quasi-random search, 16 trials of 4 seeds each, over ranges shared by all recipes; a small calibration run sets those ranges.

**Builds on.** Chapters 3, 5 and 6.

**Concept.**
- **Why tune each recipe.** If one recipe gets a good learning rate and another a poor one, their difference reflects tuning luck, not the design choice. Nuisance hyperparameters must be optimized for each scientific setting [paper: Playbook, "Creating a set of studies"].
- **Quasi-random search.** Instead of random points, use a low-discrepancy sequence such as scrambled Halton, which covers the space evenly. It does not adapt to results, so all trials can be drawn up front and trained together. The Playbook prefers it during exploration [paper: Playbook].
- **Search spaces matter.** Choi et al. show that optimizer comparisons can flip when the search space changes; the search space may be the single most important factor behind published rankings [paper: Choi, abstract and Sec. 1].
- **Edge hits.** If the best value lands at the edge of the range, the true best may lie outside it. The Playbook lists this as a standard check [paper: Playbook; Choi].
- **Paired trials.** Giving all trials the same initialization and data order reduces the noise in comparisons *between* trials [paper: Bouthillier].
- **The winner's curse.** Picking the best of many noisy scores favours the candidates whose scores vary most, because an unstable setting that happened to land well beats a stable one. The more trials and the noisier each score, the stronger the effect.

**In this study**
- Tuning happens per recipe, in the tuning stage. It is a means, not the subject of study.
- **Tuned:** the learning rate, and lambda when weight decay is on. beta, beta2, eps and warmup stay fixed. This follows Defazio et al.'s practical need [paper: Defazio].
- **Method:** 16 scrambled Halton points with a fixed seed, mapped log-uniformly onto the ranges, the same points for every recipe of the same dimension. The points are drawn with scipy's Halton generator, the engine behind Optuna's quasi-Monte Carlo sampler, because Optuna's sampler draws the first trial of a batch requested in advance at random; Optuna stores the trials. Every trial runs the full 500 epochs, with no pruning.
- **Four seeds per trial:** each trial is trained with 4 seeds, and its score is the mean over them. This blunts the winner's curse: a setting that blows up on some seeds and gets lucky on one no longer wins. With a single seed per trial, seed noise was larger than the real differences between learning rates [measured on a project run].
- **Batched trials:** a recipe's 16 x 4 = 64 (trial, seed) pairs train as one batched model, each member with its own hyperparameters, weights and optimizer state. Seed j has the same initialization and data order in every trial, so comparisons between trials are paired seed by seed. The averaging weights do not depend on the learning-rate scale (it cancels), so they are the same for every member [calc].
- **Objective (maximized):** validation R^2 of the implied return, using the evaluation weights, on the test asset, averaged over the last 25 epochs and then over the trial's 4 seeds.
- **Calibration run:** 24 recipes at the extremes (Chapter 13), with wide ranges: learning rate 1e-5 to 1e-1 and lambda 1e-5 to 1, log-uniform [from memory]. The upper end leaves room for schedule-free learning rates, which ran about 1 to 10 times higher than usual [paper: Defazio, Sec. 4.1, App. G].
- **Main-run ranges:** one shared range per tuned hyperparameter, for *all* recipes. The best learning rate moves with batch size [paper: Li et al.; transfer is inference] and with width [from memory]. Ranges per tier x history cell would follow batch size only, and each data cell would get its own search space, so the data-size factor would carry tuning quality. -> **C6**
- **Range rule:** each main-run range spans from the lowest to the highest best value found in calibration, widened by 0.75 decade on each side.
- **Edge hits** are flagged, reported and marked on the lattices (Chapter 12), not re-tuned, since re-tuning would break the single protocol.
- **Tuning-to-final transfer:** tuned values are reused unchanged in the final stage, at the same step count and an about 1.25 times larger batch [calc]. No project paper covers this transfer for schedule-free AdamW. -> **G8**
- **Infeasible trials** (any seed diverged or NaN) count as failed with the worst objective; the infeasible fraction per recipe is reported. lr x lambda is reported alongside lambda.

**Worked example**
```text
# Recipe E, tuning stage: N ~ 16,800, B ~ 672
space    = {lr: logUniform(main_lr_range), lambda: logUniform(main_wd_range)}   # 2-D: weight decay is on
trials   = Halton(scrambled, seed=fixed).sample(n=16) mapped onto space          # drawn up front
model    = batched(16 x 4 members; member (i, j) uses trials[i] and seed j's init and data order)
train model for 500 epochs x 25 steps
score_ij = mean(validation_R2_testasset(x_ij, epochs 476..500))
score_i  = mean_j score_ij          # -inf if any seed of trial i diverged
best     = argmax_i score_i
edge_hit = best value within one trial spacing of either range end, per tuned hyperparameter
infeasible_fraction = (# trials with a diverged or NaN seed) / 16
# final stage: reuse best (lr, lambda) unchanged; N ~ 21,000, B ~ 840 (~1.25x larger)
```
Recipe E's data cell (tier 3 x 7 years) is one of the three calibration cells. Its capacity (2, 32) is not a calibration corner, so Recipe E itself is not calibrated.

**Watch out**
- **Unequal tuning effort.** With 16 trials in every recipe, weight-decay recipes spread their trials over two dimensions while no-decay recipes use one. If the best learning rate depends on lambda, weight-decay recipes are tuned less precisely [inference].
- **Calibration at the range edge.** A calibration best value at an end of the wide range puts the main-run range beyond what calibration tested, by the margin.
- **Pairing may fade.** A one-bit change at initialization produces about as much variability as any other source [paper: Summers, Table 3], and trials differ in hyperparameters from the first step [inference].
- Four seeds halve the seed noise of the objective but do not remove it; it is largest at tier 1 with 3 years, with about 150 validation windows. -> **C4**
- Batched trials cost little memory, but compute still grows with the number of members once the hardware is saturated [inference].
- Gaps: tuning-to-final transfer (**G8**); objective, ranges, trial and seed counts (**G9**).

**Further reading and carry forward**
- Tuning Playbook: "Creating a set of studies"; "Why use quasi-random search"; "Extracting insight from experimental results" (edge check) [paper: Playbook].
- Choi et al. 2019, *On Empirical Comparisons of Optimizers for Deep Learning*, abstract and Sec. 1 [paper: Choi].
- 16 quasi-random trials x 4 seeds per recipe over ranges shared by all recipes, set by a 24-recipe calibration run plus a 0.75-decade margin.
- The objective is validation R^2 on the test asset, averaged over the last 25 epochs and over 4 seeds. Edge hits are flagged, not fixed.

---

## Chapter 8. Randomness and judging differences

Synced: S8 af2c037d

**In one line.** Each recipe is trained with 32 independent seeds, and differences between recipes are judged by *equivalence* tests with margins fixed by alpha, beta and n alone.

**Builds on.** Chapters 5 and 7.

**Concept.**
- **Sources of run-to-run variation.**
  - Bouthillier et al. measured each source separately. Resampling the data (bootstrap) is by far the largest; initialization is generally under half of it, on a par with data order [paper: Bouthillier, Fig. 1]. They recommend about 29 runs for their comparison test [paper: Bouthillier, App. C.3].
  - Summers and Dinneen find that every source of nondeterminism they studied has a similar effect on variability. Even a one-bit change in one initial weight produces about as much, because training is unstable [paper: Summers, Sec. 3, Table 3].
- **Equivalence testing.** A standard test can show that two things *differ*. It cannot show that they are the same, since a non-significant result may simply lack power. An equivalence test turns the question around: pick a margin Delta, then test whether you can *reject* that the difference is at least Delta. Lakens describes the "two one-sided tests" (TOST) procedure, one test against each bound; if both reject, the difference lies within +/-Delta, which at alpha = 0.05 is the same as the 90% confidence interval lying inside the bounds [paper: Lakens]. When no theory sets Delta, a defensible choice is the smallest effect the available sample can detect [paper: Lakens, eq. 5, solved for the bound].

**In this study**
- Every metric's variance is accounted for, using **32 seeds** per recipe. The relative error of a seed standard deviation is about 13% at 32 seeds [calc].
- **What a seed is.** One seed sets both initialization and data order; bundling them is supported by Summers and Dinneen [paper: Summers]. Each seed draws from its own random stream, independent of which other seeds share its batch, and streams are independent *across* recipes: seed i of one recipe shares nothing with seed i of another, so comparisons between recipes are unpaired. In tuning, by contrast, seed j of a recipe is shared by all of its trials (Chapter 7).
- Seeds run batched on the same data; each seed shuffles its own minibatches.
- **Error rates:** alpha = 0.05 and beta = 0.2, marked provisional by the owner.
- **Margins**, fixed for every comparison and derived from alpha, beta and n only:
  - **delta_mu** = (t_(1-alpha) + t_(1-beta/2)) x sqrt(2/n), in units of the seed standard deviation, with 2(n - 1) degrees of freedom. At n = 32, delta_mu ~ 0.74 sigma [calc; paper: Lakens, eq. 5].
  - **delta_sigma** = (z_(1-alpha) + z_(1-beta/2)) / sqrt(n - 1), on a log ratio of standard deviations. At n = 32, delta_sigma ~ 0.53, a ratio of about 1.7 [calc].
- **Tests decide merges only.** Hypothesis tests are used only inside regionalization (Chapter 12); no test result is reported as a finding. Status (Chapter 10) compares point estimates with fixed thresholds, without a test.

**Worked example**
```text
for s in 1..32:
    rng_s = independent_stream(recipe_E, s)     # unrelated to seed s of any other recipe
    init_s, order_s = draw(rng_s)               # one seed sets both
train all 32 seeds as one batched model on the same data, each with its own shuffle
# per quantity q: mean_q = mean over seeds, sd_q = std over seeds
```

**Watch out**
- Reproducibility holds for the *distribution* across seeds, not for individual seeds. A different batching of seeds can change floating-point order, and under unstable training any such change makes individual seeds diverge [paper: Summers; the batching mechanism is inference].
- The seed variance measured here covers initialization and data order on one fixed dataset. Variance from sampling the data, the largest source in Bouthillier et al., is not measured. -> **C7**

**Further reading and carry forward**
- Bouthillier et al. 2021, *Accounting for Variance in Machine Learning Benchmarks*, Fig. 1 and App. C.3 [paper: Bouthillier].
- Summers and Dinneen 2021, *Nondeterminism and Instability in Neural Network Optimization*, Sec. 3 and Table 3 [paper: Summers].
- Lakens 2017, *Equivalence Tests: A Practical Primer*, the TOST section and eq. 5 [paper: Lakens].
- 32 independent seeds per recipe; a seed sets initialization and data order; comparisons between recipes are unpaired.
- delta_mu ~ 0.74 sigma for means and delta_sigma ~ 0.53 for log ratios of standard deviations, both fixed in advance.
- Tests decide merges only; they are never findings.

---

## Chapter 9. What is measured

Synced: S9 dae887e5

**In one line.** Per recipe and per seed: training and test R^2, the gap between them, how much forecasts vary across seeds (T1), whether the curves are flat (V1), and two diagnostics, all read from the evaluation weights at the end of the final stage.

**Builds on.** Chapters 1, 3, 5 and 8.

**Concept.**
- **Training and test curves.** A metric recorded after every epoch, on training and on held-out data.
- **Generalization gap.** Training performance minus test performance.
- **Bias-variance decomposition across seeds.** For a given input, train many seeds. The *variance* term is how much their forecasts disagree; the *bias* term is the error of their average forecast. Neal et al. use the law of total variance to separate variance due to initialization (seeds on fixed data) from variance due to sampling the training data [paper: Neal, Sec. 2.3, eq. 4].
- **Flatness under schedule-free AdamW.** The evaluation weights x track the frontier of loss against training time, so a flat curve for x means that training longer, even with an ideal decay, would not help [paper: Defazio, Fig. 1, Sec. 2.3].

**In this study**

Every quantity uses the one-step forecast, the final stage and the evaluation weights, and is reported for every recipe together with its standard deviation across seeds. The **training curve** is R^2 per epoch on the test-asset training windows; the **test curve** is R^2 per epoch on the test window.

| Quantity | What it is | Read on | Feeds status? |
|---|---|---|---|
| **V1a slope** | Training curve: mean of the last 25 epochs minus mean of the preceding 25, in R^2 units | Test-asset training windows | Yes |
| **V1b slope** | The same on the test curve | Test window | Yes |
| **T1** | Mean over windows of the variance of the forecast across seeds, on each side, and the log ratio 0.5 log(T1_test / T1_train), positive in the overfitting direction like the gap | Both sides | Yes (the log ratio) |
| **Training R^2** | Mean of the last 25 epochs of the training curve | Test-asset training windows | No |
| **Test R^2** | Mean of the last 25 epochs of the test curve | Test window | Only through V1b |
| **Generalization gap** | Training R^2 minus test R^2, per seed | Both | No |
| **Prediction spread** | Standard deviation over windows of the network output in window units, per seed | Test window | No |
| **Relative weight change** | How far the weights moved from their initialization, relative to its size | Parameters | No |

- T1 is the variance term of the decomposition across seeds, the counterpart of Neal et al.'s variance due to initialization, here also including data order [paper: Neal]. Seeds share the target, so T1 equals the variance of the forecast error. Its uncertainty is a jackknife over seeds (Chapter 12).
- **Why prediction spread is read on the network output.** A network that ignores its input outputs a constant in window units. Its forecast, mean + sigma x constant, still moves from window to window, so the spread of the forecast would not reveal it; the spread of the raw output is exactly zero for it.
- Relative weight change separates networks that barely moved (lazy, or switched off) from ones that learned features.

**Worked example**
```text
# Recipe E, final stage; forecast[s, w] = forecast of seed s on window w, from x after epoch 500
train_curve[s, e] = R2(test-asset training windows, seed s, epoch e)   # ~1,750 windows
test_curve[s, e]  = R2(test window, seed s, epoch e)                   # ~500 windows

V1a[s] = mean(train_curve[s, 476..500]) - mean(train_curve[s, 451..475])
V1b[s] = the same on test_curve
report |mean_s V1a|, sd_s V1a   (and the same for V1b)

trainR2[s] = mean(train_curve[s, 476..500]);   testR2[s] = mean(test_curve[s, 476..500])
gap[s]     = trainR2[s] - testR2[s]

T1_side         = mean_w var_s(forecast[s, w])        # on each side
T1_logratio     = 0.5 * log(T1_test / T1_train)
SE(T1_logratio) = jackknife over the 32 seeds

spread[s]     = std_w(output_test[s, w])              # raw network output, window units
rel_change[s] = ||x_s - x_init_s|| / ||x_init_s||     # weight matrices only
```

**Watch out**
- V1a reads the test asset only, not the pooled training objective; a flat curve on the test asset does not guarantee that the pooled objective is flat.
- Because x averages all steps since warmup, a steady drift of the base iterate shows up in x at reduced speed (half for a linear drift), and step-to-step noise is averaged away [calc, idealized]. The flatness threshold (Chapter 10) applies on this smoothed scale.
- The bias term itself is not among the quantities; status uses the T1 log ratio instead. -> **G12**
- Variance due to initialization depends on width and on parameters versus data [paper: Neal], and this grid crosses that boundary, so T1 moves for reasons other than overfitting. -> **C6**, **C9**
- The gap is in R^2 units and the T1 log ratio is dimensionless; they share no scale.
- Near-constant networks: -> **C1**. History: -> **C4**. One test window for everything: -> **C8**. Gaps: **G11** (saturation criterion), **G13** (prediction spread).

**Further reading and carry forward**
- Neal et al. 2018, Sec. 2.3 (eq. 4) and Sec. 3.4 [paper: Neal]; Defazio et al. 2024, Fig. 1 and Sec. 2.3 [paper: Defazio]; Chizat et al. on parameter movement as a sign of lazy training [paper: Chizat].
- Main quantities: test R^2, the gap, the T1 log ratio, and the V1a/V1b slopes over the last 50 epochs.
- Every quantity is read on the test asset, from the evaluation weights, with its seed standard deviation.
- Prediction spread (on the raw output) and relative weight change flag trivial or unmoved networks.

---

## Chapter 10. Classifying recipes

Synced: S10 640b3f3e

**In one line.** A recipe is *failure-unsaturated* if either curve is still moving, otherwise *failure-overfit* if seeds disagree much more on test than on training windows, and otherwise *valid*.

**Builds on.** Chapters 8 and 9.

**Concept.**
- **Bias regime.** A network is in the bias regime when its error is dominated by underfitting rather than overfitting.
- **Variance as a proxy.** An overfitting network fits idiosyncrasies of its particular training run, so different seeds tend to agree on training data but disagree on unseen data. A large rise in seed disagreement from the training side to the test side is therefore used as the overfitting signal. This proxy is the study's own construction; no project paper establishes it. -> **G12**

**In this study**
- **Mapping, in order of precedence:**
  1. If the V1a or V1b slope magnitude exceeds the **flatness threshold**, the recipe is **failure-unsaturated**.
  2. Otherwise, if the T1 log ratio exceeds **delta_sigma**, it is **failure-overfit**.
  3. Otherwise it is **valid**.
- **Flatness threshold** = delta_mu x (the median, over all recipes, of the seed standard deviation of test R^2): one value, shared by V1a and V1b. The rule is fixed in advance; the number is computed from the population after the runs, leaving no free choice. It means the change between the last two 25-epoch blocks is smaller than the test-R^2 difference the maps can resolve for a typical recipe [inference]: the end of training is flat *at map resolution*, not that further training could not help.
- **Overfit threshold** = delta_sigma, so it adds no new number; at n = 32 it corresponds to T1_test / T1_train ~ 2.9 [calc].
- Status compares point estimates with the thresholds, without a test.
- All recipes are reported and enter every map. Status is a categorical characteristic, regionalized like the others, and failures are counted per option and option pair (Chapter 11). Findings about the bias regime are read inside valid-status regions.

**Worked example**
```text
flat_thr = delta_mu x median_over_480_recipes( sd_s(testR2) )      # one number, after all runs

status(E) =
    "failure-unsaturated" if |mean_s V1a| > flat_thr or |mean_s V1b| > flat_thr
    else "failure-overfit" if T1_logratio > delta_sigma             # ~0.53, a T1 ratio of ~2.9
    else "valid"
```

**Watch out**
- delta_sigma is the regionalization's resolution, not a theory-set boundary between regimes. -> **G12**
- The training curve's seed standard deviation may differ from the test curve's, so one shared threshold is stricter on one side than the other [inference].
- The test window decides status *and* supplies test R^2, V1b and T1; the flatness threshold depends on the same results it classifies, and the quantities that feed status act as filters. -> **C8**
- A near-constant network can pass V1 trivially. -> **C1**
- See also **C6**; gap **G11** (saturation criterion).

**Further reading and carry forward**
- Neal et al. 2018, Sec. 2.3 (the decomposition behind T1) [paper: Neal]; Lakens 2017 (the margin behind the threshold) [paper: Lakens]. Not in the project: Jiang et al. 2022, *Assessing Generalization of SGD via Disagreement* (seed disagreement tracks test error; classification only) [from memory].
- Unsaturated (a slope above threshold) takes precedence over overfit (T1 log ratio above delta_sigma); everything else is valid.
- Both thresholds come from the regionalization margins, so status resolution equals map resolution.

---

## Chapter 11. From recipes to characteristics

Synced: S11 0c761646

**In one line.** Beyond per-recipe numbers, the study reports failure rates by option and option pair, how results change with more data (data saturation), and a variance decomposition over the factors.

**Builds on.** Chapters 6, 8, 9 and 10.

**Concept.**
- **Data saturation.** The point where adding data stops helping. It is a population-level trend, distinct from *training* saturation (Chapter 10).
- **Functional ANOVA** (Hutter et al. 2014). Split the variation of a performance function over its whole domain into a constant, *main effects* (one factor varied, all others averaged out) and *interaction effects* (what pairs or sets of factors add beyond their main effects). Each component's share of the total variance measures its importance [paper: Hutter, Sec. 3.1, eqs. 4 to 6]. Hutter et al. fit random forests because they observe only samples of configurations; here the full grid is observed, so the decomposition can be computed directly from cell means [inference].

**In this study**
- **Per recipe:** the quantities of Chapter 9. **Per region:** Chapter 12.
- **Failure rates** (unsaturated, overfit) per option and per pair of options. Failures are expected to come from combinations (C9). No failure-rate cutoff is applied, and the space is not changed.
- **Data saturation**, along history within each tier and along the nested financial tiers (1 -> 2 -> 3 -> 4) within each history. Tier 5 is not on the size axis, because its composition differs (C5). It uses all recipes, with status against data size reported alongside (C8).
- **Two levels of variance.** The seed standard deviation describes variation within a recipe (Chapter 8); the functional ANOVA describes variation between recipes, computed from cell means over the full grid, for both the seed mean and the seed standard deviation.
- **Secondary outputs:** tuned hyperparameter values, including the tuned learning rate against batch size as a diagnostic for C6 [paper: Li et al.; transfer is inference]. Tuned values, edge hits and the infeasible fraction are shown in the map layout, not regionalized.
- **Deferred:** which DWT bands a network relies on (confounded by the volatility features, C2). **Out of scope:** feature learning, feature similarity and test-curve shape; relative weight change is logged.

**Worked example**
```text
# Recipe E contributes to:
failure_rate[option]              for each of its 5 options (depth=2, width=32, wd, tier=3, 7y)
failure_rate[option_a, option_b]  for each of its 10 option pairs
data_saturation along history:    tier 3 x {3, 7, 18} years   (E sits at 7)
data_saturation along tiers:      {1, 2, 3, 4} x 7 years      (E sits at tier 3)
fANOVA:                           E's cell mean (and its seed sd) enters the 480-cell table
lr_vs_B diagnostic:               (B ~ 840, tuned lr of E)
```

**Watch out**
- Data-saturation curves mix more data with less gradient noise, because B ~ N/25. -> **C6**
- Status shares the test window with everything else. -> **C8**
- Band attribution is confounded by the volatility features. -> **C2**
- Gap: data-saturation measurement has no backing. -> **G14**

**Further reading and carry forward**
- Hutter, Hoos and Leyton-Brown 2014, *An Efficient Approach for Assessing Hyperparameter Importance*, Sec. 3.1 [paper: Hutter]; Li et al. 2024 for the learning-rate versus batch-size diagnostic [paper: Li et al.].
- Failure rates per option and option pair; data saturation along history and the nested financial tiers, never tier 5.
- The functional ANOVA is computed directly over the full grid, for the mean and the seed standard deviation.

---

## Chapter 12. Maps: regionalization and visual outputs

Synced: S12 5493029c

**In one line.** For each of four quantities, recipes are merged step by step into connected regions of equivalent results, and the regions are drawn as lattices, region graphs and distribution plots fixed in advance.

**Builds on.** Chapters 6, 8, 9 and 10.

**Concept.**
- **Regionalization** comes from geography: dividing spatial units, such as counties, into contiguous regions that are internally similar. Guo's REDCAP family does this by agglomerative clustering constrained by contiguity: start with every unit as its own region and repeatedly merge the most similar *adjacent* pair [paper: Guo, Sec. 1, Sec. 3].
  - **Dynamic contiguity:** after a merge, the new region's neighbours are the union of its members' neighbours [paper: Guo].
  - **Linkage** defines similarity between regions: single (closest pair), average, or complete (farthest pair). Single linkage tends to "chain"; Guo finds complete and average linkage best for internal homogeneity [paper: Guo, Sec. 2, Sec. 5]. Guo cuts the merge tree into a chosen number of regions.
- Here the "space" is the factor grid, and two recipes are adjacent when they differ by one step in one factor.
- **Empirical distribution function (EDF).** The fraction of models whose value lies below a given level. One curve per option compares whole distributions, not best cases [paper: RegNet, eq. 1, Fig. 2].

**In this study**

*What is mapped*

| Map | Quantity | Merge rule |
|---|---|---|
| Test R^2 | per-seed values | mean rule |
| Generalization gap | per-seed values | mean rule |
| T1, test vs. train | T1 log ratio | log-ratio rule |
| Status | label | connected areas with the same label; no test |

- Training R^2 is not mapped (it equals test R^2 plus the gap), nor are the per-recipe diagnostics, the seed standard deviation (reported per region), the failure rate (it becomes the status composition of a region) and data saturation. Tuned hyperparameters, edge hits and the infeasible fraction are shown in the map layout but not regionalized.

*Adjacency.* Depth, width and history connect neighbouring levels; tier connects 1-2, 2-3, 3-4 and 4-5 (C5); the two regularization options are adjacent. That gives about 6.9 neighbours per recipe on average [calc].

*Summaries.* A recipe has a mean and a standard deviation over its seeds. A region's mean mu_R is the mean of its members' means, and its sigma_R^2 the mean of its members' seed variances; the spread of the members' means is reported separately as heterogeneity.

*Merge rules* (margins from Chapter 8)
- **Mean rule (equivalence).** Two regions pass when the TOST procedure rejects that their means differ by at least delta_mu x sigma_p, where sigma_p pools the two regions' seed standard deviations. In formula: D = (|mu_A - mu_B| + t_(1-alpha) x SE_d) / (delta_mu x sigma_p), with Welch degrees of freedom, and the pair passes if D < 1 [paper: Lakens].
- **Log-ratio rule (T1 map).** A recipe's value is its T1 log ratio with a jackknife standard error; a region's value is the mean of its members'. D = (|difference| + z_(1-alpha) x SE) / delta_sigma; the pair passes if D < 1.

*Procedure*
1. Every recipe starts as its own region.
2. Compute D for every adjacent pair.
3. Merge the pair with the smallest D (exact ties broken by recipe ID).
4. Recompute the merged region's summary and the D values of its adjacent pairs.
5. Stop when the smallest D is 1 or more.

The procedure is deterministic, with no starting point to choose. Each region is reported with its members, size, mu_R, sigma_R, heterogeneity, and the factors it spans.

*Visual outputs*, fixed in advance and produced in full for every mapped quantity, so no view is chosen after seeing results:
- **Colors:** a diverging scale centred at 0 for test R^2, the gap and T1 (0 means persistence, no gap, or equal seed variance); one fixed color per status label. Each quantity keeps one scale across all its figures.
- **Lattice:** one per mapped quantity per regularization option. The outer grid has tier as rows and history as columns; each outer cell is a 4 x 4 heatmap of depth by width. Each square shows its value as color and its region ID as a label; region borders are drawn between depth and width neighbours, and borders along tier, history and regularization are read from the region IDs. Edge hits are marked with a symbol.
- **Region graph:** one per mapped quantity. Nodes are regions, labeled by ID, sized by recipe count and colored by mu_R (status: by label); edges join regions that contain an adjacent pair, with width proportional to the number of such pairs. The layout is deterministic.
- **EDF panels:** for test R^2, the gap and T1, one panel per factor with one curve per option. Status has no EDF; its per-option composition is the failure-rate table.
- **Tuning lattices** (not regionalized): the same layout, without outlines, for the tuned learning rate, tuned lambda (weight decay only) and the infeasible fraction. Borders that coincide with jumps in tuned learning rate are read by comparing lattices side by side.
- **Count:** 8 map lattices, 4 region graphs, 15 EDF panels, 5 tuning lattices [calc].

**Worked example**
```text
neighbours(E) = {
  depth 1 and depth 4                     (width 32, wd, tier 3, 7y)
  width 16 and width 64                   (depth 2, wd, tier 3, 7y)
  tier 2 and tier 4                       (depth 2, width 32, wd, 7y)
  history 3y and 18y                      (depth 2, width 32, wd, tier 3)
  no weight decay                         (depth 2, width 32, tier 3, 7y)
}   # 9 neighbours: E is interior on every ordered axis

# regionalization of one quantity q with the mean rule
regions = {each recipe: own region}
loop:
    D_min, (A, B) = min over adjacent region pairs of D(A, B)
    if D_min >= 1: stop
    merge A and B; neighbours(AB) = neighbours(A) + neighbours(B) - {A, B}
    recompute mu_R, sigma_R, SE for AB and D for its adjacent pairs
```
Recipe E appears in the "weight decay" lattice, in the outer cell (tier 3, 7 years), in the square (depth 2, width 32).

**Watch out**
- alpha holds per merge, not per map; later merges reuse data, so the nominal alpha is approximate [inference].
- The margins assume near-normal per-seed values; no normality check or tail adjustment is planned [inference]. A single recipe with heavy-tailed seeds inflates sigma_R for every region that contains it.
- The tolerated absolute difference scales with sigma, so noisy pairs merge at larger differences. This distributional view was chosen consciously; Lakens notes the same property of standardized bounds [paper: Lakens].
- **Centroid-like merging.** Merges compare region summaries, whereas Guo's evidence favours complete linkage. A drifting summary can admit a recipe that is equivalent to the summary but not to the member it borders; heterogeneity is reported for this reason [paper: Guo].
- Stopping by test has no counterpart in Guo, who cuts to a chosen number of regions. -> **G15**
- Tuning variance is not in the seed standard errors, so differences between neighbours partly reflect tuning luck; the tuning lattices and edge-hit marks are the check [paper: Bouthillier].
- EDF panels pool over all other factors, hiding effects that exist only in some combinations; the lattices show those [inference].
- See also **C1**, **C4** (gap regions spanning history options mix training-side periods), **C6**, **C7**, **C8**.

**Further reading and carry forward**
- Guo 2008, *Regionalization with Dynamically Constrained Agglomerative Clustering and Partitioning (REDCAP)*: Sec. 1, Sec. 2, Sec. 3, Sec. 5 [paper: Guo]. Lakens 2017 for TOST [paper: Lakens]. RegNet, eq. 1 and Fig. 2, for EDFs [paper: RegNet].
- Four maps: test R^2, gap, T1 log ratio, status. Regions are connected and merged by equivalence until no adjacent pair passes.
- The output set is fixed: 8 map lattices, 4 region graphs, 15 EDF panels and 5 tuning lattices.

---

## Chapter 13. Running it

Synced: S13 00a1b14e

**In one line.** Calibrate 24 extreme recipes, run all 480 recipes (16 trials x 4 seeds, then 32 seeds each), then analyze, with nothing looping back.

**Builds on.** Chapters 7, 8 and 12.

**Concept.** The order matters. Search ranges must be set before the main run, so that every analyzed recipe sees the same protocol, and the analysis must not feed back into the design.

**In this study**
- Every recipe is run once: no sampling, no iteration. Changing the recipe space based on results is out of scope.
- **Run order**
  1. **Calibration run:** 24 recipes at the extremes of what moves the best hyperparameters: three batch-size cells, tier 1 x 3 years (B ~ 30), tier 3 x 7 years (the middle, because the best Adam learning rate peaks at an intermediate batch size [paper: Li et al.]) and tier 4 x 18 years (B ~ 4,300); the four capacity corners, (depth, width) = (1, 8), (1, 64), (8, 8), (8, 64); and both regularization options. Tier 5 is left out because it has tier 4's batch size [inference]. The run uses the wide ranges, 16 trials x 4 seeds each, tuning stage only; its best values set the main-run ranges and its results are otherwise discarded.
  2. **Main run:** all 480 recipes, 16 trials x 4 seeds each in the tuning stage, then 32 seeds each in the final stage.
  3. **Analysis:** quantities (Chapter 9), status (Chapter 10), characteristics and the functional ANOVA (Chapter 11), regionalization and visual outputs (Chapter 12).
- **Cost:** with 16 trials of 4 seeds per recipe, calibration took about 17 minutes and the main run about 4.6 hours on two GPUs [measured].
- **Still open:** a batch-size control, rerunning a subset at more steps per epoch to separate batch-size from data-size effects (C6), is deferred by the owner.

**Worked example**
```text
calibration_cells   = [(tier 1, 3y), (tier 3, 7y), (tier 4, 18y)]
calibration_corners = [(1, 8), (1, 64), (8, 8), (8, 64)]
calibration = cells x corners x {none, wd}                        # 24 recipes x 64 members = 1,536 runs
ranges      = widen([min best, max best], 0.75 decade each side)  # per tuned hyperparameter
main        = 480 recipes x (16 x 4 tuning + 32 final seeds)     # 46,080 runs
# Recipe E: its cell (tier 3, 7y) is calibrated; E itself is not. It enters the main run only.
```

**Watch out**
- Record the ranges and computed thresholds actually used, so the later project can reproduce the protocol (Chapter 5).

**Further reading and carry forward**
- Li et al. 2024, Fig. 1, for why a middle batch-size cell is included [paper: Li et al.].
- Calibration (24 recipes) -> main run (480 x 64 tuning members + 480 x 32 seeds) -> analysis. Nothing loops back.

---

## Chapter 14. Known caveats (confound register)

Synced: S14 e132e954

**In one line.** Eight root causes that can make an effect look like something it is not, each with where it bites and what the study does about it.

**Builds on.** All earlier chapters.

**Concept.** A *confound* is a second cause that moves together with the factor you are studying, so an observed effect cannot be attributed to the factor alone. A confound is not a mistake; it is a limit on interpretation, and listing it is what makes the findings readable.

**In this study**

| ID | Root cause | Where it bites | What the study does |
|---|---|---|---|
| **C1** | **Persistence baseline** (Ch. 1). Next-day returns carry little predictable signal; a linear model on the same information scores below zero [measured] | A network that ignores its input outputs a constant in window units; such a network looks low-variance (small T1) and can pass V1 trivially [speculation] | Reports prediction spread on the raw output and relative weight change (Ch. 9); check how they separate within failure rates |
| **C2** | **Input representation** (Ch. 2, 4, 5) | mu is the only input that is not shift-invariant, and it leaves its training range when prices trend [measured]; MLPs extrapolate linearly there [paper: Xu]. Back-adjusted prices make mu partly reflect later corporate actions [inference]. mu and log sigma are not centred and act like an extra initialization scale [inference]. Gradients are small and eps is fixed. The volatility features are exact functions of the details [calc], so band attribution is confounded | The forecast's level comes from de-normalization, so the risk enters through one input only; band attribution deferred |
| **C4** | **History length** (Ch. 3) | It also changes regime content (2008 is inside 18 years) [inference], tuning noise (about 150 validation windows at tier 1, 3 years), training-side noise (about 750 windows), and time distance to the test window. Drift can mimic overfitting in the T1 log ratio; gap regions spanning history options mix scoring periods | Named for interpretation; no correction |
| **C5** | **Data pool and tiers** (Ch. 3) | Survivorship: current members, tenure as a lower bound on listing history, today's sector labels, dual share classes not deduplicated | Tier 4 vs. 5 isolates composition at equal size; tier 5 is kept off the data-size axis |
| **C6** | **Data-size couplings** (Ch. 3, 5, 7, 11). B ~ N/25, so gradient noise moves with tier and history | *Gains:* equal steps and epochs everywhere, so warmup and the V1 windows mean the same thing in every recipe; per-recipe tuning absorbs part of the batch effect, and at saturation batch size barely affects solution quality in small workloads [paper: Playbook; Shallue; SGD family, so inference here]. *Hides:* that guidance concerns best performance, not characteristics. Small batches regularize and large ones can overfit more [paper: Playbook]; recipes without weight decay have no tuned regularizer to compensate. Per-step noise scales with lr/B; x averages much of it, but not necessarily all [inference]. Data-saturation curves mix more data with less noise | Shared tuning ranges; tuned lr against B as a diagnostic; batch-size control deferred (Ch. 13). Unaffected: comparisons within one tier x history cell, and tier 4 vs. 5 |
| **C7** | **Single draw, small effective sample** (Ch. 1, 3, 8) | Conclusions are conditional on one asset draw, one test asset and one test period (about 500 windows). T1 omits data-sampling variance, the largest source [paper: Bouthillier]. Overlapping windows are autocorrelated | Stated as conditional; -> G17 |
| **C8** | **Test-window reuse and status** (Ch. 9 to 12) | The test window feeds V1b, test R^2 and test-side T1, *and* decides status; the flatness threshold comes from the same results it classifies, by a fixed rule; status-feeding quantities act as filters | Hyperparameters are fixed before the test window is read; the rule is fixed in advance; findings on filter quantities are read as conditional on validity |
| **C9** | **Capacity vs. data** (Ch. 4, 6) | Widths 8 and 16 are bottlenecks against 19 inputs. Depth 8 x width 64 has about 30,000 parameters against about 750 correlated windows [calc], so many overfit failures are expected [speculation]. Depths 4 and 8 straddle the hypothesized boundary at 6 | Visible as jumps in failure rate (Ch. 11) |

**Worked example.** For Recipe E, C6 applies to any comparison with other tiers or histories, and C4 applies along history. C9 is mild: about 1,729 parameters against about 21,000 windows [calc]. C2 applies through E's mu input on the test window.

**Further reading and carry forward**
- Shallue et al. 2018; Tuning Playbook (batch size); Bouthillier et al. 2021 [paper: Shallue; Playbook; Bouthillier].
- C6 (batch size moves with data size) and C8 (one test window does everything) touch nearly every finding.
- C1 and C2 are why the representation maps the output as it does and why the diagnostics exist.

---

## Chapter 15. Where evidence is thin (evidence-gap register)

Synced: S15 d7cf0b09

**In one line.** Fifteen design choices that no project paper directly supports, with the nearest support and what would close each gap; and how the project's literature is kept.

**Builds on.** All earlier chapters.

**Concept.** A gap is not a warning that a choice is wrong. It marks where the study rests on its own reasoning rather than on literature, and therefore where further reading or a small check would add the most.

**In this study**

*How the literature is kept.* Every paper the brief cites is listed in `literature/index.md`. The literature is kept small: a source is added only when a decision needs it, and one source should back a whole mechanism rather than one detail. Where the Tuning Playbook and Defazio et al. disagree on optimizer configuration, Defazio et al. applies. The brief (S15) also lists candidate additions and sources considered and not added, with the reason for each, so that they are not proposed again.

*The register*

| ID | Choice lacking backing | Chapter | Nearest support | Would close it |
|---|---|---|---|---|
| G1 | Input representation: Haar DWT of the window-normalized prices, re-injected statistics, volatility features, 16-day lookback | 2 | Liu (normalization and re-injection, for Transformers); Rhif (review only) | Studies of normalized wavelet inputs for neural return forecasting |
| G2 | Data design: tiers, single test asset, test-asset-only evaluation, histories, 80/20 split, merged validation, no embargo | 3 | Gu, Kelly and Xiu (pooling) | Evaluation-protocol literature for financial ML |
| G4 | Depth and width ranges; hypothesized overfit boundary at depth 6 | 6 | Neal, Summers (variance only); Gu, Kelly and Xiu (shallow beats deep) | n/a |
| G5 | Weight decay's role in the bias regime under schedule-free AdamW | 6 | Playbook (general) | D'Angelo et al. 2024 (candidate) |
| G6 | 500 epochs, 25 steps per epoch | 5 | Consequences only (Shallue, Playbook) | n/a |
| G7 | PyTorch default initialization [from memory] | 4 | Chizat, Kunin (initialization scale moves lazy vs. rich) | n/a |
| G8 | Reusing tuned values at the same steps and a 1.25x larger batch, under schedule-free AdamW | 7 | Defazio (training length only) | n/a |
| G9 | Tuning objective (last-25-epoch mean); shared ranges from extreme recipes plus 0.75 decade; 16 trials x 4 seeds, objective averaged over seeds | 7 | Choi (ranges, edge check); Defazio (learning-rate shift); Playbook (search-space refinement); Li et al. (batch-size dependence) | n/a |
| G10 | Fixed eps with small gradients | 5 | Choi (Adam: eps matters); Defazio (eps not tuned) | n/a |
| G11 | Saturation criterion: V1 on the test asset; shared flatness threshold from delta_mu and the median seed standard deviation | 9, 10 | Defazio (meaning of flatness); Lakens (margin) | No paper on the criterion itself |
| G12 | Overfitting criterion via the T1 log ratio with delta_sigma; no direct measure of the bias share | 9, 10 | Neal (decomposition); Summers (diversity) | Jiang et al. 2022 (candidate; classification only) |
| G13 | Prediction spread | 9 | None (relative weight change: Chizat, partial) | n/a |
| G14 | Data-saturation measurement | 11 | None | Hestness et al. 2017 (candidate) |
| G15 | Regionalization details: jackknife SE and the log-ratio rule; centroid-like merging; stopping by test; error control over sequential merges | 12 | Lakens (means only); Guo (linkage, fixed number of regions); Berger and Hsu (cite only) | n/a |
| G17 | Survivorship; effective sample size of overlapping windows | 3 | None | n/a |

IDs are never reused, so G3 and G16 are unused.

**Further reading and carry forward**
- `literature/index.md` for every source; the brief's S15 for candidates and for sources considered and not added.
- The least-supported parts are the status criteria (G11, G12), the regionalization rules (G15) and the input design (G1).

---

## Appendix. Symbols

| Symbol | Meaning | Ch. |
|---|---|---|
| P(t) | Closing price at date t | 1 |
| p_1 ... p_16 | The window's 16 log closing prices; p_16 = log P(t) | 2 |
| mu, sigma | Window statistics: mean and population standard deviation of the window's log prices | 2 |
| z, y, x | Schedule-free sequences: base iterate, gradient point, evaluation weights | 5 |
| x_1 | Initialization (the start of x) | 9 |
| beta, beta2, eps | Schedule-free interpolation; Adam second-moment rate; Adam denominator constant | 5 |
| gamma_t, c_(t+1) | Learning rate at step t, with warmup; averaging weight gamma_t^2 / sum(gamma_i^2) | 5 |
| lambda | Weight-decay strength | 5, 7 |
| N, B, K | Windows per stage; batch size (~N/K); steps per epoch | 5 |
| n | Seeds per recipe | 8 |
| alpha, beta (tests) | Type I and Type II error rates (not the optimizer's beta) | 8 |
| delta_mu, delta_sigma | Equivalence margins for means (in seed-sd units) and for log ratios of standard deviations | 8 |
| V1a, V1b | Training- and test-curve slopes over the last 50 epochs | 9 |
| T1 | Mean over windows of the forecast variance across seeds | 9 |
| mu_R, sigma_R, n_R | Region mean, pooled seed standard deviation, member count | 12 |
| D | Merge statistic; a pair passes if D < 1 | 12 |
