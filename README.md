# Warehouse Demand Forecasting with First-Principles Self-Attention

A controlled time-series forecasting project that implements
self-attention from first principles and evaluates it on a synthetic
warehouse-demand problem.

> **Scope:** The warehouse data are synthetic. Results describe this
> controlled experiment and should not be treated as claims about real
> warehouse operations.

## Problem

For each target time `t`, the model uses the previous 24 hourly
observations:

$$
X_t=[D_{t-23},D_{t-22},...,D_t]
$$

to predict:

$$
y_t=D_{t+1}.
$$

The 24-hour window represents one complete daily cycle.

## Synthetic data

Demand is generated as:

$$
D_t=\max(0,B+T_t+S_t+W_t+E_t+\epsilon_t)
$$

where:

-   `B`: baseline demand;
-   `T_t`: slow trend;
-   `S_t`: daily seasonality;
-   `W_t`: weekly effect;
-   `E_t`: rare spike/drop event;
-   `epsilon_t`: ordinary random noise.

Configuration:

  Component                                         Value
  ------------------- -----------------------------------
  Duration                                       365 days
  Observations                               8,760 hourly
  Seed                                                 42
  Baseline                                            300
  Trend                                       `0.005 * t`
  Daily seasonality     `45 sin(2πh/24) + 20 cos(2πh/24)`
  Noise                                       `N(0, 12²)`
  Spike probability                            0.005/hour
  Spike magnitude                       Uniform(100, 220)
  Drop probability                             0.003/hour
  Drop magnitude                         Uniform(80, 140)

Seed-42 sanity checks:

-   mean: `328.20`
-   standard deviation: `44.79`
-   minimum: `90.63`
-   maximum: `587.19`
-   spikes: `51`
-   drops: `27`

Demand is clipped at zero because negative demand is not physically
meaningful.

## Project stages

### Stage 1 --- Synthetic Data

Creates and tests the 365-day warehouse environment.

Checks include length, timestamps, reproducibility, non-negativity,
component reconstruction, weekday effects, and event magnitudes.

### Stage 2 --- Forecasting Dataset

Creates 8,736 supervised examples using 24 observed hours to predict the
next hour.

Chronological split:

-   train: `6,115`
-   validation: `1,310`
-   test: `1,311`

The split is by target timestamp rather than random shuffling to avoid
temporal leakage.

### Stage 3 --- First-Principles Attention

The core mechanism is implemented directly:

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
S=QK^T
$$

$$
S_{scaled}=\frac{QK^T}{\sqrt{d_k}}
$$

$$
A=softmax(S_{scaled})
$$

$$
H=AV.
$$

A single attention head is used. No high-level Transformer/attention API
is used.

Softmax subtracts the maximum value of each row before exponentiation
for numerical stability.

### Stage 4 --- Mathematical Verification

Central finite differences are used:

$$
g_i^{FD}=\frac{L(\theta_i+\epsilon)-L(\theta_i-\epsilon)}{2\epsilon}.
$$

Recorded relative errors:

  Parameter       Relative error
  ----------- ------------------
  `W_Q`         `8.221 × 10^-11`
  `W_K`         `5.175 × 10^-11`
  `W_V`         `2.037 × 10^-11`

Numerical-stability tests also pass.

### Stage 5 --- Toy Learning

A controlled temporal rule is learned:

$$
y=0.60x_{-1}+0.25x_{-2}+0.10x_{-3}+0.05x_{-4}.
$$

Scaled and unscaled attention use the same data, model setup, optimizer,
learning rate, epochs, and seed.

Final MSE:

  Variant             MSE
  ---------- ------------
  Scaled       `0.088134`
  Unscaled     `0.087712`

This does not support claiming that scaling must always produce lower
final loss.

### Stage 6 --- Warehouse Forecasting

Baselines:

**Persistence**

$$
\hat D_{t+1}=D_t
$$

**Previous-day same-hour**

$$
\hat D_{t+1}=D_{t-23}
$$

**24-hour mean**

$$
\hat D_{t+1}=\frac{1}{24}\sum_{i=0}^{23}D_{t-i}
$$

Metrics are MAE, RMSE, and MAPE.

The attention model uses scalar demand embeddings, `d_model=4`,
sinusoidal positional encoding, one attention head, a final-position
representation, and a linear prediction head. Training uses Adam,
learning rate `0.01`, MSE, and the best validation checkpoint.

### Stage 7 --- Distribution Shift

The Stage 6 checkpoint is frozen and tested without retraining under:

1.  a `+40` demand-level shift;
2.  ordinary noise standard deviation increased from `12` to `24`.

### Stage 8 --- Failure Investigation

The 20 largest absolute errors are inspected using actual demand,
prediction, previous demand, and target jump.

Five largest absolute errors:

    Actual   Prediction    Error   Target jump
  -------- ------------ -------- -------------
    587.19       375.66   211.53        189.96
    586.98       376.24   210.74        199.09
    523.57       333.75   189.82        198.95
    495.67       310.85   184.82        188.77
    505.86       323.40   182.47        189.10

Large abrupt one-hour changes are associated with the largest failures.
This is diagnostic evidence, not proof of causality.

### Stage 9 --- Reproducibility and QA

The project checks same-seed data equality and hashes, repeated training
loss histories, model parameters, and normalization statistics.

Final QA:

``` text
Project structure:          PASS
Required generated outputs: PASS
Result files:               PASS
Full test suite:            PASS
FINAL PROJECT QA:           PASS
```

The full test suite passed **50 tests**.

## Base test results

  Model                          MAE      RMSE    MAPE
  ------------------------ --------- --------- -------
  Persistence                18.5331   30.0543   5.34%
  Previous-day same-hour     24.7375   36.0324   7.25%
  24-hour mean               33.8771   41.5632   9.94%
  Attention                  18.5462   26.5753   5.30%

Persistence has slightly lower MAE, while attention has lower RMSE and
slightly lower MAPE in this test. The project does not claim an overall
winner.

## Distribution-shift results

  Scenario       Model                MAE      RMSE     MAPE
  -------------- -------------- --------- --------- --------
  Original       Persistence      18.5331   30.0543    5.34%
  Original       Previous-day     24.7375   36.0324    7.25%
  Original       24-hour mean     33.8771   41.5632    9.94%
  Original       Attention        18.5462   26.5753    5.30%
  Level +40      Persistence      18.5636   30.0817    4.78%
  Level +40      Previous-day     25.0080   36.4000    6.54%
  Level +40      24-hour mean     33.8002   41.5871    8.84%
  Level +40      Attention        25.3494   34.4030    6.29%
  Noise std 24   Persistence      33.5483   44.9858    9.84%
  Noise std 24   Previous-day     37.6624   49.2329   11.08%
  Noise std 24   24-hour mean     37.9344   47.3996   11.26%
  Noise std 24   Attention        30.1605   39.1146    8.85%

## Key findings

1.  Attention can learn the controlled temporal task.
2.  Persistence is a strong baseline because the synthetic process is
    strongly temporally correlated.
3.  Attention does not automatically dominate simpler forecasting rules.
4.  RMSE and MAE reveal different aspects of the error distribution.
5.  The scaling ablation did not show lower final toy loss for scaled
    attention in this run.
6.  The fixed attention model degrades under the tested distribution
    shifts.
7.  Abrupt one-hour changes are associated with the largest forecast
    failures.
8.  Attention weights are treated as descriptive, not causal
    explanations.

## Limitations

-   The warehouse data are synthetic.
-   The attention model is intentionally small: `d_model=4`, single
    head.
-   Only a 24-hour history is used.
-   Only two distribution shifts are tested.
-   Changes in seasonality, spike frequency, missing observations, and
    other structural shifts are not tested.
-   Rare events can be unpredictable because the model receives no event
    indicator.
-   Results should not be generalized directly to real warehouse
    operations.

## Why first-principles attention?

The implementation deliberately exposes the core mechanism instead of
hiding it inside a Transformer abstraction:

``` text
Input sequence
      |
      +--> Q = XW_Q
      +--> K = XW_K
      +--> V = XW_V
               |
               v
             QK^T
               |
               v
          / sqrt(d_k)
               |
               v
        stable softmax
               |
               v
          attention A
               |
               v
             A V
               |
               v
          output H
```

This makes the mathematics directly inspectable and testable.

## Repository organization

``` text
project/
├── Attention_mechanism/
├── Data_generation/
├── Gradient_verification/
├── Numerical_stability/
├── Toy_learning/
├── Warehouse_forecasting/
├── Distribution_shift/
├── Failure_investigation/
├── Reproducibility/
├── Final_verification/
├── 01_Phase_0_Design_Document.pdf
├── 02_Mathematical_Derivation.pdf
├── 03_Experiment_Report.pdf
├── 04_Reflection_Report.pdf
├── 05_AI_Tools_Document.pdf
└── README.md
```

## Execution

The later forecasting stages reuse the generated train/validation/test
data and first-principles attention implementation.

Stage 6:

``` bash
python Warehouse_forecasting/run_warehouse_forecasting.py
```

Stage 7:

``` bash
python Distribution_shift/distribution_shift.py
```

Stage 8:

``` bash
python Failure_investigation/failure_investigation.py
```

Stages 6--8 together:

``` bash
python run_stage6_8.py
```

Full test suite:

``` bash
pytest
```

## Documentation

The submission documentation contains:

1.  `01_Phase_0_Design_Document.pdf` --- design, assumptions,
    hypotheses, failure modes, evaluation, and planned generalization.
2.  `02_Mathematical_Derivation.pdf` --- attention mathematics,
    gradients, stability, and verification.
3.  `03_Experiment_Report.pdf` --- experiment configurations, results,
    distribution shifts, failure analysis, and QA.
4.  `04_Reflection_Report.pdf` --- expectations, surprises, limitations,
    failures, learning, and intellectual honesty.
5.  `05_AI_Tools_Document.pdf` --- AI assistance log, verification, and
    corrected/rejected suggestions.

## Intellectual honesty

The project does not claim:

-   that attention is universally better than persistence;
-   that scaled attention always produces lower loss;
-   that attention weights are causal explanations;
-   that synthetic results predict real warehouse performance;
-   that this small model establishes how larger Transformer
    architectures behave.

All conclusions are limited to the tested configuration and
data-generating environment.
