# Stage 7 — Distribution Shift

## Purpose

Stage 6 evaluates the warehouse attention model on the original synthetic
test period.

Stage 7 asks a different question:

> What happens when the future demand process changes after training?

The Stage 6 model is loaded and kept fixed.

There is no retraining on the shifted data.

---

## Scenario 1 — Original test

The original Stage 6 test period is evaluated again.

This is the reference condition.

No observations are modified.

---

## Scenario 2 — Level shift

Starting at the first forecasted target, add 40 demand units:

\[
D'_t = D_t + 40
\]

The first 24 observations remain unchanged so that the model has a valid
pre-shift context.

### Why +40?

The Stage 6 training standard deviation is approximately 43.9 demand units.

A +40 change is therefore large enough to create a visible distribution
shift while remaining in a plausible range for this synthetic environment.

The value is an explicit stress-test assumption.

It is not learned from the test data.

---

## Scenario 3 — Increased noise

Starting at the first forecasted target:

\[
D'_t =
\max(0, D_t+\epsilon_t)
\]

with:

\[
\epsilon_t
\sim
\mathcal{N}(0,24^2)
\]

The original synthetic generator uses noise standard deviation 12.

Therefore this experiment doubles the random-noise scale:

\[
24 = 2 \times 12
\]

A fixed seed of 123 makes the noisy scenario reproducible.

---

## Why keep the first 24 observations unchanged?

The forecasting input contains the previous 24 hours.

If the shift were applied to those first 24 observations as well, the model
would begin the experiment already inside the shifted regime.

Keeping them unchanged gives a clean transition:

```text
pre-shift 24-hour context
          ↓
distribution changes
          ↓
shifted future observations
```

This makes the stress test easier to interpret.

---

## Model and preprocessing

The model comes from:

```text
Warehouse_forecasting/warehouse_attention_model.pt
```

That checkpoint contains:

- the trained attention-model parameters;
- the training mean;
- the training standard deviation.

Those training statistics are not changed for the shifted data.

This is important because using new test-period statistics would partially
adapt the model to the distribution shift.

---

## What is evaluated?

The same four methods from Stage 6:

1. Persistence
2. Previous-day same-hour
3. 24-hour mean
4. Attention

The same three metrics are retained:

\[
MAE =
\frac{1}{N}
\sum_{i=1}^{N}
|y_i-\hat y_i|
\]

\[
RMSE =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat y_i)^2
}
\]

\[
MAPE =
\frac{100}{N}
\sum_{i=1}^{N}
\left|
\frac{y_i-\hat y_i}{y_i}
\right|
\]

Keeping the same metrics allows direct comparison with Stage 6.

---

## Expected outputs

Run:

```bash
python Distribution_shift/distribution_shift.py
```

The script creates:

```text
distribution_shift_results.csv
distribution_shift_results.png
```

The CSV contains:

```text
scenario
model
MAE
RMSE
MAPE_percent
```

The figure shows MAE across the three scenarios for all four methods.

---

## Verified results

### Original test

```text
Persistence:
MAE  = 18.5331
RMSE = 30.0543
MAPE = 5.34%

Previous-day same-hour:
MAE  = 24.7375
RMSE = 36.0324
MAPE = 7.25%

24-hour mean:
MAE  = 33.8771
RMSE = 41.5632
MAPE = 9.94%

Attention:
MAE  = 18.5462
RMSE = 26.5753
MAPE = 5.30%
```

### Level shift +40

```text
Persistence:
MAE  = 18.5636
RMSE = 30.0817
MAPE = 4.78%

Previous-day same-hour:
MAE  = 25.0080
RMSE = 36.4000
MAPE = 6.54%

24-hour mean:
MAE  = 33.8002
RMSE = 41.5871
MAPE = 8.84%

Attention:
MAE  = 25.3494
RMSE = 34.4030
MAPE = 6.29%
```

### Noise std 24

```text
Persistence:
MAE  = 33.5483
RMSE = 44.9858
MAPE = 9.84%

Previous-day same-hour:
MAE  = 37.6624
RMSE = 49.2329
MAPE = 11.08%

24-hour mean:
MAE  = 37.9344
RMSE = 47.3996
MAPE = 11.26%

Attention:
MAE  = 30.1605
RMSE = 39.1146
MAPE = 8.85%
```

These are observations from this synthetic environment only.

---

## Design principle

The purpose of Stage 7 is not to make the attention model look better.

The purpose is to expose how the fixed forecasting system behaves when the
data-generating process changes.

That makes the result useful for robustness and failure analysis.
