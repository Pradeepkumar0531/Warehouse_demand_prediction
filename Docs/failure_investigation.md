# Stage 8 — Failure Investigation

## Purpose

Stage 6 produced aggregate forecasting metrics.

Stage 8 goes one level deeper:

1. Which individual test forecasts have the largest errors?
2. Are those cases associated with large one-hour demand changes?
3. What does the attention mechanism focus on on average?

This is an error-analysis stage, not another training stage.

The Stage 6 model is loaded and kept fixed.

---

# 1. Absolute forecast error

For each test example:

\[
e_t =
|y_t-\hat y_t|
\]

where:

- \(y_t\) = actual next-hour demand;
- \(\hat y_t\) = predicted next-hour demand.

The test cases are sorted by \(e_t\).

The 20 largest-error cases are stored.

---

# 2. Why inspect the previous demand?

For every high-error case, the script stores:

\[
D_t
\]

which is the most recent observed demand before the target.

It also calculates:

\[
\text{target jump}
=
y_{t+1}-D_t
\]

### Why?

The synthetic warehouse generator deliberately contains rare positive spikes
and negative drops.

A large one-hour target jump is therefore useful evidence when investigating
why a forecast was difficult.

This does not prove that every large error was caused by an injected event.

It is a diagnostic signal.

---

# 3. Saved high-error table

The file:

```text
top_forecast_errors.csv
```

contains:

```text
rank
target_timestamp
target_index
previous_demand
actual
prediction
absolute_error
target_jump
```

The ranking is by absolute error from largest to smallest.

---

# 4. Attention inspection

The forecasting model predicts from:

```python
output[:, -1, :]
```

Therefore the attention row corresponding to the final query is the most
direct attention distribution for the actual forecast representation.

For every test example, that row contains 24 weights.

The script averages those final-query weights across the test set.

---

# 5. Attention lag convention

The 24 input positions are:

```text
position 0  -> 23 hours before target
position 1  -> 22 hours before target
...
position 22 -> 1 hour before target
position 23 -> 0 hours before target
```

Therefore the output is labeled using:

\[
23,22,\ldots,1,0
\]

hours before the target.

This makes the attention plot easier to interpret.

---

# 6. Why average attention?

One forecast can have a very different attention pattern from another.

Averaging gives a compact summary of the model's general behavior over the
test period.

However, average attention should not be interpreted as causal proof.

It tells us where the model placed weight, not that those observations caused
the prediction.

---

# 7. Outputs

Run:

```bash
python Failure_investigation/failure_investigation.py
```

The script creates:

```text
top_forecast_errors.csv
average_attention_by_lag.csv
forecast_error_analysis.png
average_attention_by_lag.png
```

---

# 8. Verified observations

The verified Stage 6 model produced:

```text
Test forecasts: 1311
Mean absolute error: 18.5462
Largest absolute error: 211.5344
Mean target jump among top 20 errors: 49.5123
```

The five largest errors were:

```text
2025-11-27 08:00:00
actual     = 587.19
prediction = 375.66
error      = 211.53
jump       = 189.96

2025-12-30 02:00:00
actual     = 586.98
prediction = 376.24
error      = 210.74
jump       = 199.09

2025-12-12 18:00:00
actual     = 523.57
prediction = 333.75
error      = 189.82
jump       = 198.95

2025-12-01 18:00:00
actual     = 495.67
prediction = 310.85
error      = 184.82
jump       = 188.77

2025-12-24 15:00:00
actual     = 505.86
prediction = 323.40
error      = 182.47
jump       = 189.10
```

These observations indicate that the largest forecast errors occur alongside
very large one-hour demand changes in this synthetic dataset.

That is a useful failure-mode finding because the base forecasting window
only contains historical information available before the sudden change.

---

# Design principle

The goal here is not to hide the model's difficult cases behind one average
metric.

A strong analysis should show:

- overall forecasting quality;
- where the model fails;
- how large those failures are;
- what temporal positions receive attention.

That is why this stage is kept separate from model training.
