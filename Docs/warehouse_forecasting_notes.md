# Stage 6 — Warehouse Forecasting

## What this stage does

The assignment now moves from the toy attention experiment to the actual
synthetic warehouse forecasting problem.

The task is:

> Use the previous 24 hourly demand values to predict the next hour.

For a target at time `t`:

\[
X_t = [D_{t-23}, D_{t-22}, \ldots, D_t]
\]

and:

\[
y_t = D_{t+1}
\]

The existing chronological train/validation/test files from
`Data_generation` are reused. No new random split is created here.

---

## Files

Only three files are needed for this stage:

```text
Warehouse_forecasting/
├── warehouse_forecasting.py
├── test_warehouse_forecasting.py
└── stage6_notes.md
```

`warehouse_forecasting.py` contains both the implementation and the
runnable experiment.

Run it with:

```bash
python warehouse_forecasting.py
```

Run the tests with:

```bash
pytest test_warehouse_forecasting.py -q
```

---

## Data

The script reads:

```text
Data_generation/train_data.npz
Data_generation/validation_data.npz
Data_generation/test_data.npz
```

The split sizes established earlier are:

- train: 6115
- validation: 1310
- test: 1311

The chronological split is preserved because this is a forecasting problem.
Randomly shuffling observations could allow future patterns to influence
training while evaluating earlier observations.

---

# Baselines

Before using attention, three simple forecasting rules are evaluated.

## 1. Persistence

\[
\hat{D}_{t+1} = D_t
\]

The prediction is simply the most recently observed demand.

### Why use it?

Warehouse demand is temporally correlated. If demand changes relatively
slowly from one hour to the next, the latest value is already a strong
prediction.

This gives us a very simple reference point.

---

## 2. Previous-day same-hour

\[
\hat{D}_{t+1} = D_{t-23}
\]

Because the input contains 24 previous hours, the first value corresponds
to approximately the same hour on the previous day.

### Why use it?

The synthetic warehouse environment contains daily seasonality.

This baseline therefore tests whether simply using yesterday's corresponding
hour captures useful daily structure.

---

## 3. 24-hour mean

\[
\hat{D}_{t+1}
=
\frac{1}{24}
\sum_{i=0}^{23} D_{t-i}
\]

### Why use it?

This removes short-term variation and predicts the recent average demand.

It provides a simple smoothing reference.

---

# Evaluation metrics

Three metrics are calculated.

## MAE

\[
MAE =
\frac{1}{N}
\sum_{i=1}^{N}
|y_i-\hat{y}_i|
\]

### Why?

MAE is easy to interpret in warehouse-demand units.

For example, an MAE of 20 means the prediction is off by about
20 demand units on average.

---

## RMSE

\[
RMSE =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
}
\]

### Why?

RMSE penalizes large errors more strongly than MAE.

That matters here because the synthetic generator deliberately contains
rare spikes and drops.

---

## MAPE

\[
MAPE =
\frac{100}{N}
\sum_{i=1}^{N}
\left|
\frac{y_i-\hat{y}_i}{y_i}
\right|
\]

### Why?

MAPE expresses error as a percentage, making the result easier to interpret
relative to demand magnitude.

The generated demand remains positive, so zero-target division is not a
problem for this experiment.

---

# Input normalization

The attention model standardizes its inputs:

\[
z =
\frac{x-\mu_{train}}
{\sigma_{train}}
\]

where:

- \(\mu_{train}\) is the training-set mean;
- \(\sigma_{train}\) is the training-set standard deviation.

The same training statistics are used for validation and test data.

### Why?

The demand values are around hundreds of units.

The attention mechanism calculates dot products. Very large raw magnitudes
can make optimization poorly conditioned.

Standardization puts the input on a more convenient numerical scale.

### Why only training data?

Using validation or test statistics would allow information from later
periods to influence preprocessing.

Therefore the scaler is fitted only on training inputs.

---

# Attention forecasting model

The model is deliberately small.

The sequence is processed as:

```text
scalar demand
      ↓
1 → 4 dimensional projection
      ↓
sinusoidal position information
      ↓
first-principles single-head attention
      ↓
final time-position representation
      ↓
linear prediction head
      ↓
next-hour demand
```

No Transformer library or high-level attention implementation is used.

The existing:

```text
Attention_mechanism/attention.py
```

is reused.

---

# Why positional encoding is included

The attention mechanism operates on content.

Without explicit position information, two equal demand values at different
hours can look identical to the attention mechanism.

The warehouse task has strong temporal structure, including daily and weekly
patterns, so the model needs information about where each observation occurs
inside the 24-hour window.

The implementation uses sinusoidal position features:

\[
PE(pos,2i)
=
\sin
\left(
\frac{pos}
{10000^{2i/d_{model}}}
\right)
\]

\[
PE(pos,2i+1)
=
\cos
\left(
\frac{pos}
{10000^{2i/d_{model}}}
\right)
\]

These are added to the projected demand representation before attention.

### Why this method?

It is deterministic and requires no additional learned parameters.

It also keeps the positional mechanism visible instead of hiding it inside
a library abstraction.

---

# Model dimensions

The model uses:

```text
window size = 24
d_model = 4
single attention head
```

### Why d_model = 4?

This is intentionally small.

The goal at this stage is to study the attention mechanism rather than build
a large forecasting architecture.

A small representation also keeps the number of trainable parameters easy
to inspect.

---

# Training

The model uses:

```text
optimizer = Adam
learning rate = 0.01
epochs = 200
seed = 42
loss = mean squared error
```

The loss is calculated on standardized demand.

### Why MSE?

MSE is straightforward for a continuous next-value prediction task and
penalizes large forecasting errors.

This is also consistent with the earlier toy learning experiment.

---

# Validation selection

During training, validation loss is recorded after every epoch.

The model state with the lowest validation loss is saved and restored.

The test set is evaluated only after this selection.

### Why?

The test set should represent unseen future data.

If the test set were repeatedly used to choose epochs or tune the model,
it would no longer be a clean final evaluation.

---

# Outputs

Running:

```bash
python warehouse_forecasting.py
```

creates:

```text
warehouse_forecasting_results.csv
warehouse_training_history.csv
warehouse_training_history.png
warehouse_attention_model.pt
```

The CSV contains:

```text
model
MAE
RMSE
MAPE_percent
```

The training-history CSV records training and validation loss by epoch.

The PNG shows training and validation loss.

The `.pt` file stores the selected model state together with the training
mean and standard deviation so the exact trained model can be reused in
later stages.

---

# Expected experiment result

The already-verified run produced approximately:

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

These are recorded observations from the fixed synthetic experiment, not a
claim that the attention model will behave the same way on real warehouse
data.

---

# Important failure finding carried into this stage

An earlier version of the warehouse attention model did not include explicit
time-position information.

That version performed substantially worse than the simple persistence
baseline.

Because the synthetic warehouse data contain temporal structure, this was a
useful failure signal rather than something to hide.

Adding explicit sinusoidal position information substantially changed the
model behavior.

The failure and revision are kept as part of the reasoning for the design.
