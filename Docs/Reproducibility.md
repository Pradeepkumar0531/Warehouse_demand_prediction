# Stage 9 — Reproducibility

## Purpose

The assignment should be reproducible rather than dependent on one successful
run.

This stage checks two important sources of randomness:

1. synthetic-data generation;
2. neural-network initialization and training.

It also records the runtime environment.

---

# 1. Synthetic-data reproducibility

The warehouse environment is generated twice with:

```text
num_days = 365
seed = 42
```

The two generated tables are compared directly.

A SHA-256 hash of the demand vector is also calculated.

For a deterministic generation process, both runs should have:

```text
same values = True
same demand hash = True
```

## Why compare both?

The direct table comparison is the actual equality check.

The hash gives a compact fingerprint that can be recorded in a report and
checked later.

---

# 2. Training reproducibility

The attention model is trained twice using the same Stage 6 configuration:

```text
seed = 42
learning rate = 0.01
epochs = 200
```

The same training and validation arrays are used for both runs.

The following are compared:

- loss history;
- trained parameter tensors;
- training mean;
- training standard deviation.

The parameter tensors must be exactly equal.

This is stronger than merely checking that the final loss is approximately
the same.

---

# 3. Why set the seed before model creation?

The attention model contains randomly initialized parameters.

PyTorch uses the random-number generator when those parameters are created.

Therefore the seed must be set before the model is constructed.

The Stage 6 training function already calls:

```python
torch.manual_seed(seed)
```

before creating the model.

This stage verifies that this design actually produces identical training
results for repeated runs.

---

# 4. Environment recording

The report records:

- Python version;
- platform;
- NumPy version;
- PyTorch version;
- experiment seed.

## Why?

Exact reproducibility can also depend on the software environment.

Recording versions makes it easier to determine whether a future numerical
difference comes from changed code or changed dependencies.

---

# 5. Output

Run:

```bash
python Reproducibility/reproducibility.py
```

The script creates:

```text
reproducibility_report.txt
```

The report contains the data hashes, training equality checks, environment
versions, and final pass/fail status.

---

# 6. What counts as a pass?

The stage passes only when all of these are true:

```text
same synthetic data
same synthetic data hash
same training history
same trained parameters
same training mean
same training standard deviation
```

This makes the final check deliberately strict.

---

# 7. Scope

This stage verifies deterministic behavior under the same software/runtime
environment and explicit random seeds.

It does not claim that every numerical operation on every hardware platform
will be bit-for-bit identical.

The goal is to establish reproducibility for the project configuration used
for this assignment.
