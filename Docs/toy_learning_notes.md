# Stage 5 — Toy Learning Dynamics and Scaling Ablation

## Task

Use a small next-value prediction problem before moving to warehouse forecasting. Each example contains 8 observations. The target is a weighted combination of the four most recent observations:

`y = 0.60*x[-1] + 0.25*x[-2] + 0.10*x[-3] + 0.05*x[-4]`

The task is deliberately simple so that training behavior can be inspected without confounding it with the warehouse generator.

## Model

The toy model uses the first-principles `SelfAttention` implementation followed by a single linear output head. The prediction is produced from the attention output at the final sequence position.

Two otherwise identical models are trained:

1. scaled attention: `QK^T / sqrt(d_k)`
2. unscaled attention: `QK^T`

This is an ablation: only the score scaling choice changes.

## Training

- 512 deterministic examples
- sequence length: 8
- model dimension: 4
- Adam optimizer
- learning rate: 0.01
- 100 epochs
- mean squared error loss
- fixed random seed for initialization

The purpose is not to claim that one optimizer or hyperparameter setting is universally best. The purpose is to observe the effect of score scaling under a controlled, repeatable setup.

## What to inspect

`toy_learning_results.csv` records the loss at every epoch for both variants. The expected evidence is decreasing training loss and a measurable difference in optimization behavior between scaled and unscaled attention.
