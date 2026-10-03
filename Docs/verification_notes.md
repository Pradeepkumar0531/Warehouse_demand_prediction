# Stage 4 — Gradient Verification and Numerical Stability

## 1. Gradient verification

The attention implementation is differentiable, but passing an autograd test alone does not verify that the forward equations were implemented correctly. We therefore compare PyTorch's automatic derivative with a finite-difference approximation.

For a scalar loss L and parameter theta_i:

    dL/dtheta_i ~= [L(theta_i + epsilon) - L(theta_i - epsilon)] / (2 epsilon)

Central difference is used because the first-order errors from the positive and negative perturbations largely cancel, giving a more accurate check than a one-sided difference.

The check uses float64 and epsilon = 1e-5. Float64 reduces round-off error, while epsilon is small enough to approximate a local derivative without making the loss difference indistinguishable from numerical noise.

The scalar loss is mean squared error between the attention output and a deterministic random target. Every scalar in W_Q, W_K, and W_V is perturbed independently. The original value is restored after each perturbation.

Relative gradient error:

    ||g_autograd - g_finite_difference||
    ------------------------------------
    ||g_autograd|| + ||g_finite_difference||

A value close to zero means the two independently computed derivatives agree.

## 2. Numerical stability

Softmax contains exp(score). Large positive scores can overflow floating-point arithmetic. Attention therefore subtracts the maximum score in each row before exponentiation:

    softmax(s) = exp(s - max(s)) / sum(exp(s - max(s)))

This does not change the mathematical probabilities because the same constant is subtracted from every value in the row:

    exp(s_i - c) / sum_j exp(s_j - c)
    = exp(s_i) / sum_j exp(s_j)

The experiment uses scores around 1000, where direct exponentiation overflows, and confirms that max-subtracted softmax remains finite and sums to one.

## 3. Why this stage matters

Gradient verification checks the backward behavior against an independent numerical approximation. Numerical-stability testing checks that the implementation remains finite for extreme logits. Together they address two separate failure modes: an incorrect derivative path and a mathematically correct but numerically unsafe computation.
