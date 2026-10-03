# Stage 3 — First-Principles Attention

## What we are implementing

For an input sequence `X`:

Q = X W_Q

K = X W_K

V = X W_V

The query and key representations are compared with:

S = Q K^T

For scaled attention:

S_scaled = S / sqrt(d_k)

The rows are converted into weights with softmax:

A_ij = exp(S_ij) / sum_j exp(S_ij)

Finally:

Y = A V

The implementation uses the same sequence-level calculation directly in PyTorch tensor operations.

## Why Q, K, and V?

The three projections give the mechanism three different learned representations of the same input.

- `Q` represents what each position is looking for.
- `K` represents what each position offers for matching.
- `V` represents the information that is actually aggregated.

Keeping these roles separate lets the model learn which historical positions are relevant and what information should be taken from them.

## Why Q K^T?

A dot product measures similarity between a query and a key.

If there are `T` sequence positions, comparing every query with every key produces a `T x T` score matrix.

For the warehouse task, `T = 24`, so the attention score matrix will eventually be `24 x 24`.

## Why divide by sqrt(d_k)?

The dot product contains `d_k` terms. As `d_k` becomes larger, the raw dot products can also become larger in magnitude.

Softmax is sensitive to large differences between its inputs. Very large logits can make the distribution extremely concentrated.

Dividing by `sqrt(d_k)` keeps the score scale more controlled.

The PRD specifically requires the scaled dot-product formulation and later asks us to compare it against the unscaled version as an ablation.

## Why softmax?

Softmax converts each row of scores into non-negative weights that sum to one.

That makes the next operation:

Y = A V

a weighted combination of the value vectors.

## Why subtract the maximum before exp()?

Mathematically:

softmax(z) = softmax(z - c)

for any constant `c` applied to every element of the same row.

We choose:

c = max(z)

so the largest shifted value is zero. This prevents `exp(z)` from overflowing when the original scores are very large.

This is a numerical-stability implementation detail, not a change to the mathematical softmax result.

## Tensor shapes

For input:

X: `(batch, sequence_length, d_model)`

the projections are:

Q: `(batch, sequence_length, d_k)`
K: `(batch, sequence_length, d_k)`
V: `(batch, sequence_length, d_v)`

Then:

Q K^T: `(batch, sequence_length, sequence_length)`

A V: `(batch, sequence_length, d_v)`

For the later warehouse model, sequence length will be 24.

## Why single-head?

The assignment asks us to understand and implement the attention mechanism itself. A single attention mechanism is enough to expose all required mathematical operations without adding the extra complexity of multiple heads.

## Why no Transformer block yet?

At this stage we are verifying the core attention operation in isolation. Adding feed-forward layers, residual blocks, layer normalization, and other Transformer components would make it harder to identify whether a problem came from attention itself.

Those components are not required for the core first-principles attention task.
