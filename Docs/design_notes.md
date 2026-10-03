# Stage 1 — Synthetic Warehouse Environment

## Purpose

The PRD requires a synthetic warehouse environment with meaningful temporal structure, controlled randomness, and unusual events. We keep the generator simple so that every part of the data-generating process can be inspected and explained.

## Demand equation

For hour `t`:

D_t = max(0,
          B
        + T_t
        + S_t
        + W_t
        + E_t
        + epsilon_t)

Where:

- `B` is the baseline demand.
- `T_t` is a small long-term trend.
- `S_t` is daily/hour-of-day seasonality.
- `W_t` is a day-of-week effect.
- `E_t` represents rare spikes or drops.
- `epsilon_t` is random noise.

### Why each term is used

**Baseline (`B = 300`)**

Provides a stable demand level around which the other effects vary. Without a baseline, the seasonal and event terms would have no natural operating level.

**Trend (`T_t = 0.005 t`)**

Adds a slow change over the year. It prevents the series from being perfectly stationary while keeping the trend small enough that it does not dominate the daily and weekly patterns.

**Daily seasonality**

45 * sin(2*pi*h/24) + 20 * cos(2*pi*h/24)

This creates a smooth 24-hour cycle. Using sine and cosine together lets the daily pattern have both amplitude and phase rather than forcing its peak to occur at a specific arbitrary hour.

**Weekly effect**

A fixed day-of-week adjustment creates a second time scale. This gives the environment weekly structure in addition to the daily pattern.

**Noise (`Normal(0, 12^2)`)**

Adds realistic random variation around the structured signal. The standard deviation is intentionally moderate so the temporal pattern remains visible.

**Events**

Rare spikes and drops make the problem less perfectly smooth and provide realistic candidate cases for the later failure-analysis requirement.

**Non-negativity**

Demand cannot be negative, so the final `max(0, ...)` enforces that physical constraint.

## Fixed configuration

- 365 days
- 24 observations per day
- 8760 observations total
- hourly frequency
- primary seed: 42
- baseline: 300
- trend: 0.005 per hour
- daily sine amplitude: 45
- daily cosine amplitude: 20
- noise standard deviation: 12
- spike probability: 0.005 per hour
- spike magnitude: Uniform(100, 220)
- drop probability: 0.003 per hour
- drop magnitude: Uniform(80, 140)

## Auditability

The generator stores the individual components (`baseline`, `trend`, `daily_seasonality`, `weekly_effect`, `event`, and `noise`) in the output. This is deliberate: it lets us verify the equation directly and later explain why a particular demand value was produced.

## Initial seed-42 sanity check

The generated 8760-hour series had:

- mean demand: 328.20
- standard deviation: 44.79
- minimum demand: 90.63
- maximum demand: 587.19
- spikes: 51
- drops: 27

The sample autocorrelations were:

- lag 1 hour: 0.7952
- lag 24 hours: 0.7120
- lag 168 hours: 0.8236

These are sanity checks that the generated series contains meaningful temporal dependence. They are not model-performance results.
