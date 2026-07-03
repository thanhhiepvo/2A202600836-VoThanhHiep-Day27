# Reflection

**Which fault types were hardest to catch, and why?**

The hardest faults were the **subtle-tier** instances that still fall inside the
published `ctx.baseline` bounds. In particular:

- **`distribution_shift` / `volume_spike` / `volume_drop` on `data_batch`** — row
  count, mean amount, null rate, and staleness can all look normal individually,
  so a single static threshold misses them.
- **`runtime_anomaly` on `lineage_run`** — duration stays below
  `lineage_duration_ms_max` but above the recent stream mean; detecting this
  required keeping a rolling duration history in `ctx.state`.
- **`corpus_staleness` / `embedding_drift` on `embedding_batch`** — centroid
  shift and document age can sit just under the global max while still being
  abnormal relative to earlier batches in the same run.
- **`missing_upstream` on `lineage_run`** — not a magnitude issue at all; the
  upstream list simply omits `raw.customers`, which only shows up in the graph
  slice, not in the baseline constants.

Obvious-tier faults (schema breaks, large volume spikes, clear feature skew) were
straightforward once the right tool was called and compared to baseline.

**What would you change about your cost/coverage tradeoff, if you had another pass?**

I would add a **two-stage policy** per pillar: a cheap baseline pass first, then
stream-relative checks only when the baseline is inconclusive. That would cut
false positives from blanket subtle thresholds (e.g. `null_rate > 0.0075`) on
clean events that happen to run slightly high.

I would also invest more in **multi-signal scoring for `data_batch`** — combining
z-scores on `std_amount` and `mean_amount` over `ctx.state` — instead of fixed
secondary thresholds, to catch distribution shifts without alerting on every
mild null-rate bump. The final detector uses rolling staleness z-scores and
embedding-age spikes in `ctx.state`, which raised the private score from 39 to
~42.6; the remaining misses are mostly first-event or near-boundary subtle faults
where no generalizable rule beats the FPR cost.
