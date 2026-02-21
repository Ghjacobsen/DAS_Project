# Pipeline Changes — February 2026

## Summary

Major refactoring of the DAS anomaly detection pipeline to improve model generalizability, training efficiency, and reproducibility.

---

## 1. Preprocessing: Symmetric Log Normalization

**Before:** Per-file z-score normalization + bandpass filter + channel mean subtraction.

**After:** `sign(x) * log1p(|x|)` scaled by a fixed constant (~10.4).

**Why:**
- Z-score normalization destroys absolute magnitude information, making models non-transferable across DAS systems with different amplitude ranges.
- Log1p compresses the dynamic range of int16-origin data into a consistent ~[-1, 1] range regardless of the DAS instrument, enabling future transfer learning.
- The fixed scale constant (`log1p(32768) ≈ 10.4`) means the same normalization applies identically to any dataset without needing to compute per-file statistics.

## 2. Removed Bandpass Filter & Channel Mean Subtraction

**Why:**
- Bandpass filtering removes frequency content that the autoencoder could learn to use for distinguishing normal vs. anomalous signals.
- Channel mean subtraction removes hardware-specific artifacts that the autoencoder can learn to reconstruct as "normal background" — they are repetitive and deterministic.
- Both steps make the model dependent on specific preprocessing, reducing transferability.
- Bandpass filtering may be reintroduced in **post-processing** to classify anomaly types by frequency signature if needed.

## 3. Optuna Hyperparameter Search (replaces Grid Search)

**Before:** Grid search over 2 learning rates × 2 latent dims = 4 trials.

**After:** Optuna with Bayesian TPE sampling, configurable trial count, and automatic pruning.

**Why:**
- Grid search scales combinatorially and wastes GPU time on unpromising regions.
- Optuna's MedianPruner kills bad trials early (after 3 warmup epochs), saving significant compute.
- The search space now covers continuous ranges (LR: 5e-5 to 5e-3, latent dim: 32–256) instead of fixed lists.
- Grid search is still available as a fallback via `config.yaml` (`method: "grid"`).

## 4. Loss Function Selection

**Before:** Hard-coded MSE.

**After:** Configurable via `config.yaml` — supports L1 (MAE), Huber, and MSE. Default: **L1**.

**Why:**
- MSE disproportionately penalizes large errors, incentivizing the model to reconstruct anomalies well. This is counterproductive — we want the model to be good at normal data and *bad* at anomalies.
- L1 loss applies linear penalty, reducing the model's incentive to perfectly reconstruct large-amplitude anomalous events. This produces larger residuals during inference, improving detection sensitivity.
- Huber loss is available as a middle ground (MSE for small errors, L1 for large).

## 5. Loss History & Visualization

Training now saves per-epoch train and validation loss to `models/loss_history.json`. A standalone script (`scripts/plot_loss_curve.py`) reads this file and generates a loss curve plot — no model retraining required.

## Files Changed

| File | Change |
|------|--------|
| `config.yaml` | New structure: normalization, Optuna params, loss function |
| `src/dasproject/data.py` | Log1p normalization, removed bandpass/channel mean |
| `src/dasproject/loss.py` | **New** — loss function factory |
| `src/dasproject/train.py` | Optuna integration, early stopping, loss history saving |
| `main.py` | Updated imports and function calls |
| `pyproject.toml` | Added `optuna` dependency |
| `scripts/plot_loss_curve.py` | **New** — one-off loss curve plotter |
| `batch/plot_loss_curve.sh` | **New** — batch script for plotting |
