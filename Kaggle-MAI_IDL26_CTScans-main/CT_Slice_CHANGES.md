# Changes made to `CT_Slice.ipynb`

Summary of edits to improve the Kaggle MSE score (lower is better).  
Approximate result: **~0.71 → ~0.2** on the public leaderboard.

---

## 1. Target scaling (normalisation)

**Before:** `TARGET` was used directly as the label.

**After:**
- Compute `y_mean` and `y_std` from the full training set
- Train on scaled labels: `y_train_scaled = (y_train_raw - y_mean) / y_std`
- After prediction, convert back: `predictions = predictions * y_std + y_mean`

**Why:** The network learns more easily when the output is roughly mean 0 and std 1.

---

## 2. Model architecture

**Before:**
- Hidden layers: `[256, 128, 64]`
- Only `Linear` + `ReLU`

**After:**
- Hidden layers: `[512, 256, 128]`
- Each hidden block: `Linear` → `BatchNorm1d` → `ReLU` → `Dropout(0.25)`

**Why:** More capacity, more stable training (BatchNorm), less overfitting (Dropout).

---

## 3. Training hyperparameters

| Setting | Before | After |
|---------|--------|-------|
| Epochs | 30 | **80** |
| Batch size | 64 | **256** |
| Optimiser | Adam (`lr=0.001`) | **AdamW** (`lr=0.002`, `weight_decay=1e-3`) |
| LR schedule | StepLR (halve every 10 epochs) | **CosineAnnealingLR** (`eta_min=1e-5`) |

**Why:** Longer training with a smoother learning-rate decay and weight decay usually generalises better.

---

## 4. Gradient clipping

**Added:** `torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)` inside `train_epoch`.

**Why:** Stops occasional large gradient steps from destabilising training.

---

## 5. Best-model checkpoint (early stopping by validation)

**Before:** The model from the **last epoch** was used for test predictions.

**After:**
- Track `best_val` and `best_state` during training
- After all epochs: `model.load_state_dict(best_state)`
- Predictions use the **best validation** weights, not the final epoch

**Why:** The last epoch is not always the best; validation MSE often improves earlier then worsens slightly.

---

## 6. Test predictions

**Added:**
- Unscale predictions with `y_mean` and `y_std`
- Clip to `[0, 180]` (competition body-location range)
- Save both `submission_cl.csv` and `submission.csv`

---

## 7. Training log output

**Before:** Progress printed every 5 epochs.

**After:** Progress every 10 epochs, including running **best** validation MSE; final lines report best val MSE and RMSE in TARGET units.

---

## What was *not* changed

- Data files and feature columns (`value0` … `value383`)
- 80/20 train/validation split (`random_split`, `seed=0`)
- Feature z-score normalisation on full training data (same as before)
- Loss function: `nn.MSELoss()`
- Overall notebook structure and flow
