# CT Slice Location Prediction

This repository contains a PyTorch solution for a Kaggle regression task on CT scan slices. The goal is to predict the `TARGET` value for each CT slice from 384 numerical image-derived features (`value0` to `value383`). The `TARGET` represents the approximate anatomical position of a slice in the body, so this is treated as a continuous regression problem rather than a classification task.

The final notebook builds a multi-layer perceptron, trains it on the labelled CT slice data, validates it with an 80/20 split, and generates a Kaggle-ready `submission.csv`.

## Dataset

The project uses three main CSV files:

| File | Description |
|---|---|
| `ct_slice_train.csv` | Training data with `ID`, 384 feature columns, and `TARGET` |
| `ct_slice_test_noLabels.csv` | Test data with `ID` and the same 384 feature columns, but no target |
| `sample_submission.csv` | Example submission format |

Dataset shapes used in the notebook:

- Training set: `42,800` rows
- Test set: `10,700` rows
- Input features: `384`
- Output: one continuous `TARGET` value

## Project Structure

```text
Kaggle/
+-- CT_Slice.ipynb
+-- Kaggle-MAI_IDL26_CTScans-main/
|   +-- CT_Slice.ipynb
|   +-- CT_Slice_CHANGES.md
|   +-- Code Explanation.docx
+-- ct_slice_train.csv
+-- ct_slice_test_noLabels.csv
+-- sample_submission.csv
+-- submission.csv
+-- submission_cl.csv
+-- ct_model.pt
```

The updated and improved notebook is located in:

```text
Kaggle-MAI_IDL26_CTScans-main/CT_Slice.ipynb
```

## Methodology

### 1. Data Loading

The notebook loads the training and test CSV files using pandas. The feature columns are selected as:

```python
feature_cols = [f"value{i}" for i in range(384)]
```

The model uses only the `value0` to `value383` columns as inputs. The `ID` column is kept only for creating the final submission file, and `TARGET` is used as the regression label during training.

### 2. Feature Normalization

All input features are normalized with z-score normalization:

```python
X_train_norm = (X_train_raw - train_mean) / train_std
X_test_norm = (X_test_raw - train_mean) / train_std
```

The mean and standard deviation are computed from the training data and then applied to both train and test data. This keeps the test set independent and avoids using test-set statistics during preprocessing.

### 3. Target Scaling

The improved version also standardizes the target values:

```python
y_train_scaled = (y_train_raw - y_mean) / y_std
```

The neural network trains on the scaled target because it is easier to optimize when the output distribution is close to mean `0` and standard deviation `1`. After prediction, the values are converted back to the original target scale:

```python
predictions = predictions * y_std + y_mean
```

### 4. Train/Validation Split

The labelled data is split into training and validation sets using an 80/20 split:

- Training samples: 80%
- Validation samples: 20%
- Random seed: `0`

The validation set is used to track generalization performance and select the best model weights.

## Model Architecture

The final model is a fully connected neural network implemented in PyTorch:

```text
Input: 384 features
Hidden layer 1: 512 units
Hidden layer 2: 256 units
Hidden layer 3: 128 units
Output: 1 regression value
```

Each hidden block uses:

```text
Linear -> BatchNorm1d -> ReLU -> Dropout
```

The final output layer is a plain linear layer because this is a regression problem.

## Training Setup

The improved notebook uses the following training configuration:

| Setting | Value |
|---|---|
| Loss function | `MSELoss` |
| Optimizer | `AdamW` |
| Learning rate | `0.002` |
| Weight decay | `1e-3` |
| Scheduler | `CosineAnnealingLR` |
| Epochs | `80` |
| Batch size | `256` |
| Dropout | `0.25` |
| Gradient clipping | max norm `5.0` |

During training, the notebook stores the model state with the best validation MSE and restores those weights before generating test predictions. This is better than simply using the final epoch, because the final epoch is not always the model with the best validation performance.

## Improvements Over the Initial Version

The updated notebook improves the earlier baseline in several ways:

- Added target scaling and inverse-scaling for predictions
- Increased model capacity from `[256, 128, 64]` to `[512, 256, 128]`
- Added `BatchNorm1d` for more stable training
- Added `Dropout(0.25)` to reduce overfitting
- Switched from `Adam` to `AdamW`
- Replaced `StepLR` with `CosineAnnealingLR`
- Increased training from `30` to `80` epochs
- Increased batch size from `64` to `256`
- Added gradient clipping
- Used the best validation checkpoint for final predictions
- Clipped final predictions to the expected CT target range `[0, 180]`

These changes improved the approximate Kaggle public leaderboard MSE from around `0.71` to around `0.2` (lower is better).

## Output Files

After running the notebook, the following files are produced:

| File | Description |
|---|---|
| `ct_model.pt` | Saved PyTorch model weights |
| `submission.csv` | Final Kaggle submission file |
| `submission_cl.csv` | Duplicate submission output kept from the notebook workflow |

The submission file has the required Kaggle format:

```text
ID,TARGET
1,23.45
2,41.82
...
```

## How to Run

1. Open the updated notebook:

```text
Kaggle-MAI_IDL26_CTScans-main/CT_Slice.ipynb
```

2. Make sure the dataset files are available in the working directory:

```text
ct_slice_train.csv
ct_slice_test_noLabels.csv
sample_submission.csv
```

3. Run all notebook cells from top to bottom.

4. Upload the generated `submission.csv` file to Kaggle.

## Requirements

The notebook uses:

```text
Python
PyTorch
pandas
NumPy
matplotlib
```

Install the main dependencies with:

```bash
pip install torch pandas numpy matplotlib
```

## Summary

This project demonstrates a complete deep learning workflow for tabular CT scan regression: loading structured medical imaging features, normalizing inputs, scaling targets, training a PyTorch MLP, validating performance, saving the best model, and generating a competition submission. The final version improves training stability and prediction quality through better preprocessing, a stronger architecture, regularization, and validation-based checkpointing.
