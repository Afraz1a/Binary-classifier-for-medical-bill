# High-Cost Healthcare Member Prediction
### SOFTEC '26 — Machine Learning Competition

Predicts which healthcare members will incur high costs in the following year using a LightGBM + XGBoost ensemble model trained on historical claims and utilization data.

---

## Problem Statement

Healthcare organizations face significant financial and operational challenges when high-cost members — those requiring expensive treatments, frequent hospitalizations, or complex care — go unidentified until it's too late to intervene.

Given **historical healthcare utilization and claims data**, the goal is to predict which members will become **high-cost in the next year**. This is a binary classification task:

- `HighCostLabel = 1` — member will be high-cost next year
- `HighCostLabel = 0` — member will not

Early identification allows care managers to proactively intervene, reducing costs and improving patient outcomes.

---

## Results

Metrics evaluated on the out-of-fold (OOF) validation set:

| Metric | Score |
|---|---|
| ROC-AUC | 0.9445 |
| PR-AUC | 0.7710 |
| F1 Score | 0.7043 |
| Precision | 0.7185 |
| Recall | 0.6906 |
| Threshold | 0.89 |

The model achieves a **ROC-AUC of 0.9445**, indicating strong discrimination between high-cost and low-cost members. The tuned threshold of 0.89 reflects the class imbalance in the dataset — pushing confidence high before labelling a member as high-cost keeps precision strong (71.9%) while maintaining reasonable recall (69.1%).

---

## Approach

### 1. Feature Engineering

Raw claims and utilization data was transformed into a rich feature set (~380 features) covering:

- **Utilization patterns** — inpatient, outpatient, home health, hospice visit counts
- **Cost aggregations** — total cost, provider charges, per-claim-type breakdowns
- **Risk scores** — Member HCC (Hierarchical Condition Category) scores, a standard measure of predicted healthcare cost
- **Temporal features** — year/month-level trends in spending and utilization
- **Demographics** — age buckets, member identifiers

### 2. Models

Two gradient boosted tree models were trained independently and combined:

| Model | Strengths |
|---|---|
| LightGBM | Fast training, handles large feature sets efficiently, low memory usage |
| XGBoost | Strong regularization, robust to overfitting, different error patterns to LGBM |

Both models were trained on the same feature set with independently tuned hyperparameters.

### 3. Ensemble Strategy — Rank Averaging

Rather than simply averaging probabilities, the ensemble uses **rank averaging**:

1. Each model's predictions are converted to percentile ranks (0–1)
2. The ranks are averaged across both models
3. This is more robust than probability averaging because it neutralizes scale differences between the two models

```python
def rank_average(proba_list):
    n = len(proba_list[0])
    ranks = np.zeros(n)
    for p in proba_list:
        order    = np.argsort(p)
        r        = np.empty_like(order, dtype=float)
        r[order] = np.arange(1, n + 1) / n
        ranks   += r
    return ranks / len(proba_list)
```

### 4. Threshold Tuning

The default classification threshold of 0.5 is suboptimal for imbalanced healthcare data. The threshold was tuned on the validation set to **maximize F1 score**, balancing precision and recall. The optimal threshold of **0.89** was saved to `metrics_summary.json` and is automatically loaded at inference time.

---

## Project Structure

```
softec-26-machine-learning-competition/
│
├── output/
│   ├── lgbm_model.pkl          # Trained LightGBM model
│   ├── xgb_model.pkl           # Trained XGBoost model
│   ├── metrics_summary.json    # Validation metrics + optimal threshold
│   ├── features_test.csv       # Engineered features for test set
│   └── predictions.csv         # Final predictions output
│
├── predict.py                  # Run inference on new data
└── evaluate_model.py           # Evaluate model with full metrics report
```

---

## Setup

**Requirements:** Python 3.10+

```bash
pip install lightgbm xgboost scikit-learn pandas numpy matplotlib joblib
```

---

## Usage

### Generate Predictions

```bash
python predict.py
```

Edit the `CONFIG` section at the top of `predict.py`:

```python
TEST_CSV     = "path/to/features_test.csv"
LGBM_MODEL   = "path/to/lgbm_model.pkl"
XGB_MODEL    = "path/to/xgb_model.pkl"
METRICS_JSON = "path/to/metrics_summary.json"
OUTPUT_DIR   = "path/to/output/"
```

Output saved to `predictions.csv`:

| Column | Description |
|---|---|
| `Member_Key` | Member identifier |
| `LGBM_Proba` | LightGBM predicted probability |
| `XGB_Proba` | XGBoost predicted probability |
| `Ensemble_Proba` | Rank-averaged ensemble probability |
| `PredictedLabel` | Final binary prediction (0 = low cost, 1 = high cost) |

---

### Evaluate on Labelled Data

```bash
python evaluate_model.py
```

Prints a full metrics report and saves ROC + Precision-Recall curve plots to the output directory:

```
ROC-AUC        : 0.94447
PR-AUC         : 0.77096
F1 Score       : 0.70428
Precision      : 0.71855
Recall         : 0.69056
Threshold used : 0.8900

Confusion Matrix (rows=Actual, cols=Predicted):
           Pred:0   Pred:1
Actual:0     TN       FP    (TN / FP)
Actual:1     FN       TP    (FN / TP)
```

---

## Key Design Decisions

**Why LightGBM + XGBoost?**
Both are state-of-the-art for tabular data. Using two models with different inductive biases means their errors don't fully overlap — the ensemble catches cases either model alone would miss.

**Why rank averaging over probability averaging?**
LightGBM and XGBoost produce probabilities on different internal scales. Rank averaging normalizes both to the same 0–1 range before combining, giving a fairer and more stable blend.

**Why tune the threshold?**
Healthcare cost prediction data is heavily imbalanced — high-cost members are a small minority. The default 0.5 threshold heavily favors predicting the majority class. Tuning to 0.89 pushes the model to only flag members it is highly confident about, keeping precision strong while still achieving solid recall.

**Why ~380 features?**
Healthcare utilization data is rich and multi-dimensional. Rather than aggressively reducing features upfront, the full engineered set was passed to gradient boosted trees, which perform their own implicit feature selection through the splitting process.
