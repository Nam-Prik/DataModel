# Modeling

---

## 7.1 Model Selection

Three tree-based gradient boosting models are compared, chosen because they are the industry standard for tabular classification tasks.

| Model | Mechanism | Strength |
|-------|-----------|----------|
| **XGBoost** | Gradient boosting on decision trees — builds trees sequentially where each tree corrects the errors of the previous one | Best overall for tabular data; handles mixed feature types and ordinal features natively |
| **LightGBM** | Gradient boosting like XGBoost but uses histogram binning for faster processing and lower memory usage | Nearly equal performance to XGBoost with significantly faster training time |
| **Random Forest** | Builds many independent trees and averages their predictions (bagged ensemble of uncorrelated trees) | More robust to overfitting; serves as a strong baseline |

**Why only tree-based models?**
All three models find split thresholds on features — they are scale-invariant and require no standardization. They handle the mix of binary, ordinal, continuous, and one-hot encoded features in our 27-feature matrix without any additional transformation.

---

## 7.2 Cross-Validation

### Process Order

The test set is locked away **before any model decisions are made** to prevent data leakage:

```
1. Split: 80% train / 20% test  ← test set locked here, not touched until evaluation
2. CV comparison on TRAIN only  → pick best model + feature set
3. Optuna on TRAIN only         → pick best hyperparameters
4. Final model trains on TRAIN only
5. Evaluate on TEST             ← first and only time test set is used
```

---

### Three Feature Sets

Three progressively richer feature sets are compared to measure what each engineering layer actually contributes:

| Set | Description | Features |
|-----|-------------|:--------:|
| **A** | Raw survey features only — binary, ordinal, one-hot encoding. No SMOTE recovery, no composites. | 14 |
| **B** | Set A + SMOTE integer recovery (_int versions replace raw floats) | 19 |
| **C** | Set B + all engineered composites + age_group (full feature matrix) | 27 |

**Why compare feature sets?**

| Result | Interpretation |
|--------|----------------|
| F1(C) >> F1(A) | Feature engineering genuinely added signal |
| F1(C) ≈ F1(A) | Composites added noise — raw features are sufficient, simplify the model |
| F1(B) >> F1(A) | SMOTE integer recovery was the most important step |

---

### What is K-Fold Cross-Validation?

Instead of one train/validation split (which can be lucky or unlucky), we split the training data into **K = 5 folds** and rotate which fold is used for validation:

```
Round 1: train on folds [2,3,4,5] → validate on fold [1]
Round 2: train on folds [1,3,4,5] → validate on fold [2]
Round 3: train on folds [1,2,4,5] → validate on fold [3]
Round 4: train on folds [1,2,3,5] → validate on fold [4]
Round 5: train on folds [1,2,3,4] → validate on fold [5]

Final score = average of 5 validation scores
```

**Why 5-fold is better than a single split:**

A single 80/20 split might accidentally put easy cases in the test set or hard cases in training — making the score artificially high or low by luck. Five-fold CV gives 5 independent performance estimates and averages them, removing this luck factor. The standard deviation across folds also reveals how stable the model is.

**Stratified K-Fold** is used (`StratifiedKFold`) — each fold preserves the same class distribution as the full dataset. This is critical for a 7-class problem where classes can be imbalanced.

---

### Evaluation Metric: Macro F1

**F1 score** is the harmonic mean of Precision and Recall:

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

Where:
- **Precision** = of all rows predicted as class X, how many actually are class X
- **Recall** = of all rows that are actually class X, how many did the model find

**Macro F1** averages the F1 score across all 7 classes equally:

$$F1_{macro} = \frac{1}{7} \sum_{i=1}^{7} F1_i$$

**Why Macro F1 and not Accuracy?**

Accuracy rewards getting the large classes right and ignores small ones. Macro F1 penalises the model equally for getting any class wrong — including classes with fewer rows. This is appropriate for a 7-class problem where adjacent classes (e.g. Overweight_I vs Overweight_II) are genuinely hard to separate.

---

## Results

### Dataset Split

| | Rows | Share |
|--|------|-------|
| **Training set** | 1,688 | 80% |
| **Test set (locked)** | 423 | 20% |
| Total | 2,111 | 100% |

---

### 9-Way Comparison: 3 Models × 3 Feature Sets

*(All scores are Macro F1 on training CV only — test set not touched)*

| Model | Feature Set | Features | Mean F1 | Std |
|-------|-------------|:--------:|--------:|----:|
| XGBoost | A (raw) | 14 | 0.6942 | ±0.0170 |
| XGBoost | **B (int)** | 19 | **0.7975** | ±0.0181 |
| XGBoost | C (full) | 27 | 0.7890 | ±0.0178 |
| LightGBM | A (raw) | 14 | 0.6696 | ±0.0266 |
| LightGBM | B (int) | 19 | 0.7902 | ±0.0135 |
| LightGBM | C (full) | 27 | 0.7955 | ±0.0061 |
| Random Forest | A (raw) | 14 | 0.6937 | ±0.0155 |
| Random Forest | B (int) | 19 | 0.7928 | ±0.0194 |
| Random Forest | C (full) | 27 | 0.7906 | ±0.0205 |

**Best combination: XGBoost + Feature Set B (int) → F1 = 0.7975**

---

### Reading the Results

**1. Feature set progression (A → B → C):**

| Transition | F1 change (XGBoost) | Meaning |
|-----------|:-------------------:|---------|
| A → B | +0.1033 | SMOTE integer recovery was the most impactful single step |
| B → C | −0.0085 | Composites slightly hurt XGBoost (added noise relative to B) |
| A → B | +0.1206 | LightGBM improved even more from SMOTE recovery |
| B → C | +0.0053 | Composites gave LightGBM a small additional gain |

**Key finding:** The jump from Set A to Set B (SMOTE integer recovery) is by far the largest gain — ~0.10 F1 points. Recovering the ordinal survey columns from SMOTE floats back to integers was the most important preprocessing step.

**2. Composites — mixed results:**
- XGBoost: Set C slightly worse than B (−0.009) — composites added marginal noise for XGBoost
- LightGBM: Set C slightly better than B (+0.005) — composites helped LightGBM marginally
- Random Forest: Set C slightly worse than B (−0.002) — essentially equal

The composites did not dramatically improve any model. Their main value may be in reducing the number of splits a tree needs (one composite feature vs. two raw features), but the trees found the patterns in Set B already.

**3. Model comparison at Set C:**
All three models achieve nearly identical F1 at Set C (0.789–0.796), confirming that feature quality matters more than model choice for this dataset.

**4. Performance interpretation:**

| F1 Range | Interpretation |
|----------|---------------|
| < 0.65 | Weak — raw features alone without SMOTE recovery (Set A baseline) |
| 0.65 – 0.75 | Expected baseline for 7-class behavioral prediction without height/weight |
| **0.75 – 0.82** | **Good — our models achieve 0.79–0.80, above the baseline** |
| > 0.85 | Excellent (would suggest data leakage or very clean data) |

Achieving ~0.79 Macro F1 on 7 classes using only behavioral and lifestyle features (no height, no weight) is a strong result. The remaining error comes from adjacent classes (e.g., Overweight_I vs Overweight_II) whose behavioral profiles genuinely overlap.

**5. Stability (Std):**
- LightGBM + C has the lowest std (±0.0061) — most consistent across folds
- LightGBM + A has the highest std (±0.0266) — raw features alone are unstable for LightGBM
- Low std means the score is reliable, not a lucky fold

---

### Why XGBoost + B Wins Over XGBoost + C

Adding the composites (Set B → C) slightly hurt XGBoost's performance. This happens because:

1. **Composites are correlated with their components** — `caloric_risk` contains NCP, CAEC, FAVC which are already in the feature set. XGBoost sees redundant information and may split on composites instead of the more informative raw components.
2. **SMOTE-reversed components** — `caloric_risk` contains CAEC (ρ = −0.353, SMOTE-reversed) and FAVC (ρ = +0.250). The opposite signs nearly cancel in the composite (composite ρ = +0.002), making it a weak signal.
3. **More features = more overfitting risk** — 27 features vs 19 features with the same number of trees gives more chance to overfit on the training folds.

The winner (XGBoost + B) goes forward to hyperparameter tuning with Optuna.