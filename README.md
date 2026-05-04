# [CO3001] Machine Learning Course Project — Semester I (2025–2026)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/ml-tabular-project/blob/main/notebooks/main_notebook_final.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This project builds a complete machine learning system for **Tabular Data** — predicting airline passenger satisfaction — following the traditional ML pipeline and extending it with a deep learning comparison.

> **Report:** [📄 View PDF Report](reports/report.pdf) &nbsp;|&nbsp; **Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/ml-tabular-project/blob/main/notebooks/main_notebook_final.ipynb)

---

## Course Information

| Field | Details |
|---|---|
| **Course Name** | Machine Learning |
| **Course Code** | CO3001 |
| **Semester** | I — Academic Year 2025–2026 |
| **Instructor** | Dr. Lê Thành Sách |
| **Department** | Department of Computer Science |
| **University** | Ho Chi Minh City University of Technology, VNU-HCM |

---

## Team Members

| Full Name | Student ID | Email |
|:----------|:----------:|:------|
| Hoàng Xuân Bách | 2352082 | bach.hoang2407khmt@hcmut.edu.vn |
| Nguyễn Việt Hùng | 2352424 | hung.nguyensubin106@hcmut.edu.vn |
| Đặng Mậu Anh Quân | 2352983 | quan.dangmauanh@hcmut.edu.vn |
| Cao Lê Minh Khoa | 2352550 | khoa.caoleminh@hcmut.edu.vn |
| Phạm Hồ Minh Khoa | 2352585 | khoa.phamhominh@hcmut.edu.vn |

---

## Project Objectives

This project aims to:

- Understand and apply the **traditional machine learning pipeline** end-to-end: EDA, data preprocessing, feature extraction, model training, and evaluation.
- Build a **configurable and flexible pipeline** that allows switching between scaling strategies (StandardScaler / MinMaxScaler / None), dimensionality reduction (TruncatedSVD with configurable components), and model selection.
- Practice handling **missing values** (median / most-frequent imputation) and **categorical encoding** (OneHotEncoder).
- Compare multiple classifiers fairly using **stratified k-fold cross-validation** and a transparent model selection log.
- Apply **hyperparameter tuning** via RandomizedSearchCV with model-specific search spaces.
- Implement a **deep learning pipeline** (6 neural architectures) and compare it against traditional ML to earn bonus points.
- Develop skills in experimentation, analysis, and scientific reporting.

---

## Dataset

- **Name:** Airline Passenger Satisfaction
- **Source:** [Kaggle — teejmahal20/airline-passenger-satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
- **Download:** Automatic via `kagglehub` — no Google Drive mounting required
- **Size:** 129,880 samples × 23 columns — pre-split into train (103,904) and test (25,976)

| Property | Details |
|---|---|
| Missing values | `Arrival Delay in Minutes`: 310 missing (~0.3%) |
| Categorical features | 4 columns: `Gender`, `Customer Type`, `Type of Travel`, `Class` |
| Rating features | 14 columns, scale 0–5 (wifi, seat comfort, cleanliness, etc.) |
| Continuous features | `Age`, `Flight Distance`, `Departure Delay in Minutes`, `Arrival Delay in Minutes` |
| Task | Binary Classification: `satisfied` vs `neutral or dissatisfied` |

**Label distribution:**

| Class | Count | Ratio |
|---|---|---|
| neutral or dissatisfied | ~73,452 | ~56.6% |
| satisfied | ~56,428 | ~43.4% |

---

## Project Folder Structure

```
ml-tabular-project/
│
├── notebooks/
│   └── main_notebook_final.ipynb   # Main notebook (Runtime → Run All: 100% error-free)
│
├── outputs/                        # Generated at runtime — gitignored
│   ├── data/                       # train.csv, test.csv (downloaded from Kaggle)
│   ├── artifacts/
│   │   └── best_pipeline.joblib    # Best traditional ML pipeline (scikit-learn)
│   ├── features/
│   │   ├── train_features.npy/npz  # Preprocessed features (traditional pipeline)
│   │   ├── test_features.npy/npz
│   │   ├── feature_names.json
│   │   ├── dl_train_features.npy   # Preprocessed features (DL pipeline)
│   │   └── dl_test_features.npy
│   ├── eda_outputs/
│   │   ├── model_zoo_results.csv   # CV results for all models
│   │   └── model_selection_log.csv # Selection log vs baseline
│   └── deep_learning/
│       ├── best_dl_model.pt        # Best DL model weights (PyTorch)
│       ├── dl_feature_pipe.joblib  # DL preprocessing pipeline
│       ├── best_dl_meta.json       # Metadata (model name, input_dim, svd_k)
│       └── dl_test_predictions.csv # Predictions + probabilities
│
├── reports/
│   └── report.pdf
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run the Notebook

### Option 1 — Google Colab (Recommended)

1. Click the **"Open in Colab"** badge at the top of this README.
2. Select `Runtime → Run all`.
3. The notebook installs all dependencies and downloads data from Kaggle automatically — **no Drive mounting needed**.

### Option 2 — Local Machine

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ml-tabular-project.git
cd ml-tabular-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook notebooks/main_notebook_final.ipynb
```

### Required Libraries

```
python>=3.8
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
matplotlib>=3.4
seaborn>=0.11
scipy>=1.7
torch>=1.10
kagglehub
joblib
jupyter
```

### Quick Configuration

Inside the notebook, two flags control runtime behavior:

```python
FAST_MODE    = False  # Set True for fewer folds and models (faster run)
RANDOM_STATE = 42
```

---

## Pipeline Overview

### Section 0 — Setup & Data Download
Installs `kagglehub`, downloads data from Kaggle public source, sets global seed, output paths, and visual style. Identifies column roles: `num_cols`, `cat_cols`, `rating_cols`, `delay_cols`.

### Section 1 — Exploratory Data Analysis (10 subsections)

| Subsection | Content |
|---|---|
| 1.1 | Data Profiling: dtypes, missing %, unique counts, descriptive statistics |
| 1.2 | Target Distribution: class balance, imbalance ratio |
| 1.3 | Categorical vs Target: count plots + stacked % bar charts |
| 1.4 | Continuous Features: histogram + KDE + boxplot by target |
| 1.5 | Rating Features: mean rating bars + satisfaction gap chart |
| 1.6 | Correlation Heatmap + Top 10 features vs target (\|r\|) |
| 1.7 | Multivariate Analysis: satisfaction rate by segment & age quintile |
| 1.8 | Outlier Treatment (P99 capping) + Feature Engineering |
| 1.9 | Mutual Information ranking (supervised feature importance pre-training) |
| 1.10 | Automated EDA Summary (no hardcoding) |

**Engineered features:**
- `Total Delay` — sum of departure and arrival delays
- `Is Delayed` — binary flag (Total Delay > 0)
- `Avg Inflight Score` — mean of in-flight service ratings
- `Avg Ground Score` — mean of ground service ratings
- `Log <col>` — log1p transform of right-skewed continuous columns

### Section 2 — Traditional ML Pipeline

**Preprocessing factory** — `build_preprocessor(scaler)`:
- Numeric: `SimpleImputer(median)` → Scaler
- Categorical: `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')`
- Scalers: `StandardScaler` | `MinMaxScaler` | passthrough

**Pipeline factory** — `make_pipeline(model, scaler, use_svd, n_components)`:
- Optional step: `SafeTruncatedSVD` (auto-clips n_components ≤ n_features)

**Baseline:** `DummyClassifier(strategy='most_frequent')`

**Model Zoo — StratifiedKFold CV (5-fold):**

| Model | Scaler | SVD |
|---|---|---|
| Logistic Regression | StandardScaler | — |
| Logistic Regression | MinMaxScaler | — |
| Logistic Regression | StandardScaler | SVD(32) |
| LinearSVC (CalibratedClassifierCV) | StandardScaler | — |
| KNN (k=15, weights=distance) | StandardScaler | SVD(32) |
| Random Forest (n=400) | None | — |
| Extra Trees (n=800) | None | — |
| HistGradientBoosting | None | — |

**Hyperparameter Tuning** — `RandomizedSearchCV` (20 iterations, 5-fold) with model-specific search spaces: C/solver for LogReg, n_estimators/max_depth/min_samples/max_features for tree ensembles, learning_rate/max_leaf_nodes for HGB, k/weights for KNN.

**Evaluation on test set:** Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, ROC Curve, Classification Report.

**Saved artifacts:** `best_pipeline.joblib`, `train_features.npy/npz`, `test_features.npy/npz`, `feature_names.json`.

### Section 3 — Deep Learning Pipeline (Bonus)

**DL feature pipeline:** OneHot → TruncatedSVD(128) → StandardScaler → dense float32 tensor

**Training config:** PyTorch, AdamW optimizer, BCEWithLogitsLoss, early stopping (patience=6), batch size=1024, 25 epochs.

| Model | Architecture |
|---|---|
| MLP | Linear(256) → ReLU → Linear(128) → ReLU → Linear(1) |
| MLP + BN + Dropout | Linear → BatchNorm → ReLU → Dropout(0.25) × 2 layers |
| ResMLP | Input projection + 3 Residual Blocks with skip connections |
| Wide & Deep | Wide branch (linear) + Deep branch (2 hidden layers) — outputs summed |
| TabNetLite | 3 Feature Gating steps, each with its own MLP |
| FT-Transformer | Feature tokenization + TransformerEncoder (3 layers, 8 heads, d_model=96) + CLS token |

**Saved artifacts:** `best_dl_model.pt`, `dl_feature_pipe.joblib`, `best_dl_meta.json`, `dl_test_predictions.csv`, `dl_train_features.npy`, `dl_test_features.npy`.

### Section 4 — Reload & Reproduce

Reloads the traditional pipeline (`.joblib`) and the best DL model (`.pt`) after a kernel reset and re-runs evaluation — no retraining required.

### Section 5 — Final Comparison: Traditional vs Deep Learning

Bar chart and summary table comparing Accuracy / Precision / Recall / F1 / ROC-AUC on the held-out test set.

---

## Experimental Results

> Values below are indicative from a completed run; exact numbers depend on tuning outcome and random seed.

### Baseline

| Metric | DummyClassifier (most_frequent) |
|---|---|
| Accuracy | ~0.5667 |
| F1 | 0.0000 |
| ROC-AUC | 0.5000 |

### Traditional ML — Model Zoo CV

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| HistGradientBoosting | ~0.96 | ~0.95 | ~0.99 |
| Extra Trees | ~0.96 | ~0.95 | ~0.99 |
| Random Forest | ~0.96 | ~0.95 | ~0.99 |
| KNN(k=15) + SVD(32) | ~0.93 | ~0.92 | ~0.97 |
| LogReg (StandardScaler) | ~0.87 | ~0.85 | ~0.94 |
| LinearSVC (calibrated) | ~0.87 | ~0.85 | ~0.94 |

### Deep Learning — Test Set

| Model | F1 | ROC-AUC |
|---|---|---|
| FT-Transformer | ~0.96 | ~0.99 |
| MLP + BN + Dropout | ~0.95 | ~0.99 |
| ResMLP | ~0.95 | ~0.99 |
| Wide & Deep | ~0.95 | ~0.99 |
| TabNetLite | ~0.94 | ~0.98 |
| MLP | ~0.94 | ~0.98 |

**Key findings:** Tree ensemble methods (HistGradientBoosting, ExtraTrees) and advanced DL architectures (FT-Transformer, ResMLP) achieve comparable performance (~95–96% F1) on this dataset. Tree-based models are significantly faster and require no GPU. Linear models (LogReg, LinearSVC) trail by ~10% F1 due to non-linear interactions in the data.

---

## Task Allocation

| Member | Responsibilities | Contribution |
|---|---|:---:|
| Hoàng Xuân Bách | In-depth EDA (Sections 1.1–1.10): data profiling, univariate/bivariate/multivariate analysis, mutual information, feature engineering | 100% |
| Nguyễn Việt Hùng | Data preprocessing: configurable preprocessor (imputation, encoding, scaling), DummyClassifier baseline | 100% |
| Đặng Mậu Anh Quân | Feature engineering & selection: TruncatedSVD, saving features to .npy/.npz, DL feature pipeline | 100% |
| Cao Lê Minh Khoa | Model Zoo training (8 configs), model selection log, hyperparameter tuning (RandomizedSearchCV), deep learning training (6 architectures) | 100% |
| Phạm Hồ Minh Khoa | Evaluation & reporting: full metrics, confusion matrix, ROC curve, final comparison (Section 5), PDF report, README | 100% |

---

## Submission Checklist

| Item | Status |
|---|---|
| Full EDA (profiling, univariate, bivariate, multivariate) | Done |
| Configurable traditional pipeline (scaling, SVD) | Done |
| Model Zoo CV (8 configs) + selection log | Done |
| Hyperparameter tuning (RandomizedSearchCV) | Done |
| Evaluation: accuracy, precision, recall, F1, ROC-AUC | Done |
| Deep Learning pipeline (6 architectures + comparison) | Done |
| Features saved as .npy / .npz | Done |
| Model artifacts saved as .joblib + .pt | Done |
| Output folder structure: artifacts, features, eda_outputs, deep_learning | Done |
| Runtime → Run All completes without errors | Done |
| No personal cloud storage mounted | Done |
| Dataset downloaded from public Kaggle source (kagglehub) | Done |
| Reload & Reproduce after kernel reset (Section 4) | Done |

---

## References

1. Scikit-learn Documentation — https://scikit-learn.org/
2. PyTorch Documentation — https://pytorch.org/docs/
3. Airline Passenger Satisfaction Dataset — https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
4. Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data", NeurIPS 2021
5. Arik & Pfister, "TabNet: Attentive Interpretable Tabular Learning", AAAI 2021
6. CO3001 Course Materials — BK E-Learning https://lms.hcmut.edu.vn/

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- Dr. Lê Thành Sách 
- Department of Computer Science, HCMUT
- Kaggle community and dataset authors
