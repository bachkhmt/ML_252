# [CO3001] BÀI TẬP LỚN HỌC MÁY - HỌC KỲ II (2025-2026)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/ml-tabular-project/blob/main/notebooks/main_notebook_final.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Dự án xây dựng hệ thống học máy hoàn chỉnh cho **dữ liệu dạng bảng (Tabular Data)** — dự đoán mức độ hài lòng của hành khách hàng không — theo đúng quy trình pipeline truyền thống kết hợp mở rộng so sánh với nhiều kiến trúc học sâu.

---

## Thông Tin Chung

| | |
|---|---|
| **Môn học** | Học Máy (CO3001) |
| **Học kỳ** | II — Năm học 2025–2026 |
| **Giảng viên hướng dẫn** | TS. Lê Thành Sách |
| **Trường** | Đại học Bách Khoa, ĐHQG-HCM |
| **Bộ môn** | Khoa học Máy tính |
| **Phiên bản** | v1.1 |

---

## Thành Viên Nhóm

| Họ và Tên | MSSV | Email | Github |
|:----------|:----:|:------|:--------|
| **Hoàng Xuân Bách** | 2352082 | bach.hoang2407khmt@hcmut.edu.vn | bachkhmt |
| **Nguyễn Việt Hùng** | 2352424 | hung.nguyensubin106@hcmut.edu.vn | hung.nguyensubin106 |
| **Đặng Mậu Anh Quân** | 2352983 | quan.dangmauanh@hcmut.edu.vn | dangmauanhquan |
| **Cao Lê Minh Khoa** | 2352550 | khoa.caoleminh@hcmut.edu.vn | khoalearningcode |
| **Phạm Hồ Minh Khoa** | 2352585 | khoa.phamhominh@hcmut.edu.vn | khoaphamhominh |

---

## 🎯 Mục Tiêu Dự Án

### Mục tiêu chính (bắt buộc)
-  **EDA chuyên sâu:** Data profiling, univariate / bivariate / multivariate analysis, mutual information ranking
-  **Pipeline linh hoạt, cấu hình được:** Scaling (StandardScaler / MinMaxScaler / None), giảm chiều (TruncatedSVD, cấu hình n_components), lựa chọn model
-  **Xử lý Missing Values:** SimpleImputer với strategy median (numeric) và most_frequent (categorical)
-  **Encoding Categorical:** OneHotEncoder với handle_unknown='ignore'
-  **Model Zoo đa dạng:** 8 cấu hình model với CV 5-fold, selection log minh bạch
-  **Hyperparameter Tuning:** RandomizedSearchCV với param space riêng từng model
-  **Đánh giá đầy đủ:** Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, ROC Curve

### Mục tiêu mở rộng (bonus)
-  **Deep Learning Pipeline:** 6 kiến trúc (MLP, MLP+BN+Dropout, ResMLP, Wide&Deep, TabNetLite, FT-Transformer) — so sánh trực tiếp với traditional ML
-  **Feature Importance (EDA):** Mutual Information ranking trước khi train
-  **GitHub Repository:** README đầy đủ, cấu trúc thư mục rõ ràng, code minh bạch

---

## Tập Dữ Liệu

- **Tên dataset:** Airline Passenger Satisfaction
- **Nguồn:** [Kaggle — teejmahal20/airline-passenger-satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
- **Tải về:** Tự động qua `kagglehub` (không cần mount Drive)
- **Kích thước:** 129,880 mẫu × 23 cột — chia sẵn train (103,904 mẫu) và test (25,976 mẫu)

### Đặc điểm dữ liệu

| Thuộc tính | Chi tiết |
|---|---|
| **Missing values** | Cột `Arrival Delay in Minutes`: 310 giá trị thiếu (~0.3%) |
| **Categorical features** | 4 cột: `Gender`, `Customer Type`, `Type of Travel`, `Class` |
| **Rating features** | 14 cột thang điểm 0–5 (wifi, seat comfort, cleanliness…) |
| **Continuous features** | `Age`, `Flight Distance`, `Departure Delay in Minutes`, `Arrival Delay in Minutes` |
| **Bài toán** | Binary Classification: `satisfied` vs `neutral or dissatisfied` |

### Phân phối nhãn

| Class | Số mẫu (train+test) | Tỷ lệ |
|---|---|---|
| neutral or dissatisfied | ~73,452 | ~56.6% |
| satisfied | ~56,428 | ~43.4% |

---

## Cấu Trúc Dự Án

```
ml-tabular-project/
│
├── notebooks/
│   └── main_notebook_final.ipynb   # Notebook tổng hợp (Run All thành công 100%)
│
├── outputs/                        # Sinh ra khi chạy notebook (gitignored)
│   ├── data/                       # train.csv, test.csv (tải từ Kaggle)
│   ├── artifacts/
│   │   └── best_pipeline.joblib    # Best traditional ML pipeline (sklearn)
│   ├── features/
│   │   ├── train_features.npy/npz  # Features sau preprocessing (traditional)
│   │   ├── test_features.npy/npz
│   │   ├── feature_names.json
│   ├── eda_outputs/
│   │   ├── model_zoo_results.csv   # CV kết quả toàn bộ model zoo
│   │   └── model_selection_log.csv # Selection log so với baseline
│   └── deep_learning/
│       ├── best_dl_model.pt        # Weights best DL model (PyTorch)
│       ├── dl_feature_pipe.joblib  # DL preprocessing pipeline
├── reports/
│   └── report.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  Hướng Dẫn Sử Dụng

### Option 1: Chạy trên Google Colab (khuyến nghị)

1. Nhấn badge **"Open in Colab"** ở đầu README
2. Chọn `Runtime → Run all`
3. Notebook tự động cài thư viện và tải dữ liệu từ Kaggle public — **không cần mount Drive**

### Option 2: Chạy trên Local Machine

```bash
# 1. Clone repository
git clone https://github.com/your-username/ml-tabular-project.git
cd ml-tabular-project

# 2. Cài đặt thư viện
pip install -r requirements.txt

# 3. Mở notebook
jupyter notebook notebooks/main_notebook_final.ipynb
```

### Cấu hình nhanh (trong notebook)

```python
FAST_MODE = False  # True → ít fold, ít model, huấn luyện nhanh hơn
RANDOM_STATE = 42
```

---

## Yêu Cầu Thư Viện

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

---

## Chi Tiết Pipeline

### Phần 0 — Setup & Data Download
- Cài `kagglehub`, tải dữ liệu từ Kaggle public (không mount Drive)
- Cấu hình seed toàn cục, paths, visual style
- Xác định vai trò từng cột: `num_cols`, `cat_cols`, `rating_cols`, `delay_cols`

### Phần 1 — EDA (10 mục)

| Mục | Nội dung |
|---|---|
| 1.1 | Data Profiling: dtype, missing %, unique, descriptive stats |
| 1.2 | Target Distribution: class balance, imbalance ratio |
| 1.3 | Categorical vs Target: stacked bar chart, satisfaction rate per category |
| 1.4 | Continuous Features: histogram + KDE + boxplot theo target |
| 1.5 | Rating Features: mean rating bar chart, rating gap (satisfied − dissatisfied) |
| 1.6 | Correlation Heatmap + Top 10 features vs target (|r|) |
| 1.7 | Multivariate: satisfaction % theo segment (categorical × categorical, age quintile) |
| 1.8 | Outlier Treatment (P99 capping) + Feature Engineering |
| 1.9 | Mutual Information ranking (supervised, trước khi train) |
| 1.10 | EDA Summary tự động (không hardcode) |

**Features mới tạo trong EDA:**
- `Total Delay` = tổng delay departure + arrival
- `Is Delayed` = binary flag (Total Delay > 0)
- `Avg Inflight Score` = trung bình các rating dịch vụ trên máy bay
- `Avg Ground Score` = trung bình các rating dịch vụ mặt đất
- `Log <col>` = log1p transform các cột skewed

### Phần 2 — Traditional ML Pipeline

#### 2.1 Configurable Preprocessor

```python
# Factory build_preprocessor(scaler: 'standard' | 'minmax' | 'none')
# Numeric: SimpleImputer(median) → Scaler
# Categorical: SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown='ignore')

# Factory make_pipeline(model, scaler, use_svd, n_components)
# Bước tùy chọn: SafeTruncatedSVD (tự động giới hạn n_components <= n_features)
```

#### 2.2 Baseline: DummyClassifier (most_frequent)

#### 2.3 Model Zoo (StratifiedKFold CV 5-fold)

| Model | Scaler | SVD |
|---|---|---|
| Logistic Regression | StandardScaler | ✗ |
| Logistic Regression | MinMaxScaler | ✗ |
| Logistic Regression | StandardScaler | SVD(32) |
| LinearSVC (CalibratedClassifierCV) | StandardScaler | ✗ |
| KNN (k=15, weights=distance) | StandardScaler | SVD(32) |
| Random Forest (n=400) | None | ✗ |
| Extra Trees (n=800) | None | ✗ |
| HistGradientBoosting | None | ✗ |

Metrics CV: Accuracy, F1, ROC-AUC — kết quả lưu vào `model_zoo_results.csv` và `model_selection_log.csv`.

#### 2.4 Hyperparameter Tuning (RandomizedSearchCV)

Param space riêng cho từng model được chọn: LogReg (C, solver), RF (n_estimators, max_depth, min_samples_split/leaf, max_features), ExtraTrees, HistGradientBoosting (learning_rate, max_depth, max_leaf_nodes), SVC (C), KNN (k, weights).

#### 2.5 Evaluation trên Test Set

Confusion Matrix + ROC Curve + Classification Report đầy đủ. Artifacts lưu: `best_pipeline.joblib`, features `.npy/.npz`, `feature_names.json`.

### Phần 3 — Deep Learning Pipeline (Bonus)

**Feature pipeline DL:** OneHot → TruncatedSVD(128) → StandardScaler → Dense vector  
**Training:** PyTorch, AdamW, BCEWithLogitsLoss, Early Stopping (patience=6), batch=1024

| Model | Kiến trúc |
|---|---|
| **MLP** | Linear(256) → ReLU → Linear(128) → ReLU → Linear(1) |
| **MLP + BN + Dropout** | Linear → BatchNorm → ReLU → Dropout(0.25) × 2 layers |
| **ResMLP** | Input projection + 3 Residual Blocks (skip connections) |
| **Wide & Deep** | Wide (linear) + Deep (2 hidden layers) — đầu ra cộng gộp |
| **TabNetLite** | 3 Feature Gating steps, mỗi step có MLP riêng |
| **FT-Transformer** | Feature tokenization + Transformer Encoder (3 layers, 8 heads) + CLS token |

Artifacts lưu: `best_dl_model.pt`, `dl_feature_pipe.joblib`, `best_dl_meta.json`, `dl_test_predictions.csv`, `dl_train_features.npy`, `dl_test_features.npy`.

### Phần 4 — Reload & Reproduce

Load lại model truyền thống (`.joblib`) và model học sâu (`.pt`) sau khi reset kernel — không cần train lại.

### Phần 5 — So Sánh Traditional vs Deep Learning

Bar chart và bảng so sánh Accuracy / Precision / Recall / F1 / ROC-AUC trên test set.

---

## Kết Quả Thực Nghiệm

> Kết quả dưới đây là tham chiếu từ pipeline đã chạy; giá trị chính xác tùy thuộc vào quá trình tuning và random seed.

### Baseline

| Metric | DummyClassifier (most_frequent) |
|---|---|
| Accuracy | ~0.5667 |
| F1 | 0.0000 |
| ROC-AUC | 0.5000 |

### Traditional ML (Model Zoo CV — F1)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| HistGradientBoosting | ~0.96 | ~0.95 | ~0.99 |
| Extra Trees | ~0.96 | ~0.95 | ~0.99 |
| Random Forest | ~0.96 | ~0.95 | ~0.99 |
| LogReg (StandardScaler) | ~0.87 | ~0.85 | ~0.94 |
| LinearSVC (calibrated) | ~0.87 | ~0.85 | ~0.94 |
| KNN(k=15) + SVD(32) | ~0.93 | ~0.92 | ~0.97 |

### Deep Learning (Test Set)

| Model | F1 | ROC-AUC |
|---|---|---|
| FT-Transformer | ~0.96 | ~0.99 |
| ResMLP | ~0.95 | ~0.99 |
| Wide & Deep | ~0.95 | ~0.99 |
| TabNetLite | ~0.94 | ~0.98 |
| MLP + BN + Dropout | ~0.95 | ~0.99 |
| MLP (baseline) | ~0.94 | ~0.98 |

**Nhận xét:** Ensemble cây quyết định (HistGradientBoosting, ExtraTrees) và các kiến trúc DL tiên tiến (FT-Transformer, ResMLP) đạt hiệu suất tương đương nhau (~95–96% F1) trên tập dữ liệu này. Model cây nhanh hơn đáng kể và không cần GPU. Mô hình tuyến tính (LogReg, LinearSVC) thua kém ~10% F1 do bài toán có nhiều tương tác phi tuyến.

---

## Phân Công Công Việc

| Thành viên | Nhiệm vụ | Đóng góp |
|---|---|:---:|
| **Hoàng Xuân Bách** | EDA chuyên sâu (mục 1.1–1.10): Data profiling, univariate/bivariate/multivariate analysis, mutual information, feature engineering | 100% |
| **Nguyễn Việt Hùng** | Tiền xử lý dữ liệu: Configurable preprocessor (imputation, encoding, scaling), baseline DummyClassifier | 100% |
| **Đặng Mậu Anh Quân** | Feature Engineering & Feature Selection: TruncatedSVD, lưu features .npy/.npz, DL feature pipeline | 100% |
| **Cao Lê Minh Khoa** | Huấn luyện Model Zoo (8 model), Model Selection Log, Hyperparameter Tuning (RandomizedSearchCV), Deep Learning (6 kiến trúc) | 100% |
| **Phạm Hồ Minh Khoa** | Đánh giá & Báo cáo: metrics đầy đủ, confusion matrix, ROC curve, Final Comparison (Phần 5), PDF report, README | 100% |

---

## Checklist Nộp Bài

| Hạng mục | Trạng thái |
|---|---|
| EDA đầy đủ (profiling, univariate, bivariate, multivariate) | Done |
| Pipeline truyền thống với scaling/SVD cấu hình được | Done |
| Model Zoo CV (8 configs) + selection log | Done |
| Hyperparameter tuning (RandomizedSearchCV) | Done |
| Evaluation: accuracy, precision, recall, F1, ROC-AUC | Done |
| Deep Learning pipeline (6 kiến trúc + comparison) | Done |
| Features lưu .npy / .npz | Done |
| Model artifacts .joblib + .pt | Done |
| Cấu trúc thư mục: outputs/{artifacts, features, eda_outputs, deep_learning} | Done |
| Runtime → Run All chạy thành công không lỗi | Done |
| Không mount Drive cá nhân | Done |
| Dataset tải từ Kaggle public (kagglehub) | Done |
| Reload & Reproduce sau khi reset kernel (Phần 4) | Done |

---

## Tài Liệu Tham Khảo

1. **Scikit-learn Documentation**: https://scikit-learn.org/
2. **PyTorch Documentation**: https://pytorch.org/docs/
3. **Airline Passenger Satisfaction Dataset**: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
4. **FT-Transformer paper**: Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data" (NeurIPS 2021)
5. **TabNet paper**: Arik & Pfister, "TabNet: Attentive Interpretable Tabular Learning" (AAAI 2021)
6. **Course Materials**: Tài liệu môn Học Máy (CO3001) — BK ELearning https://lms.hcmut.edu.vn/

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- Cảm ơn TS. Lê Thành Sách đã hướng dẫn tận tình trong suốt học kỳ
- Cảm ơn Bộ môn Khoa học Máy tính, Trường ĐH Bách Khoa TP.HCM
- Cảm ơn cộng đồng Kaggle và các tác giả dataset
