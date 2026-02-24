# CaoKhoa - Project Notes

Tài liệu tóm tắt cho notebook `notebooks/CaoKhoa.ipynb` theo đúng flow mục của project.

## Mục lục
- [0. Setup Environment and Download Data](#0-setup-environment-and-download-data)
  - [0.1 Setup Environment](#01-setup-environment)
  - [0.2 Download Data](#02-download-data)
  - [0.3 Load data](#03-load-data)
- [1. EDA](#1-eda)
  - [1.1 Info + describe](#11-info--describe)
  - [1.2 Target distribution](#12-target-distribution)
  - [1.3 Missing values (train/test)](#13-missing-values-traintest)
  - [1.4 Categorical vs Numeric + split X/y](#14-categorical-vs-numeric--split-xy)
  - [1.5 Insight theo nhóm categorical](#15-insight-theo-nhóm-categorical)
  - [1.6 Histogram cột số](#16-histogram-cột-số)
  - [1.7 Correlation numeric vs target](#17-correlation-numeric-vs-target)
- [2. Traditional ML](#2-traditional-ml)
  - [2.1 Preprocess + Pipeline](#21-preprocess--pipeline)
  - [2.2 Baseline + Model Zoo + Selection Log](#22-baseline--model-zoo--selection-log)
    - [2.2.1 Baseline](#221-baseline)
    - [2.2.2 Model Zoo (CV)](#222-model-zoo-cv)
    - [2.2.3 Model selection log](#223-model-selection-log)
  - [2.3 Tuning model được chọn](#23-tuning-model-được-chọn)
  - [2.4 Final: train full -> evaluate test](#24-final-train-full---evaluate-test)
  - [2.5 Save outputs](#25-save-outputs)
    - [2.5.1 Save model pipeline](#251-save-model-pipeline)
    - [2.5.2 Save EDA / selection tables](#252-save-eda--selection-tables)
    - [2.5.3 Save transformed features](#253-save-transformed-features)
    - [2.5.4 Save feature names](#254-save-feature-names)

---

# 0. Setup Environment and Download Data

## 0.1 Setup Environment
- Khai báo tham số chạy: `--out_dir`, `--data_dir`, `--fast`, `--seed`.
- Thiết lập random seed toàn cục để tái lập kết quả.
- Tạo sẵn các thư mục output:
  - `outputs/artifacts/`
  - `outputs/features/`
  - `outputs/eda_outputs/`

## 0.2 Download Data
- Tải dataset từ Kaggle (qua `kagglehub`) nếu chưa có file local.
- Copy dữ liệu về thư mục `data/`.

## 0.3 Load data
- Đọc `train.csv` và `test.csv`.
- Fallback tìm file chứa từ khóa `train`/`test` nếu tên file khác chuẩn.

# 1. EDA

## 1.1 Info + describe
- Xem cấu trúc cột, kiểu dữ liệu, thống kê mô tả ban đầu.

## 1.2 Target distribution
- Kiểm tra nhãn mục tiêu `satisfaction`.
- Mapping nhãn:
  - `neutral or dissatisfied` -> `0`
  - `satisfied` -> `1`

## 1.3 Missing values (train/test)
- Tổng hợp số lượng và tỷ lệ missing cho cả train/test.

## 1.4 Categorical vs Numeric + split X/y
- Loại bỏ cột không cần thiết (ví dụ `id`, `Unnamed: 0`).
- Tách `X_full`, `y_full`, `X_test`, `y_test`.
- Xác định danh sách cột categorical và numeric.

## 1.5 Insight theo nhóm categorical
- Tính phân phối nhãn theo từng nhóm categorical.

## 1.6 Histogram cột số
- Vẽ histogram cho các biến số quan trọng.

## 1.7 Correlation numeric vs target
- Tính tương quan giữa biến số và biến mục tiêu.

# 2. Traditional ML

## 2.1 Preprocess + Pipeline
- Xây dựng pipeline tiền xử lý:
  - Numeric: `SimpleImputer(median)` + scaler (`standard`/`minmax`/`none`)
  - Categorical: `SimpleImputer(most_frequent)` + `OneHotEncoder`
- Có tuỳ chọn `SafeTruncatedSVD` để giảm chiều an toàn.

## 2.2 Baseline + Model Zoo + Selection Log

### 2.2.1 Baseline
- Chạy baseline với `DummyClassifier`.

### 2.2.2 Model Zoo (CV)
- Chạy model zoo bằng CV (F1, Accuracy, ROC-AUC):
  - Logistic Regression
  - LinearSVC + Calibration
  - RandomForest
  - ExtraTrees
  - HistGradientBoosting
  - LogReg + SVD
  - KNN + SVD

### 2.2.3 Model selection log
- Ghi log chọn model dựa trên tiêu chí chính: F1 (tie-break bằng ROC-AUC và độ ổn định).

## 2.3 Tuning model được chọn
- Dùng `RandomizedSearchCV` để tối ưu hyperparameters cho model được chọn.
- Khi `FAST_MODE=True` thì bỏ qua tuning.

## 2.4 Final: train full -> evaluate test
- Huấn luyện trên full train và đánh giá trên test.
- Báo cáo metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
- In `classification_report`, `confusion_matrix`, và ROC curve.

## 2.5 Save outputs

### 2.5.1 Save model pipeline
- Lưu pipeline tốt nhất:
  - `outputs/artifacts/best_pipeline.joblib`

### 2.5.2 Save EDA / selection tables
- Lưu bảng EDA/model selection:
  - `outputs/eda_outputs/missing_summary.csv`
  - `outputs/eda_outputs/model_zoo_results.csv`
  - `outputs/eda_outputs/model_selection_log.csv`

### 2.5.3 Save transformed features
- Lưu features sau transform:
  - sparse: `.npz`
  - dense: `.npy`

### 2.5.4 Save feature names
- Lưu tên feature:
  - `outputs/features/feature_names.json`

---

## Ghi chú nhanh
- Notebook nguồn: `notebooks/CaoKhoa.ipynb`
- File này là bản tóm tắt mục lục + nội dung chính để nộp/report và theo dõi tiến độ project.