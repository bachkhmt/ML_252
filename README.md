# [CO3001] BÀI TẬP LỚN HỌC MÁY - HỌC KỲ I (2025-2026)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/ml-tabular-project/blob/main/notebooks/main_notebook.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Dự án này thực hiện xây dựng một hệ thống học máy hoàn chỉnh cho **dữ liệu dạng bảng (Tabular Data)**, tuân thủ quy trình Pipeline truyền thống và mở rộng so sánh với mô hình học sâu để tối ưu hóa hiệu suất dự đoán.

---

## 👨‍🏫 Thông Tin Chung

- **Môn học:** Học Máy (CO3001)
- **Học kỳ:** I - Năm học 2025-2026
- **Giảng viên hướng dẫn:** TS. Lê Thành Sách
- **Trường:** Đại học Bách Khoa, ĐHQG-HCM
- **Bộ môn:** Khoa học Máy tính
- **Phiên bản:** v1.1

---

## 👥 Thành Viên Nhóm

| Họ và Tên | MSSV | Email | Github |
|:----------|:----:|:------|:--------|
| **Hoàng Xuân Bách** | 2352082 | bach.hoang2407khmt@hcmut.edu.vn | bachkhmt |
| **Nguyễn Việt Hùng** | 2352424 | hung.nguyensubin106@hcmut.edu.vn | hung.nguyensubin106@hcmut.edu.vn |
| **Đặng Mậu Anh Quân** | 2352983 | quan.dangmauanh@hcmut.edu.vn | dangmauanhquan@gmail.com |
| **Cao Lê Minh Khoa** | 2352550 | khoa.caoleminh@hcmut.edu.vn | khoalearningcode |
| **Phạm Hồ Minh Khoa** | 2352585 | khoa.phamhominh@hcmut.edu.vn | khoa.phamhominh@hcmut.edu.vn |

---

## 🎯 Mục Tiêu Dự Án

### Mục tiêu chính
- ✅ **Thực thi Pipeline truyền thống:** Áp dụng quy trình chuẩn từ EDA, tiền xử lý, trích xuất đặc trưng đến huấn luyện mô hình
- ✅ **Đảm bảo tính linh hoạt:** Pipeline cho phép cấu hình thay đổi các kỹ thuật Scaling, Encoding và tham số mô hình
- ✅ **Xử lý Missing Values:** Thực hành các kỹ thuật imputation (Mean, Median, KNN Imputer)
- ✅ **Xử lý Categorical Data:** Áp dụng One-Hot Encoding, Label Encoding và Target Encoding
- ✅ **Phân tích so sánh:** Đánh giá hiệu quả giữa các bộ phân loại truyền thống

### Mục tiêu mở rộng (Bonus)
- 🎁 **Pipeline học sâu:** Triển khai mô hình MLP/TabNet để so sánh với phương pháp truyền thống
- 🎁 **Feature Selection:** Thử nghiệm các kỹ thuật lựa chọn đặc trưng (RFE, Feature Importance)
- 🎁 **Hyperparameter Tuning:** Sử dụng GridSearchCV/RandomizedSearchCV để tối ưu tham số

---

## 📊 Tập Dữ Liệu

* **Tên dataset:** Adult Census Income (thành phố Census 1994)
* **Nguồn:** [UCI Machine Learning Repository - Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)
* **Kích thước:** 48,842 mẫu × 15 đặc trưng (14 Features + 1 Target)

### 📊 Đặc điểm dữ liệu
- **✓ Có missing values**: Khoảng 7.4% tổng số mẫu chứa giá trị `?` (tập trung tại `workclass`, `occupation`, `native_country`). Đây là cơ hội tốt để thực hành **Imputation** (điền giá trị thiếu).
- **✓ Có categorical features**: Gồm 8 đặc trưng dạng chữ (như `education`, `occupation`, `race`...) phục vụ thực hành **Encoding** (One-Hot hoặc Label Encoding).
- **✓ Kích thước mẫu đủ lớn**: Tổng cộng **48,842** mẫu, đảm bảo Pipeline có ý nghĩa thống kê cao.
- **✓ Bài toán**: **Binary Classification** (Phân loại nhị phân: Dự đoán thu nhập >50K hoặc <=50K).

### ⚖️ Phân phối nhãn (Target Distribution)

Dựa trên toàn bộ tập dữ liệu (gộp Train & Test):

| Class | Số lượng mẫu | Tỷ lệ (%) |
| :--- | :--- | :--- |
| **Class <=50K** | 37,155 | ~76.1% |
| **Class >50K** | 11,687 | ~23.9% |


---

## 📂 Cấu Trúc Dự Án
```
ml-tabular-project/
│
├── notebooks/                  # Jupyter/Colab Notebooks
│   ├── 01_EDA.ipynb           # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb # Data Cleaning & Transformation
│   ├── 03_Traditional_ML.ipynb# Traditional ML Pipeline
│   ├── 04_Deep_Learning.ipynb # Deep Learning Pipeline (Bonus)
│   └── main_notebook.ipynb    # Notebook tổng hợp đầy đủ
│
├── modules/                    # Python modules
│   ├── __init__.py
│   ├── preprocessing.py       # Data preprocessing utilities
│   ├── features.py            # Feature engineering & extraction
│   ├── models.py              # Model definitions
│   ├── evaluation.py          # Evaluation metrics & visualization
│   └── config.py              # Configuration parameters
│
├── features/                   # Saved features
│   ├── X_train_scaled.npy
│   ├── X_test_scaled.npy
│   └── feature_names.txt
│
├── models/                     # Saved models
│   ├── best_model_rf.pkl
│   ├── best_model_svm.pkl
│   └── scaler.pkl
│
├── reports/                    # Reports & documentation
│   ├── Báo_cáo_BTL_Nhóm_XX.pdf
│   ├── figures/               # Plots and visualizations
│   └── progress_reports/      # Weekly progress reports
│
├── data/                       # Data directory (gitignored)
│   ├── raw/                   # Original data
│   ├── processed/             # Processed data
│   └── download_data.py       # Script to download dataset
│
├── requirements.txt            # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Clone Repository
```bash
git clone https://github.com/your-username/ml-tabular-project.git
cd ml-tabular-project
```

### 2. Cài Đặt Thư Viện

#### Trên Local Machine
```bash
pip install -r requirements.txt
```

#### Trên Google Colab
```python
# Run trong cell đầu tiên của notebook
!git clone https://github.com/your-username/ml-tabular-project.git
%cd ml-tabular-project
!pip install -r requirements.txt
```

### 3. Tải Dữ Liệu
```bash
python data/download_data.py
```

Hoặc trong Colab:
```python
!python data/download_data.py
```

### 4. Chạy Pipeline

#### Option 1: Chạy từng bước
```bash
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_Preprocessing.ipynb
jupyter notebook notebooks/03_Traditional_ML.ipynb
jupyter notebook notebooks/04_Deep_Learning.ipynb
```

#### Option 2: Chạy notebook tổng hợp
```bash
jupyter notebook notebooks/main_notebook.ipynb
```

#### Option 3: Chạy trên Colab
Nhấn vào badge "Open in Colab" ở đầu README hoặc truy cập link:
```
https://colab.research.google.com/github/your-username/ml-tabular-project/blob/main/notebooks/main_notebook.ipynb
```

Sau đó chọn: `Runtime → Run all`

---

## 🔧 Yêu Cầu Hệ Thống

### Thư viện chính
```
python>=3.8
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

### Thư viện mở rộng (cho Deep Learning)
```
torch>=1.10.0
pytorch-tabnet>=3.1.1
```

---

## 📈 Pipeline Học Máy

### 1. Exploratory Data Analysis (EDA)
- Thống kê mô tả dữ liệu
- Phân tích phân phối các đặc trưng
- Phát hiện missing values và outliers
- Phân tích tương quan giữa các features
- Visualizations: histograms, boxplots, correlation heatmap

### 2. Data Preprocessing

#### 2.1 Xử lý Missing Values
```python
# Các phương pháp imputation được thử nghiệm:
- Mean Imputation (cho numeric features)
- Median Imputation (khi có outliers)
- Mode Imputation (cho categorical features)
- KNN Imputer (advanced method)
```

#### 2.2 Xử lý Categorical Features
```python
# Encoding methods:
- One-Hot Encoding (cho nominal categories)
- Label Encoding (cho ordinal categories)
- Target Encoding (cho high-cardinality features)
```

#### 2.3 Feature Scaling
```python
# Scaling methods:
- StandardScaler (z-score normalization)
- MinMaxScaler (range [0,1])
- RobustScaler (robust to outliers)
```

### 3. Feature Engineering
- Feature selection dựa trên correlation
- Dimensionality reduction với PCA (90%, 95%, 99% variance)
- Tạo polynomial features (nếu cần thiết)

### 4. Model Training & Evaluation

#### Traditional ML Models
```python
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'LightGBM': LGBMClassifier()
}
```

#### Deep Learning Models (Bonus)
```python
- MLP (Multi-Layer Perceptron)
- TabNet
- NODE (Neural Oblivious Decision Ensembles)
```

#### Metrics
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

---

## 📊 Kết Quả Thực Nghiệm

### Kết quả Traditional ML Pipeline

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|:------|:--------:|:---------:|:------:|:--------:|:-------------:|
| Logistic Regression | 0.XX | 0.XX | 0.XX | 0.XX | Xs |
| SVM | 0.XX | 0.XX | 0.XX | 0.XX | Xs |
| Random Forest | 0.XX | 0.XX | 0.XX | 0.XX | Xs |
| XGBoost | 0.XX | 0.XX | 0.XX | 0.XX | Xs |

### So sánh các phương pháp Preprocessing

| Imputation | Encoding | Scaling | Best Model | Accuracy |
|:-----------|:---------|:--------|:-----------|:--------:|
| Mean | One-Hot | Standard | Random Forest | 0.XX |
| Median | Label | MinMax | XGBoost | 0.XX |
| KNN | Target | Robust | SVM | 0.XX |

### Kết quả Deep Learning (Bonus)

| Model | Accuracy | F1-Score | Parameters | Training Time |
|:------|:--------:|:--------:|:----------:|:-------------:|
| MLP | 0.XX | 0.XX | XXk | XXs |
| TabNet | 0.XX | 0.XX | XXk | XXs |

**Nhận xét:** [Thêm phân tích chi tiết về kết quả]

---

## 🎓 Bài Học Kinh Nghiệm

### Những gì học được
1. ✅ Quy trình xây dựng pipeline ML hoàn chỉnh
2. ✅ Kỹ thuật xử lý missing values và categorical data
3. ✅ So sánh và đánh giá nhiều mô hình khác nhau
4. ✅ Kỹ năng làm việc nhóm và quản lý dự án

### Thách thức gặp phải
1. ⚠️ [Mô tả thách thức 1 và cách giải quyết]
2. ⚠️ [Mô tả thách thức 2 và cách giải quyết]

### Hướng phát triển
1. 🔮 Thử nghiệm thêm các mô hình ensemble
2. 🔮 Tối ưu hyperparameters với Bayesian Optimization
3. 🔮 Deploy mô hình lên web application

---

## 📝 Phân Công Công Việc

| Thành viên | Nhiệm vụ chi tiết | Tỷ lệ đóng góp |
| :--- | :--- | :---: |
| **Hoàng Xuân Bách** | **EDA & Trực quan hóa**: Thực hiện thống kê mô tả, vẽ biểu đồ phân phối, phát hiện outliers và phân tích tương quan các thuộc tính. | 100% |
| **Nguyễn Việt Hùng** | **Tiền xử lý dữ liệu**: Xử lý dữ liệu thiếu (imputation), mã hóa biến phân loại (encoding), và chuẩn hóa dữ liệu (scaling). | 100% |
| **Đặng Mậu Anh Quân** | **Kỹ thuật đặc trưng (Feature Engineering)**: Lựa chọn đặc trưng, thực hiện giảm chiều dữ liệu (PCA) và chuẩn bị các tập dữ liệu cho huấn luyện. | 100% |
| **Cao Lê Minh Khoa** | **Huấn luyện mô hình**: Triển khai các thuật toán (Logistic Regression, SVM, Random Forest...) và thực hiện tinh chỉnh tham số (Hyperparameter tuning). | 100% |
| **Phạm Hồ Minh Khoa** | **Đánh giá & Báo cáo**: Tính toán các chỉ số (Accuracy, F1-score), so sánh các cấu hình pipeline và tổng hợp báo cáo PDF hoàn chỉnh. | 100% |

### Minh chứng làm việc nhóm
- 📸 [Folder ảnh họp nhóm](reports/team_evidence/)
- 📝 [Biên bản họp nhóm](reports/meeting_minutes/)
- 💬 [Chat log thảo luận](reports/discussion_logs/)

---

## 📚 Tài Liệu Tham Khảo

1. **Scikit-learn Documentation**: [https://scikit-learn.org/](https://scikit-learn.org/)
2. **Pandas Documentation**: [https://pandas.pydata.org/](https://pandas.pydata.org/)
3. **Adult Census Income Dataset**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult)
4. **Course Materials**: [Tài liệu môn Học Máy (CO3001) - BK ELearning](https://lms.hcmut.edu.vn/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Cảm ơn TS. Lê Thành Sách đã hướng dẫn tận tình
- Cảm ơn Bộ môn Khoa học Máy tính, Trường ĐH Bách Khoa TP.HCM
- Cảm ơn cộng đồng Kaggle và các nguồn dữ liệu mở

---

## 📧 Liên Hệ

---
