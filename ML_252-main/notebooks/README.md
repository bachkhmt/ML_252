# Notebooks

Thư mục này chứa các Jupyter notebooks cho dự án Machine Learning.

## Danh sách Notebooks

### 1. `01_EDA.py` - Exploratory Data Analysis
- Phân tích và khám phá dữ liệu
- Kiểm tra missing values, outliers
- Phân tích phân phối features
- Phân tích correlation
- Visualizations

### 2. `02_Preprocessing.py` - Data Preprocessing
- Xử lý missing values (mean, median, KNN imputation)
- Encoding categorical features (one-hot, label, target encoding)
- Feature scaling (standard, minmax, robust)
- Train-test split
- Lưu preprocessed data

### 3. `03_Traditional_ML.py` - Traditional Machine Learning
- Feature engineering
- Training các mô hình truyền thống:
  - Logistic Regression
  - SVM
  - Random Forest
  - XGBoost
  - LightGBM
- Model evaluation và comparison
- Hyperparameter tuning
- Visualizations (confusion matrix, ROC curve, feature importance)

### 4. `04_Deep_Learning.py` - Deep Learning (Bonus)
- MLP (Multi-Layer Perceptron)
- TabNet
- So sánh với traditional ML
- Feature importance từ deep learning models

## Cách sử dụng

### Option 1: Chạy file Python trực tiếp
```bash
# Di chuyển đến thư mục notebooks
cd notebooks

# Chạy file Python
python 01_EDA.py
python 02_Preprocessing.py
python 03_Traditional_ML.py
python 04_Deep_Learning.py
```

### Option 2: Convert sang Jupyter Notebook
```bash
# Cài đặt jupytext (nếu chưa có)
pip install jupytext

# Convert .py sang .ipynb
jupytext --to notebook 01_EDA.py
jupytext --to notebook 02_Preprocessing.py
jupytext --to notebook 03_Traditional_ML.py
jupytext --to notebook 04_Deep_Learning.py

# Hoặc convert tất cả
jupytext --to notebook *.py
```

### Option 3: Sử dụng với VS Code
VS Code có thể chạy file .py với cell markers (`# %%`) như một notebook.

1. Mở file .py trong VS Code
2. Click "Run Cell" hoặc "Run All Cells"
3. Kết quả sẽ hiển thị trong Interactive Window

### Option 4: Mở trong Jupyter Lab/Notebook
```bash
# Khởi động Jupyter
jupyter notebook

# Hoặc Jupyter Lab
jupyter lab
```

## Thứ tự thực hiện

1. **01_EDA.py**: Bắt đầu với EDA để hiểu dữ liệu
2. **02_Preprocessing.py**: Tiền xử lý và chuẩn bị dữ liệu
3. **03_Traditional_ML.py**: Train và evaluate các mô hình truyền thống
4. **04_Deep_Learning.py** (Optional): Thử nghiệm deep learning models

## Lưu ý

- Các file hiện tại là templates với code được comment
- Bạn cần uncomment và điều chỉnh code theo dataset của bạn
- Đặc biệt chú ý đến:
  - Đường dẫn file dataset
  - Tên cột target
  - Các hyperparameters

## Data Flow

```
Raw Data
    ↓
01_EDA.py (Explore)
    ↓
02_Preprocessing.py (Clean & Transform)
    ↓
Preprocessed Data (saved to /features)
    ↓
03_Traditional_ML.py (Train & Evaluate)
    ↓
Best Model (saved to /models)
    ↓
04_Deep_Learning.py (Optional - Compare with DL)
```

## Output Files

Notebooks sẽ tạo ra các files sau:
- `/features/`: Preprocessed features (X_train, X_test, y_train, y_test)
- `/models/`: Trained models và preprocessor
- `/reports/figures/`: Các biểu đồ và visualizations
- `/reports/model_results.csv`: Kết quả đánh giá các models