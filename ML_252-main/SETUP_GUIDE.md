# Hướng Dẫn Setup và Sử Dụng Project

## 📦 Cấu Trúc Thư Mục Đã Tạo

```
ml-tabular-project/
│
├── modules/                    # Python modules cho preprocessing, models, evaluation
│   ├── __init__.py
│   ├── config.py              # Cấu hình toàn project
│   ├── preprocessing.py       # Data preprocessing utilities
│   ├── features.py            # Feature engineering
│   ├── models.py              # Model training
│   └── evaluation.py          # Model evaluation & visualization
│
├── notebooks/                  # Jupyter/Python notebooks
│   ├── 01_EDA.py              # Exploratory Data Analysis
│   ├── 02_Preprocessing.py    # Data preprocessing
│   ├── 03_Traditional_ML.py   # Traditional ML pipeline
│   ├── 04_Deep_Learning.py    # Deep Learning (Bonus)
│   └── README.md              # Hướng dẫn sử dụng notebooks
│
├── features/                   # Thư mục lưu preprocessed features
│   └── .gitkeep
│
├── models/                     # Thư mục lưu trained models
│   └── .gitkeep
│
├── reports/                    # Thư mục báo cáo
│   └── figures/               # Thư mục lưu visualizations
│       └── .gitkeep
│
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore file
├── LICENSE                    # MIT License
└── README.md                  # README chính của project
```

## 🚀 Bước 1: Setup Môi Trường

### A. Clone/Download Project

Nếu đã có trên GitHub:
```bash
git clone https://github.com/your-username/ml-tabular-project.git
cd ml-tabular-project
```

Hoặc giải nén file zip đã tải về.

### B. Tạo Virtual Environment (Khuyến nghị)

**Trên Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Trên macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### C. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

**Lưu ý:** Nếu muốn sử dụng deep learning (bonus), cài thêm:
```bash
pip install torch pytorch-tabnet
```

## 📊 Bước 2: Chuẩn Bị Dữ Liệu

### Tạo Thư Mục Data

```bash
mkdir -p data/raw
mkdir -p data/processed
```

### Tải Dataset Của Bạn

1. Tải dataset từ Kaggle/UCI hoặc nguồn khác
2. Đặt file vào `data/raw/`
3. Ví dụ: `data/raw/your_dataset.csv`

## 🔧 Bước 3: Chạy Pipeline

### Option 1: Sử dụng File Python (.py)

```bash
cd notebooks

# Bước 1: EDA
python 01_EDA.py

# Bước 2: Preprocessing
python 02_Preprocessing.py

# Bước 3: Train Models
python 03_Traditional_ML.py

# Bước 4 (Optional): Deep Learning
python 04_Deep_Learning.py
```

### Option 2: Convert sang Jupyter Notebook

```bash
# Cài jupytext
pip install jupytext

# Convert
cd notebooks
jupytext --to notebook *.py

# Chạy Jupyter
jupyter notebook
```

### Option 3: Sử dụng VS Code

1. Mở project trong VS Code
2. Cài extension "Python" và "Jupyter"
3. Mở file .py trong notebooks
4. Click "Run Cell" để chạy từng cell

## ⚙️ Bước 4: Tùy Chỉnh Cho Dataset Của Bạn

### A. Cập Nhật File Config

Mở `modules/config.py` và điều chỉnh:
- `TEST_SIZE`: Tỷ lệ test split
- `RANDOM_STATE`: Random seed
- `MODELS_CONFIG`: Hyperparameters của models

### B. Cập Nhật Notebooks

**Trong 01_EDA.py:**
```python
# Sửa dòng này
df = pd.read_csv('../data/raw/your_dataset.csv')
```

**Trong 02_Preprocessing.py:**
```python
# Sửa tên cột target
target_col = 'your_target_column_name'
```

**Trong các notebooks khác:**
- Uncomment các dòng code
- Điều chỉnh theo dataset của bạn

## 📝 Bước 5: Làm Việc Với Code

### Quy Trình Chung

1. **EDA (01_EDA.py)**
   - Load data
   - Uncomment code
   - Chạy từng cell
   - Phân tích kết quả

2. **Preprocessing (02_Preprocessing.py)**
   - Thử nghiệm các preprocessing methods
   - Lưu preprocessed data
   - Chọn configuration tốt nhất

3. **Training (03_Traditional_ML.py)**
   - Load preprocessed data
   - Train nhiều models
   - So sánh kết quả
   - Save best model

4. **Deep Learning (04_Deep_Learning.py)** - Optional
   - Train MLP/TabNet
   - So sánh với traditional ML

### Custom Modules

Các modules trong `modules/` có thể được import:

```python
from modules.preprocessing import DataPreprocessor
from modules.models import ModelTrainer
from modules.evaluation import ModelEvaluator
from modules.features import FeatureEngineer
```

## 📊 Bước 6: Xem Kết Quả

Sau khi chạy xong:

### Features
```
features/
├── X_train.npy
├── X_test.npy
├── y_train.npy
├── y_test.npy
└── feature_names.txt
```

### Models
```
models/
├── preprocessor.pkl
├── best_model_RandomForest.pkl
└── ...
```

### Reports
```
reports/
├── model_results.csv
└── figures/
    ├── model_comparison.png
    ├── confusion_matrix_*.png
    ├── roc_curve_*.png
    └── feature_importance_*.png
```

## 🐛 Troubleshooting

### Lỗi Import Module
```bash
# Đảm bảo bạn đang ở đúng directory
cd ml-tabular-project/notebooks

# Hoặc thêm vào đầu notebook
import sys
sys.path.append('..')
```

### Lỗi Missing Dependencies
```bash
# Cài từng package riêng lẻ
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Lỗi CUDA (cho Deep Learning)
```bash
# Nếu không có GPU, sử dụng CPU version của PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 💡 Tips

1. **Bắt đầu với dataset nhỏ** để test pipeline
2. **Uncomment từng phần** một và kiểm tra kết quả
3. **Lưu checkpoints** sau mỗi bước quan trọng
4. **Đọc output logs** để hiểu pipeline đang làm gì
5. **Thử nhiều preprocessing configurations** để tìm ra best setup

## 📚 Tài Liệu Tham Khảo

- **Scikit-learn**: https://scikit-learn.org/
- **Pandas**: https://pandas.pydata.org/
- **Matplotlib**: https://matplotlib.org/
- **Seaborn**: https://seaborn.pydata.org/

## 🤝 Đóng Góp

Các thành viên nhóm có thể:
1. Fork project
2. Tạo branch mới cho feature của mình
3. Commit và push changes
4. Tạo Pull Request

## 📞 Liên Hệ

Nếu có vấn đề, liên hệ:
- **Hoàng Xuân Bách**: bach.hoang2407khmt@hcmut.edu.vn
- GitHub Issues: [Link to your repo issues]

---

**Chúc các bạn thực hiện project thành công! 🎉**