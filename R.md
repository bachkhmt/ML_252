# Mô tả Bộ dữ liệu Adult Income (Census Income)

Bộ dữ liệu này được trích xuất từ cơ sở dữ liệu điều tra dân số năm 1994, được sử dụng để dự đoán xem một cá nhân có mức thu nhập trên hay dưới **$50,000/năm**.

## 1\. Danh sách các thuộc tính (Attributes)

|Tên cột|Ý nghĩa|Kiểu dữ liệu|Ví dụ|
|-|-|-|-|
|**age**|Độ tuổi của cá nhân.|Số nguyên (Integer)|39, 50|
|**workclass**|Hình thức/Phân nhóm việc làm.|Phân loại (Categorical)|Private, State-gov, Self-emp|
|**fnlwgt**|Trọng số dân số (Final weight).|Số nguyên (Integer)|77516, 83311|
|**education**|Trình độ học vấn cao nhất đạt được.|Phân loại (Categorical)|Bachelors, HS-grad, Masters|
|**education\_num**|Số năm đi học tương ứng.|Số nguyên (Integer)|13 (Bachelors), 9 (HS-grad)|
|**marital\_status**|Tình trạng hôn nhân hiện tại.|Phân loại (Categorical)|Divorced, Married-civ-spouse|
|**occupation**|Nghề nghiệp cụ thể đang làm.|Phân loại (Categorical)|Exec-managerial, Sales|
|**relationship**|Mối quan hệ trong gia đình.|Phân loại (Categorical)|Husband, Not-in-family, Wife|
|**race**|Chủng tộc của cá nhân.|Phân loại (Categorical)|White, Black, Asian-Pac-Islander|
|**sex**|Giới tính.|Nhị phân (Binary)|Male, Female|
|**capital\_gain**|Lợi nhuận thu được từ đầu tư vốn.|Số nguyên (Integer)|2174, 0|
|**capital\_loss**|Tổn thất từ đầu tư vốn.|Số nguyên (Integer)|0, 1902|
|**hours\_per\_week**|Số giờ làm việc trung bình mỗi tuần.|Số nguyên (Integer)|40, 13, 50|
|**native\_country**|Quốc gia gốc của cá nhân.|Phân loại (Categorical)|United-States, Vietnam, Mexico|
|**income**|**(Biến mục tiêu)** Mức thu nhập năm.|Phân loại (Categorical)|>50K, <=50K|

---

## 2\. Ghi chú quan trọng cho xử lý dữ liệu (R)

* **Thiếu dữ liệu (Missing Values):** Trong bộ dữ liệu này, các giá trị thiếu thường được ký hiệu bằng dấu chấm hỏi `?`. Bạn có thể xử lý bằng cách chuyển `?` thành `NA` trong R.
* **Cột fnlwgt:** Đây là trọng số do Cục điều tra dân số tính toán. Trong phần lớn các mô hình học máy cơ bản, cột này có thể được lược bỏ vì nó không trực tiếp phản ánh đặc điểm sinh học/xã hội của cá nhân đó.
* **Mối tương quan:** Cột `education` và `education\_num` cung cấp thông tin giống nhau nhưng ở định dạng khác nhau (Chữ vs Số). Thông thường, chúng ta sẽ sử dụng `education\_num` để mô hình dễ tính toán hơn.

---

## 3\. Code R để gán tên nhanh

```R
# Gán dữ liệu 
my\_data <- read.csv("C:/Users/PC/Downloads/adult/adult.data", header = FALSE)



my\_data <- read.csv("C:/Users/PC/Downloads/metro+interstate+traffic+volume/Metro\_Interstate\_Traffic\_Volume.csv")

# Gán tên cột sau khi đã load dữ liệu
colnames(my\_data) <- c("age", "workclass", "fnlwgt", "education", "education\_num", 
                       "marital\_status", "occupation", "relationship", "race", "sex", 
                       "capital\_gain", "capital\_loss", "hours\_per\_week", "native\_country", "income")

# xem dữ liệu bảng
View(my\_data)

