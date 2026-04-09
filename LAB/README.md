# VKU Lab — Sentiment Analysis (SemEval-14 ABSA)

Dự án phân tích cảm xúc theo khía cạnh (Aspect-Based Sentiment Analysis) trên tập dữ liệu nhà hàng SemEval-2014, sử dụng các mô hình Machine Learning cổ điển kết hợp kỹ thuật Cross-Validation và Data Augmentation.

---

## Mục lục

1. [Tổng quan bài toán](#1-tổng-quan-bài-toán)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Dữ liệu](#3-dữ-liệu)
4. [Cài đặt môi trường](#4-cài-đặt-môi-trường)
5. [Notebooks](#5-notebooks)
6. [Pipeline xử lý](#6-pipeline-xử-lý)
7. [Mô hình & Kết quả](#7-mô-hình--kết-quả)
8. [Kỹ thuật Cross-Validation](#8-kỹ-thuật-cross-validation)
9. [Data Augmentation](#9-data-augmentation)

---

## 1. Tổng quan bài toán

**Aspect-Based Sentiment Analysis (ABSA)** là bài toán phân tích cảm xúc ở mức độ chi tiết hơn so với phân tích cảm xúc thông thường. Thay vì phân loại cảm xúc của toàn bộ câu, ABSA xác định cảm xúc đối với từng **khía cạnh cụ thể** (aspect term) được đề cập trong câu.

**Ví dụ:**
```
Câu: "The food was great but the service was terrible."
→ food      → positive
→ service   → negative
```

**Nhãn phân loại:**
| Nhãn | Ý nghĩa |
|------|---------|
| `positive` | Cảm xúc tích cực với aspect |
| `negative` | Cảm xúc tiêu cực với aspect |
| `neutral`  | Không có cảm xúc rõ ràng |
| `conflict` | Vừa tích cực vừa tiêu cực (bị loại khỏi training) |

---

## 2. Cấu trúc thư mục

```
VKU_Lab_Sentiment/
│
├── Restaurants_Train_v2.csv          # Tập dữ liệu gốc SemEval-14
├── train.csv                         # Tập train (dùng cho notebook augmented)
├── test.csv                          # Tập test (dùng cho notebook augmented)
│
├── data_1 - negative.csv             # Dữ liệu augment gốc (nhãn negative, dạng list)
├── data_1 - negative_expanded.csv    # Dữ liệu augment đã expand (mỗi câu 1 dòng)
├── data_1 - neutral.csv              # Dữ liệu augment gốc (nhãn neutral, dạng list)
├── data_1 - neutral_expanded.csv     # Dữ liệu augment đã expand (mỗi câu 1 dòng)
│
├── sa-lab1_cv.ipynb                  # Notebook chính: CV + 5 mô hình ML
├── sa-lab1_augemented.ipynb          # Notebook: Data Augmentation + ML
│
├── thang.py                          # Script expand augmented data từ list → rows
└── README.md
```

---

## 3. Dữ liệu

### Restaurants_Train_v2.csv (nguồn chính)

Tập dữ liệu gốc từ **SemEval-2014 Task 4** — Aspect-Based Sentiment Analysis, domain nhà hàng.

| Cột | Mô tả |
|-----|-------|
| `Sentence` | Câu đánh giá gốc |
| `Aspect Term` | Khía cạnh được đề cập (food, service, price...) |
| `polarity` | Nhãn cảm xúc: positive / negative / neutral / conflict |
| `from` | Vị trí bắt đầu của aspect term trong câu |
| `to` | Vị trí kết thúc của aspect term trong câu |

**Thống kê:**
```
Tổng mẫu:   3,693
positive:   2,164  (58.6%)
negative:     805  (21.8%)
neutral:      633  (17.1%)
conflict:      91   (2.5%)  ← bị loại khi training
```

> Lưu ý: Tập dữ liệu bị **mất cân bằng** (imbalanced) — positive chiếm gần 60%. Đây là lý do cần dùng Stratified K-Fold và metric F1-weighted thay vì chỉ dùng Accuracy.

### Dữ liệu Augmented

Các file `data_1 - negative_expanded.csv` và `data_1 - neutral_expanded.csv` chứa câu được sinh ra bằng LLM (Google Sheets + LLM API) để tăng cường dữ liệu cho 2 nhãn thiểu số.

Script `thang.py` dùng để chuyển đổi cột `augmented_sentences` từ dạng Python list string sang từng dòng riêng biệt.

---

## 4. Cài đặt môi trường

### Yêu cầu

- Python 3.8+
- Jupyter Notebook / JupyterLab

### Cài đặt thư viện

```bash
pip install numpy pandas matplotlib seaborn scikit-learn spacy
python -m spacy download en_core_web_sm
```

### Chạy notebook

```bash
jupyter notebook sa-lab1_cv.ipynb
```

---

## 5. Notebooks

### `sa-lab1_cv.ipynb` — Notebook chính (Cross-Validation)

Notebook đầy đủ nhất, áp dụng **5-Fold Stratified Cross-Validation** cho 5 mô hình ML.

**Các bước thực hiện:**

| Bước | Nội dung |
|------|---------|
| 1 | Load data từ `Restaurants_Train_v2.csv` |
| 2 | Tiền xử lý văn bản (lowercase, lemmatize, remove stopwords) |
| 3 | Thiết lập Stratified K-Fold CV (k=5) |
| 4 | Chạy cross-validation cho 5 mô hình |
| 5 | Vẽ biểu đồ so sánh Train vs CV Accuracy |
| 6 | Vẽ Learning Curves cho từng mô hình |
| 7 | Đánh giá cuối trên hold-out test set |
| 8 | Confusion matrices + bảng tổng hợp |

---

### `sa-lab1_augemented.ipynb` — Notebook Data Augmentation

Notebook thực nghiệm với dữ liệu được tăng cường (augmented), chạy trên **Kaggle**.

**Điểm khác biệt so với notebook CV:**
- Dùng tập train/test tách sẵn (`train.csv` / `test.csv`)
- Gộp thêm 1,100 mẫu negative và 1,200 mẫu neutral từ LLM augmentation
- Tổng tập train sau augment: ~5,176 mẫu
- Dùng train/test split thay vì cross-validation

---

## 6. Pipeline xử lý

### Tiền xử lý văn bản

```
Câu gốc
  → Lowercase
  → spaCy tokenize + lemmatize
  → Loại bỏ stopwords (giữ lại từ quan trọng)
  → Chuỗi đã xử lý
```

**Từ được giữ lại dù là stopword** (KEEP_WORDS):

| Nhóm | Từ |
|------|----|
| Phủ định | not, no, never, none, neither, nor, cannot, without |
| Tăng cường | very, too, so, quite, really, most, least, much |
| Tương phản | but, however, although, though, nevertheless, yet |
| Tần suất | always, often, sometimes, ever |

> Lý do: Các từ này mang thông tin cảm xúc quan trọng. Ví dụ "not good" khác hoàn toàn "good".

### Feature Extraction

Dùng **TF-IDF** với bigrams:

```python
TfidfVectorizer(ngram_range=(1, 2))
```

- Unigrams: "food", "great", "terrible"
- Bigrams: "not good", "very tasty", "bad service"

---

## 7. Mô hình & Kết quả

### Các mô hình được so sánh

| Mô hình | Đặc điểm |
|---------|---------|
| **Naive Bayes** | Nhanh, hiệu quả với text, giả định độc lập giữa features |
| **Logistic Regression** | Baseline mạnh cho text classification |
| **SVM** | Tốt với high-dimensional sparse data (TF-IDF) |
| **KNN** | Dựa trên khoảng cách, dùng Gaussian weight |
| **Random Forest** | Ensemble, ít bị overfit hơn Decision Tree đơn lẻ |

### Cấu hình mô hình

```python
Naive Bayes:         MultinomialNB()
Logistic Regression: LogisticRegression(max_iter=1000)
SVM:                 SVC(kernel='linear', decision_function_shape='ovo')
KNN:                 KNeighborsClassifier(n_neighbors=7, weights=gaussian)
Random Forest:       RandomForestClassifier(n_estimators=100)
```

### Metrics đánh giá

| Metric | Ý nghĩa |
|--------|---------|
| **Accuracy** | Tỷ lệ dự đoán đúng tổng thể |
| **F1-Macro** | F1 trung bình không trọng số — phản ánh hiệu quả trên từng nhãn đều nhau |
| **F1-Weighted** | F1 trung bình có trọng số theo số mẫu — phù hợp với dữ liệu mất cân bằng |

---

## 8. Kỹ thuật Cross-Validation

### Tại sao cần Cross-Validation?

Khi chia train/test một lần duy nhất, kết quả phụ thuộc nhiều vào cách chia ngẫu nhiên. Cross-Validation giải quyết vấn đề này bằng cách đánh giá mô hình trên **nhiều lần chia khác nhau**.

### Stratified K-Fold (k=5)

```
Toàn bộ dữ liệu (3,602 mẫu)
  ┌─────┬─────┬─────┬─────┬─────┐
  │ F1  │ F2  │ F3  │ F4  │ F5  │
  └─────┴─────┴─────┴─────┴─────┘

Lần 1: Train=[F2,F3,F4,F5]  Test=[F1]
Lần 2: Train=[F1,F3,F4,F5]  Test=[F2]
...
Lần 5: Train=[F1,F2,F3,F4]  Test=[F5]

Kết quả = mean ± std của 5 lần
```

**Stratified** đảm bảo mỗi fold có tỷ lệ nhãn giống tập gốc — quan trọng khi dữ liệu mất cân bằng.

### Đọc kết quả

```
SVM: Accuracy = 0.812 ± 0.015
```
- `0.812` = accuracy trung bình qua 5 fold
- `± 0.015` = độ lệch chuẩn — càng nhỏ càng ổn định

### Learning Curve

Biểu đồ Learning Curve cho thấy mô hình học như thế nào khi tăng dần lượng dữ liệu:

```
Accuracy
  │  Train ──────────────────
  │                    ╲
  │                     ╲ gap nhỏ = tốt
  │                      ╲
  │  Val  ────────────────────
  └──────────────────────────→ Training size
```

- **Gap lớn** giữa train và val → Overfitting
- **Cả hai đều thấp** → Underfitting
- **Gap nhỏ, cả hai cao** → Mô hình tốt

---

## 9. Data Augmentation

### Vấn đề mất cân bằng

```
positive: 2,164  ████████████████████████ 58.6%
negative:   805  █████████               21.8%
neutral:    633  ███████                 17.1%
```

Mô hình có xu hướng thiên về nhãn `positive` nếu không xử lý.

### Giải pháp: LLM Augmentation

Dùng LLM (qua Google Sheets) để sinh câu mới có cùng nghĩa cho nhãn `negative` và `neutral`:

```
Câu gốc:  "The service was slow."  [negative]
Augment:  "We waited a long time to be served."  [negative]
Augment:  "Staff took forever to attend to us."  [negative]
```

**Kết quả sau augment:**
```
positive: 2,164  (gốc)
negative: 1,100  (augment, sample từ expanded)
neutral:  1,200  (augment, sample từ expanded)
Tổng:     5,176 mẫu
```

### Script expand (`thang.py`)

File augment gốc lưu nhiều câu trong 1 ô dưới dạng Python list. Script `thang.py` expand ra thành từng dòng riêng:

```python
# Input:  1 dòng với augmented_sentences = "['câu 1', 'câu 2', 'câu 3']"
# Output: 3 dòng riêng biệt
python thang.py
```

---

## Tác giả

Thực hiện trong khuôn khổ môn học tại **VKU (Vietnam-Korea University of Information and Communication Technology)**.
