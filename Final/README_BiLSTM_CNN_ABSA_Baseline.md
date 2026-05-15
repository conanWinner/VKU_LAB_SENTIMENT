# README - BiLSTM-CNN Baseline for Vietnamese Student Feedback ABSA

## 1. Mục tiêu

Xây dựng lại mô hình **BiLSTM-CNN** làm baseline theo đúng tinh thần paper:

> Aspect-Based Sentiment Analysis on Student's Feedback in Vietnamese

Mô hình này được dùng để so sánh với các mô hình cải tiến như:

- mBERT Multi-task ABSA
- PhoBERT Multi-task ABSA
- PhoBERT + aspect-aware sentiment head

Dataset hiện tại có khoảng **16k samples**, lớn hơn paper gốc khoảng **5k samples**, vì vậy mục tiêu chính là:

1. Tái hiện baseline BiLSTM-CNN trên bộ dữ liệu mới.
2. Đánh giá xem việc tăng dữ liệu có giúp baseline cải thiện không.
3. So sánh công bằng với mô hình Transformer-based Multi-task Learning trên cùng train/dev/test split.

---

## 2. Bài toán

Bài toán là **Aspect-Based Sentiment Analysis trên phản hồi sinh viên tiếng Việt**.

Với mỗi câu phản hồi, mô hình cần xác định:

1. Câu có đề cập đến aspect nào không.
2. Nếu có, sentiment của aspect đó là gì.

Có tổng cộng **11 aspect**:

```python
ASPECTS = [
    "ky_nang_giang_day",
    "kinh_nghiem",
    "hanh_vi",
    "bai_tap",
    "cham_diem",
    "cung_cap_tai_lieu",
    "kien_thuc",
    "chuong_trinh_hoc",
    "thiet_bi_day_hoc",
    "de_xuat",
    "noi_chung"
]
```

Có **3 polarity chính**:

```python
POLARITIES = [
    "positive",
    "neutral",
    "negative"
]
```

Tuy nhiên, để tái hiện đúng kiến trúc BiLSTM-CNN của paper gốc, mỗi aspect sẽ được phân loại thành **4 trạng thái**:

```python
LABEL_MAP = {
    "none": 0,
    "positive": 1,
    "neutral": 2,
    "negative": 3
}
```

Trong đó:

- `none`: aspect không xuất hiện trong câu.
- `positive`: aspect xuất hiện và có cảm xúc tích cực.
- `neutral`: aspect xuất hiện và có cảm xúc trung tính.
- `negative`: aspect xuất hiện và có cảm xúc tiêu cực.

---

## 3. Format label đầu ra

Mỗi câu cần được biểu diễn thành vector label có độ dài **11**.

Mỗi vị trí trong vector tương ứng với một aspect.

Ví dụ câu:

```text
Thầy rất nhiệt tình nhưng giảng hơi khó hiểu.
```

Giả sử:

- `hanh_vi = positive`
- `ky_nang_giang_day = negative`

Thì label có dạng:

```python
labels = [
    3,  # ky_nang_giang_day = negative
    0,  # kinh_nghiem = none
    1,  # hanh_vi = positive
    0,  # bai_tap = none
    0,  # cham_diem = none
    0,  # cung_cap_tai_lieu = none
    0,  # kien_thuc = none
    0,  # chuong_trinh_hoc = none
    0,  # thiet_bi_day_hoc = none
    0,  # de_xuat = none
    0   # noi_chung = none
]
```

Shape label trong batch:

```python
labels.shape == (batch_size, 11)
```

Output logits của model:

```python
logits.shape == (batch_size, 11, 4)
```

---

## 4. Lưu ý quan trọng khi convert label

Nếu dữ liệu gốc đang dùng mapping sentiment như sau:

```python
0 = negative
1 = neutral
2 = positive
```

thì bắt buộc phải convert lại sang mapping mới:

```python
none     -> 0
positive -> 1
neutral  -> 2
negative -> 3
```

Ví dụ:

```python
OLD_TO_NEW_SENTIMENT = {
    0: 3,  # negative
    1: 2,  # neutral
    2: 1   # positive
}
```

Nếu dữ liệu đang ở dạng mỗi dòng là một cặp `(text, aspect, sentiment)`, cần group lại theo `text` hoặc `id`.

Pseudo-code:

```python
def build_multiaspect_label(rows_for_one_sentence):
    label = [0] * 11

    for row in rows_for_one_sentence:
        aspect_idx = ASPECT2ID[row["aspect"]]
        sentiment_new = OLD_TO_NEW_SENTIMENT[row["sentiment"]]
        label[aspect_idx] = sentiment_new

    return label
```

---

## 5. Kiến trúc tổng thể

Kiến trúc cần implement:

```text
Input sentence
      ↓
Preprocessing
      ↓
Tokenization + Padding
      ↓
Word Embedding layer
      ↓
BiLSTM layer
      ↓
CNN layers with kernel sizes 2, 3, 4
      ↓
Global Max Pooling
      ↓
Concatenate pooled features
      ↓
Fully Connected + ReLU
      ↓
Output layer
      ↓
11 aspects × 4 classes
```

Output cuối cùng:

```text
(batch_size, 11, 4)
```

Trong đó:

- `11`: số aspect
- `4`: số class của mỗi aspect gồm `none`, `positive`, `neutral`, `negative`

---

## 6. Layer chi tiết

### 6.1. Input

Input là chuỗi token id:

```python
input_ids.shape == (batch_size, max_len)
```

Trong đó:

- `batch_size`: số mẫu trong một batch.
- `max_len`: độ dài câu sau padding/truncation.

---

### 6.2. Embedding layer

Dùng word embedding tiếng Việt.

Ưu tiên:

1. `word2vecVN`
2. `fastText Vietnamese`
3. Nếu không có pretrained embedding thì dùng `nn.Embedding` train từ đầu.

Cấu hình:

```python
embedding_dim = 300
```

Output:

```python
embedding_output.shape == (batch_size, max_len, 300)
```

---

### 6.3. BiLSTM layer

Cấu hình theo paper:

```python
hidden_size = 128
bidirectional = True
batch_first = True
```

Vì BiLSTM hai chiều nên output dimension:

```python
lstm_output_dim = hidden_size * 2 = 256
```

Output:

```python
lstm_output.shape == (batch_size, max_len, 256)
```

Vai trò:

- Học ngữ cảnh trái → phải.
- Học ngữ cảnh phải → trái.
- Tạo contextual representation cho từng token.

---

### 6.4. CNN layer

Sau BiLSTM, đưa output vào các CNN branch.

Cần transpose trước khi đưa vào `Conv1d`:

```python
x = lstm_output.transpose(1, 2)
```

Shape:

```python
x.shape == (batch_size, 256, max_len)
```

Cấu hình CNN:

```python
kernel_sizes = [2, 3, 4]
num_filters = 128
```

Mỗi kernel size là một nhánh Conv1D riêng:

```python
Conv1d(
    in_channels=256,
    out_channels=128,
    kernel_size=k
)
```

Ý nghĩa:

- Kernel 2 học đặc trưng 2-gram.
- Kernel 3 học đặc trưng 3-gram.
- Kernel 4 học đặc trưng 4-gram.

---

### 6.5. Global Max Pooling

Sau mỗi CNN branch:

```python
conv_output.shape == (batch_size, 128, max_len - k + 1)
```

Áp dụng global max pooling trên chiều sequence:

```python
pooled = F.max_pool1d(conv_output, kernel_size=conv_output.size(2)).squeeze(2)
```

Output mỗi branch:

```python
pooled.shape == (batch_size, 128)
```

Có 3 branch nên concatenate:

```python
concat.shape == (batch_size, 128 * 3)
concat.shape == (batch_size, 384)
```

---

### 6.6. Fully Connected layer

Cấu hình đề xuất:

```python
fc_hidden = 128
dropout = 0.5
```

Flow:

```text
Concatenated CNN features
      ↓
Dropout
      ↓
Linear(384, 128)
      ↓
ReLU
      ↓
Dropout
```

---

### 6.7. Output layer

Output layer:

```python
Linear(128, num_aspects * num_classes)
```

Với:

```python
num_aspects = 11
num_classes = 4
```

Nên:

```python
output_dim = 11 * 4 = 44
```

Sau đó reshape:

```python
logits = logits.view(batch_size, 11, 4)
```

---

## 7. Loss function

Vì mỗi aspect là một bài toán classification 4 lớp, dùng:

```python
nn.CrossEntropyLoss()
```

Input loss:

```python
logits.view(-1, 4)
labels.view(-1)
```

Code:

```python
loss_fn = nn.CrossEntropyLoss()

loss = loss_fn(
    logits.view(-1, 4),
    labels.view(-1)
)
```

Không dùng `IGNORE_INDEX` cho baseline BiLSTM-CNN này, vì class `none` đã đại diện cho aspect không xuất hiện.

---

## 8. Code skeleton PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMCNN_ABSA(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=300,
        hidden_size=128,
        num_aspects=11,
        num_classes=4,
        num_filters=128,
        kernel_sizes=(2, 3, 4),
        fc_hidden=128,
        dropout=0.5,
        pretrained_embeddings=None,
        padding_idx=0
    ):
        super().__init__()

        self.num_aspects = num_aspects
        self.num_classes = num_classes

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(pretrained_embeddings, dtype=torch.float)
            )

        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        lstm_out_dim = hidden_size * 2

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=lstm_out_dim,
                out_channels=num_filters,
                kernel_size=k
            )
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(num_filters * len(kernel_sizes), fc_hidden)
        self.output = nn.Linear(fc_hidden, num_aspects * num_classes)

    def forward(self, input_ids, labels=None):
        # input_ids: (B, L)
        emb = self.embedding(input_ids)  # (B, L, 300)

        lstm_out, _ = self.bilstm(emb)   # (B, L, 256)

        # Conv1d expects: (B, C, L)
        x = lstm_out.transpose(1, 2)     # (B, 256, L)

        conv_features = []

        for conv in self.convs:
            c = F.relu(conv(x))          # (B, 128, L-k+1)
            p = F.max_pool1d(
                c,
                kernel_size=c.size(2)
            ).squeeze(2)                 # (B, 128)

            conv_features.append(p)

        x = torch.cat(conv_features, dim=1)  # (B, 384)

        x = self.dropout(x)
        x = F.relu(self.fc(x))
        x = self.dropout(x)

        logits = self.output(x)  # (B, 44)
        logits = logits.view(
            -1,
            self.num_aspects,
            self.num_classes
        )  # (B, 11, 4)

        loss = None

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                logits.view(-1, self.num_classes),
                labels.view(-1)
            )

        return loss, logits
```

---

## 9. Dataset class gợi ý

```python
from torch.utils.data import Dataset
import torch


class ABSADataset(Dataset):
    def __init__(self, samples, word2idx, max_len):
        self.samples = samples
        self.word2idx = word2idx
        self.max_len = max_len
        self.unk_idx = word2idx.get("<UNK>", 1)
        self.pad_idx = word2idx.get("<PAD>", 0)

    def encode_text(self, tokens):
        ids = [self.word2idx.get(tok, self.unk_idx) for tok in tokens]

        if len(ids) > self.max_len:
            ids = ids[:self.max_len]

        attention_len = len(ids)

        if len(ids) < self.max_len:
            ids = ids + [self.pad_idx] * (self.max_len - len(ids))

        return ids, attention_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # item["tokens"]: list[str]
        # item["labels"]: list[int] length = 11

        input_ids, length = self.encode_text(item["tokens"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long)
        }
```

Lưu ý: Model skeleton ở trên chưa dùng `length` để pack sequence. Nếu muốn đơn giản thì không cần pack. Nếu muốn chuẩn hơn, có thể dùng `pack_padded_sequence`.

---

## 10. Preprocessing tiếng Việt

Pipeline đề xuất:

```text
Raw text
   ↓
Lowercase
   ↓
Normalize whitespace
   ↓
Remove unnecessary punctuation / emoji nếu cần
   ↓
Vietnamese word segmentation
   ↓
Build vocabulary
   ↓
Convert tokens to ids
   ↓
Padding / truncation
```

Có thể dùng:

- PyVi
- VnCoreNLP
- Underthesea

Ví dụ:

```python
from pyvi import ViTokenizer

text = "Thầy giảng bài rất dễ hiểu."
segmented = ViTokenizer.tokenize(text)
tokens = segmented.lower().split()
```

---

## 11. Training config đề xuất

```python
CONFIG = {
    "embedding_dim": 300,
    "hidden_size": 128,
    "num_filters": 128,
    "kernel_sizes": [2, 3, 4],
    "fc_hidden": 128,
    "dropout": 0.5,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "epochs": 20,
    "early_stopping_patience": 3,
    "max_len": 128,
    "optimizer": "Adam",
    "loss": "CrossEntropyLoss"
}
```

Gợi ý:

- Nếu overfit: tăng dropout lên 0.5.
- Nếu underfit: giảm dropout về 0.3 hoặc tăng `fc_hidden = 256`.
- Theo dõi dev macro-F1 để early stopping.

---

## 12. Inference

Model output:

```python
logits.shape == (batch_size, 11, 4)
```

Dự đoán:

```python
pred = logits.argmax(dim=-1)
```

Shape:

```python
pred.shape == (batch_size, 11)
```

Diễn giải:

```python
ID2LABEL = {
    0: "none",
    1: "positive",
    2: "neutral",
    3: "negative"
}
```

Pseudo-code:

```python
def decode_prediction(pred):
    results = []

    for aspect_idx, label_id in enumerate(pred):
        if label_id == 0:
            continue

        results.append({
            "aspect": ASPECTS[aspect_idx],
            "sentiment": ID2LABEL[label_id]
        })

    return results
```

Ví dụ output:

```python
pred = [3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
```

Decode:

```python
[
    {"aspect": "ky_nang_giang_day", "sentiment": "negative"},
    {"aspect": "hanh_vi", "sentiment": "positive"}
]
```

---

## 13. Evaluation

Cần đánh giá ít nhất 2 task:

### 13.1. Aspect Detection

Convert prediction 4-class thành aspect binary:

```python
aspect_pred = pred != 0
aspect_true = labels != 0
```

Tính:

- Precision
- Recall
- Micro-F1
- Macro-F1

---

### 13.2. Aspect-Sentiment Classification

Một aspect được tính đúng khi:

```text
predicted aspect == gold aspect
and
predicted sentiment == gold sentiment
```

Tức là so trực tiếp:

```python
pred == labels
```

nhưng bỏ qua các trường hợp cả hai đều là `none` để tránh F1 bị ảo.

Nên tính F1 trên các class:

```python
positive
neutral
negative
```

hoặc tính trên cặp:

```python
aspect + sentiment
```

Ví dụ class tổng hợp:

```python
"ky_nang_giang_day_positive"
"ky_nang_giang_day_neutral"
"ky_nang_giang_day_negative"
...
```

Tổng số class aspect-sentiment:

```python
11 aspects * 3 sentiments = 33 classes
```

---

## 14. Các mô hình cần so sánh

Khuyến nghị setup thí nghiệm như sau:

| Model | Vai trò |
|---|---|
| BiLSTM-CNN | Baseline tái hiện paper gốc |
| mBERT Multi-task | Baseline Transformer |
| PhoBERT Multi-task | Strong Vietnamese baseline |
| PhoBERT + Aspect-aware Sentiment Head | Proposed model |

Tất cả model phải train/test trên cùng split:

```text
train / dev / test
```

Không so sánh trực tiếp kết quả của dataset 16k với paper gốc 5k nếu split và label khác nhau.

---

## 15. Checklist implement

### Data

- [ ] Load raw dataset.
- [ ] Chuẩn hóa tên aspect.
- [ ] Chuẩn hóa sentiment mapping.
- [ ] Group các dòng cùng `id` hoặc cùng `text` nếu data đang ở dạng long format.
- [ ] Convert mỗi câu thành label vector độ dài 11.
- [ ] Word segmentation tiếng Việt.
- [ ] Build vocabulary từ train set.
- [ ] Tạo `<PAD>` và `<UNK>`.
- [ ] Padding/truncation về `max_len`.

### Model

- [ ] Implement `BiLSTMCNN_ABSA`.
- [ ] Embedding dim = 300.
- [ ] BiLSTM hidden size = 128.
- [ ] CNN kernel sizes = 2, 3, 4.
- [ ] Num filters = 128.
- [ ] Global max pooling.
- [ ] FC + ReLU.
- [ ] Output shape = `(B, 11, 4)`.

### Training

- [ ] CrossEntropyLoss.
- [ ] Adam optimizer.
- [ ] LR = 0.001.
- [ ] Batch size = 32.
- [ ] Early stopping theo dev macro-F1.
- [ ] Save best checkpoint.

### Evaluation

- [ ] Aspect Detection micro-F1.
- [ ] Aspect Detection macro-F1.
- [ ] Aspect-Sentiment micro-F1.
- [ ] Aspect-Sentiment macro-F1.
- [ ] Classification report theo từng aspect.
- [ ] Confusion matrix nếu cần.

---

## 16. Lưu ý để tránh sai

### Không dùng BCEWithLogitsLoss cho baseline này

Baseline BiLSTM-CNN theo paper gốc nên dùng:

```python
CrossEntropyLoss
```

vì mỗi aspect có đúng 1 trong 4 trạng thái:

```text
none / positive / neutral / negative
```

---

### Không dùng `ignore_index` cho aspect không xuất hiện

Aspect không xuất hiện phải là:

```python
0  # none
```

Không được để là `IGNORE_INDEX`.

---

### Không so sánh thiếu công bằng

Không nên kết luận:

```text
Model của tôi tốt hơn paper gốc vì F1 cao hơn.
```

nếu:

- dataset khác
- số lượng data khác
- cách label khác
- split khác

Cách kết luận đúng:

```text
Trên bộ dữ liệu được mở rộng và cùng một điều kiện thí nghiệm, mô hình Transformer-based Multi-task Learning đạt hiệu quả cao hơn so với baseline BiLSTM-CNN được tái hiện từ paper gốc.
```

---

## 17. Mô tả ngắn để đưa vào báo cáo

Mô hình BiLSTM-CNN được xây dựng lại dựa trên kiến trúc của nghiên cứu gốc nhằm làm baseline so sánh. Đầu vào sau khi tiền xử lý được chuyển thành chuỗi chỉ số từ và ánh xạ qua lớp word embedding 300 chiều. Tiếp theo, lớp BiLSTM hai chiều với kích thước ẩn 128 được sử dụng để học biểu diễn ngữ cảnh theo cả hai chiều trái–phải và phải–trái. Biểu diễn đầu ra của BiLSTM được đưa qua các lớp CNN với kích thước kernel 2, 3 và 4 nhằm trích xuất các đặc trưng cục bộ dạng 2-gram, 3-gram và 4-gram. Sau đó, Global Max Pooling được áp dụng trên từng nhánh CNN để lấy các đặc trưng quan trọng nhất, các đặc trưng này được nối lại và đưa qua lớp fully connected với hàm kích hoạt ReLU. Cuối cùng, mô hình sinh ra đầu ra có kích thước 11 × 4, tương ứng với 11 khía cạnh và 4 trạng thái gồm None, Positive, Neutral và Negative.

---

## 18. Kỳ vọng kết quả

Vì dataset hiện tại có khoảng 16k samples, lớn hơn dataset gốc khoảng 5k samples, BiLSTM-CNN có thể cải thiện so với khi train trên tập nhỏ hơn. Tuy nhiên, mô hình này vẫn là baseline truyền thống, nên kỳ vọng hợp lý là:

```text
BiLSTM-CNN < mBERT/PhoBERT Multi-task < PhoBERT Aspect-aware Multi-task
```

Nếu BiLSTM-CNN tốt hơn Transformer, cần kiểm tra lại:

- Split data có bị leakage không.
- Label có đúng mapping không.
- Transformer có bị learning rate quá cao không.
- Tokenization có phù hợp không.
- Metric có bỏ qua class `none` đúng cách không.
