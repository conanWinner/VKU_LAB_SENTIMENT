# Hiện trạng dự án (kiểm tra ngày 01/11/2026)

Bản thảo: `Server/model/kkkkk.pdf` (23/05/2026). Commit tại thời điểm kiểm tra: `e89f1ce`.

## Kết luận

Bài nộp chỉ làm một miền, dữ liệu tiếng Việt (UIT-VSFC, 11 khía cạnh). Phần này đã có dữ liệu, code và kết quả, đủ làm nền cho bài nộp. Khoảng trống lớn nhất là độ tin cậy của thực nghiệm và của nhãn: mới chạy một seed và chưa có chỉ số đồng thuận cho nhãn mới.

## Dữ liệu `data/Final_Data`

| Tập | Câu | Cặp (aspect, sentiment) |
|---|---|---|
| Train | 11.426 mẫu (11.425 chuỗi khác nhau) | 16.920 |
| Validation | 1.583 | 2.320 |
| Test | 3.166 | 4.670 |
| **Tổng** | **16.175** | **23.910** |

Từ 01/11/2026, bộ dữ liệu khớp đủ số câu và split gốc của UIT-VSFC. 51 câu đã bị nhóm xóa khi rà soát (37 train, 8 validation, 6 test) được đưa lại: nhãn do Claude đề xuất, người rà soát đã duyệt ngày 01/11/2026 và đổi 4 nhãn cảm xúc, lưu kèm độ tin cậy và nhãn đề xuất ban đầu trong `data/annotation/added_51_review.csv`. Hash bộ dữ liệu đã duyệt: `c13e760bb2d47295b5dea3ddda4e3d1e65f15846d2afece54ec9531db69643f1`. Các kết quả cũ bên dưới được chạy trên bộ 16.124 mẫu trước đó.

Kiểm tra đã thực hiện:

- Không có câu trùng giữa train, validation và test.
- Không có câu cùng khía cạnh mà mang hai cảm xúc khác nhau.
- `id` lặp lại giữa các file train (1.999 id khác nhau). Notebook hiện group từng file nên kết quả không bị ảnh hưởng, nhưng code mới phải giữ đúng thứ tự này.
- **`data/paper_subset` chồng lấn với `Final_Data`:** ví dụ `train_paper` có 687 câu nằm trong test của `Final_Data`. Không thể huấn luyện trên tập này rồi đánh giá trên test 16k.
- **`data/paper_subset` không phải dữ liệu của [2]:** 100% cặp của cả hai bộ đều có trong nhãn của mình. Bộ `*_paper` (5.286 dòng) có số đếm trùng Bảng 2 của [2] ở gần như mọi ô, trừ các ô mà dữ liệu của mình không đủ (`thiet_bi_day_hoc` positive 37 so với 59), nên nhiều khả năng được lấy mẫu từ nhãn của mình để giống bảng đó. Bộ còn lại có đúng 5.010 câu, chia 3.507/501/1.002. Không dùng tập này để nói đã tái hiện [2].

Dataset Kaggle cho bài là `thngonquang/vku-feedback-v2` (riêng tư), đã đối chiếu hash `c13e760b…` với `data/Final_Data`.

## Kết quả hiện có (một seed, chưa dùng làm số cuối)

| Mô hình | Nguồn | Task A F1 | Task B F1 |
|---|---|---|---|
| BiLSTM-CNN | `billstm_cnn.ipynb` | 81,79 | 72,09 |
| BiLSTM-CNN + Focal Loss | `16k-final-bilstm-cnn.ipynb` | 81,65 | 71,62 |
| PhoBERT MTL, không cross-attention | `phobert_multitask.ipynb` tại commit `5867528` | 86,38 | 78,47 |
| PhoBERT + cross-attention (chỉ nhánh cảm xúc) | `phobert_multitask.ipynb` hiện tại | 87,10 | 79,33 |
| mBERT MTL | `mbert_multitask.ipynb` | 86,63 | Chưa có theo cùng độ đo |

Task A: phát hiện khía cạnh. Task B: đúng cả cặp khía cạnh và cảm xúc. Đều là micro-F1 (%), threshold 0,5 và test của `Final_Data`.

`mberrrrrrr.ipynb` đạt 73,31 / 63,30 trên test gộp có Coursera, nên không so sánh được với bảng trên và nằm ngoài phạm vi bài.

## Kết quả đo bằng code mới (01/11/2026, **không dùng số cho bài**)

Tất cả dùng dữ liệu cũ 16.124 mẫu, seed 42 trên Kaggle. Dữ liệu nằm trong `results/g2_reproduction_kaggle_v1/` và `results/pilot_kaggle_v1/`.

**G2 tái hiện đạt** (β = 1,5, chọn checkpoint kiểu cũ, ngưỡng 0,5, test cũ đã từng xem):

| Mô hình | Task A notebook → code mới | Task B notebook → code mới |
|---|---|---|
| E3 | 87,10 → 86,88 | 79,33 → 78,98 |
| E1 | 81,79 → 82,04 | 72,09 → 72,30 |

**Pilot** (β = 1,0, chọn theo `pair_f1_micro`, ngưỡng chung chọn trên validation, chỉ validation, một seed nên chưa so sánh được mô hình):

| Run | Ngưỡng | Task A | Task B | Thời gian train | Bộ nhớ đỉnh |
|---|---|---|---|---|---|
| E4 PhoBERT | 0,55 | 87,43 | 79,77 | 853 s | 2.885 MB |
| E2 BamiBERT | 0,65 | 86,54 | 79,22 | 884 s | 2.464 MB |
| E4 BamiBERT | 0,70 | 87,79 | 79,19 | 909 s | 2.517 MB |

- Cả ba run đạt điểm tốt nhất ở epoch cuối (epoch 8). Protocol cố định 8 epoch nên không đổi; ghi điều này trong bài.
- Tỷ lệ câu bị cắt ở độ dài 128: 1 câu train, 1 câu validation, 0 câu test, với cả hai tokenizer.
- E1 mất 108 s. Lõi P0 ước tính khoảng 4,6 giờ nếu chạy tuần tự, khoảng 2,3 giờ trên 2 GPU, chưa tính lượt lỗi. Quota Kaggle chưa đọc được nên **G3 chưa đóng**.

**BamiBERT:** chạy được với `transformers` 5.5.0, revision `57bc1340`. Model card ghi giấy phép BSD 3-Clause Clear kèm Qualcomm Responsible AI License, dành cho nghiên cứu và giáo dục, và yêu cầu trích dẫn (arXiv:2607.02259; cần kiểm tra thông tin bài trước khi đưa vào `.bib`). Trọng số `pooler` thiếu không ảnh hưởng vì code dùng `last_hidden_state`.

**Chẩn đoán lỗi trên validation (`diagnosis_validation.json` của từng run):**

| Run | Pair F1 `de_xuat` | Sai cảm xúc / 278 | Bỏ sót | Phát hiện thừa |
|---|---|---|---|---|
| E3 PhoBERT (β = 1,5, checkpoint kiểu cũ) | 0,611 | 86 | 26 | 13 |
| E4 PhoBERT | 0,616 | 85 | 24 | 17 |
| E2 BamiBERT | 0,603 | 83 | 35 | 10 |
| E4 BamiBERT | 0,609 | 88 | 25 | 11 |

- Lỗi cảm xúc của `de_xuat` gần như không đổi qua hai encoder và hai kiến trúc, nên đổi head hay encoder chưa chắc giải quyết được. Ở E3, nhầm chính là positive → neutral (55) và negative → neutral (28); train `de_xuat` có 72% neutral.
- Giả thuyết cần kiểm chứng: nhãn cảm xúc của câu đề xuất thiếu nhất quán. Ví dụ nhãn negative mà mô hình đoán neutral: "thực hành nhiều hơn .", "tăng thêm thời lượng môn học .". Không có audit nên giả thuyết này không có số đo riêng; chưa sửa nhãn và chưa đổi protocol.
- Nhầm giữa hai khía cạnh rất ít, cặp cao nhất của mỗi run chỉ 9–12 câu, nên chưa ủng hộ nhiều giả thuyết "các khía cạnh nhầm lẫn nhau do dùng chung vector câu".
- Pair F1 thấp khác nằm ở `chuong_trinh_hoc`, `kinh_nghiem`, `noi_chung`, chủ yếu do lỗi phát hiện.

## Đối chiếu với bài [2] (Sau et al., TNU JST 2021)

Bài [2] có bản PDF tại `~/Desktop/conanWinner/Projects/VKU_SA/5000.pdf`. Hai bộ nhãn dùng chung tên 11 khía cạnh và câu từ UIT-VSFC, nhưng **quy tắc gán nhãn khác nhau**, nên số F1 không so trực tiếp được.

| Đặc điểm | [2] | Bộ của mình |
|---|---|---|
| Quy mô | 5.010 câu, 5.358 cặp (1,07 cặp/câu, tính từ Bảng 2) | 16.175 câu, 23.910 cặp (1,48 cặp/câu) |
| Neutral | 2,1% | 14,8% (UIT-VSFC gốc trên train: 4,0%) |
| `de_xuat` | 2,8% số cặp; 80% positive, 19% neutral | 12,0% số cặp; 16,5% positive, 71% neutral |
| `noi_chung` | 10,6% số cặp | 3,6% số cặp |
| `thiet_bi_day_hoc` | 92% positive | 88% negative |
| Gán nhãn | 3 người, thủ công | LLM rồi người rà soát; prompt có gợi ý từ nhãn UIT-VSFC |
| Task A / Task B, BiLSTM-CNN | 78,93 / 73,78 | Xem `results/` |

Vì sao khác:

- Prompt trong `For Train 6.json` yêu cầu "góp ý về khía cạnh cụ thể thì gán cả khía cạnh đó và `de_xuat`", và định nghĩa neutral gồm "neutral suggestion". Hai quy tắc này giải thích `de_xuat` và neutral cao hơn nhiều so với [2].
- Dòng `thiet_bi_day_hoc` của Bảng 2 trái với UIT-VSFC gốc: trong train, 533/564 cặp thiết bị của mình đến từ câu gốc negative. Có thể bảng bị nhầm cột, nhưng không kiểm chứng được; không dựa vào dòng này khi viết.

**Nhãn UIT-VSFC được đưa vào prompt làm gợi ý.** Vì vậy mức khớp với nhãn gốc **không phải bằng chứng độc lập về chất lượng**. Trên câu một khía cạnh của train, mức khớp là 91,44% (6.694 câu), 93,65% nếu bỏ `de_xuat`, và 13,5% riêng `de_xuat`. Phần Dataset và Limitations phải nêu việc này.

**Kiểm chứng giả thuyết `de_xuat` trên train:** 1.920/2.044 cặp `de_xuat` (94%) đến từ câu gốc negative, trong đó nhãn của mình chia thành neutral 1.372, positive 306, negative 242. Đọc mẫu thấy các câu cùng dạng lại khác nhãn: "nên có các buổi thuyết trình theo nhóm" là positive, "cần nhiều hoạt động ngoại khóa" là negative, "nên cập nhật những phần mềm mới" là neutral. Đây là dấu hiệu thiếu nhất quán từ một mẫu nhỏ, chưa phải kết luận. Nhóm đã hủy audit nên không có số đo độc lập; trong bài chỉ viết là giả thuyết. Chưa sửa nhãn.

**E1 của mình phỏng theo [2], không phải bản tái hiện:**

| Tham số | [2] | E1 |
|---|---|---|
| Giống nhau | BiLSTM 128, 128 bộ lọc kích thước 2, 3, 4, batch 32, Adam, Word2Vec tin tức 300 chiều, softmax 4 lớp cho mỗi khía cạnh | Như [2] |
| Tốc độ học | 0,001 | 5e-4, weight decay 1e-3, giảm LR khi không cải thiện |
| Class weight, dropout | Không nêu | Căn nghịch đảo tần suất, dropout 0,6 |
| Tiền xử lý | Pyvi, chữ thường, số thành `num` | underthesea, chữ thường, giữ nguyên số |

Không chạy lại E1: nhãn và dữ liệu đã khác [2] nên chỉnh cho giống cũng không so được với số của [2]. E1 giữ vai trò baseline cùng họ kiến trúc. File Word2Vec `baomoi.vn.model.bin` có vẻ cùng nguồn `sonvx/word2vecVN` của [2]; xác nhận trước khi viết.

## Vấn đề về code và phương pháp

| # | Vấn đề | Hệ quả | Xử lý |
|---|---|---|---|
| 1 | Chỉ chạy seed 42 | Chênh 0,72 điểm giữa hai bản PhoBERT có thể chỉ là nhiễu | 3 seed đồng đều, thêm seed nếu còn quota; báo cáo trung bình ± độ lệch chuẩn và khoảng tin cậy ghép cặp |
| 2 | `sent_f1` có nghĩa khác nhau giữa các notebook (`billstm_cnn` 0,8022 so với `16k-final` 0,7162) | Dễ so sánh nhầm | Một module độ đo dùng chung |
| 3 | Test dùng threshold 0,5; ô tìm threshold chạy sau đó và kết quả (0,6) không áp dụng lại cho test; server lại dùng 0,6 | Không khớp giữa bài báo và demo | Chọn threshold trên validation rồi áp dụng cho test |
| 4 | Mỗi notebook có một bản preprocessing, loader và độ đo riêng | Khó bảo đảm so sánh công bằng | Gom về `src/` |
| 5 | Kaggle output không lưu thời gian chạy | Chưa ước lượng được quota GPU | Đo ở lần chạy đầu tiên |
| 6 | `best_model.pt` và `best_model2.pt` đều không có lớp cross-attention | Không có checkpoint của mô hình đề xuất; demo không phải mô hình trong bài | Lưu checkpoint trong lần chạy mới |
| 7 | Checkpoint chọn theo trung bình nhân của `aspect_f1` và `sent_f1`, không phải F1 cặp | Có thể chọn sai epoch cho Task B | Chọn theo `pair_f1_micro` trên validation |
| 8 | `BETA` khai báo 1,0 nhưng model dùng 1,5; không rõ giá trị này được chọn dựa trên tập nào | Nguy cơ đã tinh chỉnh theo test | Cố định β = 1,0 cho mọi mô hình; 1,5 chỉ dùng khi tái hiện |
| 9 | Code dùng `underthesea`, trang PhoBERT khuyến nghị RDRSegmenter | Có thể ảnh hưởng kết quả; chưa đo | Kiểm tra ở E13 |

## Vấn đề của bản thảo

| # | Vấn đề | Xử lý |
|---|---|---|
| 1 | Viết tiếng Việt, 17 trang, không theo template | Viết lại bằng tiếng Anh, tối đa 8 trang |
| 2 | Tiêu đề và §7 nói về đa miền, đa ngôn ngữ, ngoài phạm vi đã chốt; §7.4 cũng không có số liệu | Đổi tiêu đề, bỏ toàn bộ §7 |
| 3 | **Không báo cáo độ đồng thuận nhãn**, trong khi bộ nhãn là đóng góp chính. Bảng 1 có số này cho hai nghiên cứu trước (91,20% và 88,95%) | Đo theo `04-experiment-protocol.md` |
| 4 | Không nói bao nhiêu câu được người rà soát, bao nhiêu nhãn LLM bị sửa | Thu thập số liệu từ những người đã rà soát; báo cáo đúng phạm vi đã làm |
| 5 | Mô tả kiến trúc (mean pooling, hai nhánh) không khớp mô hình cho số tốt nhất (cross-attention) | Viết lại theo mô hình thực sự được đánh giá |
| 6 | Nói nhờ "Self-attention và quan hệ giữa các nhãn" nên cải thiện khía cạnh ít dữ liệu, nhưng không có thực nghiệm chứng minh | Có thí nghiệm thì giữ, không thì bỏ |
| 7 | Cho rằng MTL "giảm lan truyền lỗi", nhưng không có mô hình pipeline hoặc đơn nhiệm để so sánh | Thêm E7 hoặc bỏ câu này |
| 8 | Giải thích F1 theo độ phổ biến không khớp dữ liệu: `thiet_bi_day_hoc` đạt 93% với 564 mẫu train, trong khi `chuong_trinh_hoc` đạt 70% với 989 mẫu | Phân tích lỗi; khả năng cao liên quan đến độ mơ hồ của nhãn, cần kiểm chứng |
| 9 | Chỉ có 4 tài liệu tham khảo; thiếu PhoBERT, BERT và ABSA hiện đại | Bổ sung và kiểm tra từng nguồn |
| 10 | Lỗi biên tập: "Phụ lục ??", chú thích Hình 5 và Hình 8 lẫn nhau, F1 78,93 bị cắt câu ở §2 | Hết khi viết lại |
| 11 | Tên LLM "GPT-OSS-120B" cần kiểm tra lại | Đối chiếu với `For Train 6.json` và cấu hình API thực tế |
| 12 | So với [2] (F1 78,93) là so trên hai bộ dữ liệu khác nhau | Không tuyên bố vượt [2]; chỉ so trên cùng test sau khi tái hiện baseline |
