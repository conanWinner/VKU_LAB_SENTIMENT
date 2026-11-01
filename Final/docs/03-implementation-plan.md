# Kế hoạch triển khai FJCAI 2027

Bắt đầu Chủ nhật 01/11/2026. **Mục tiêu nộp: thứ Sáu 13/11/2026.** Hạn chính thức: 15/11/2026.

## Phạm vi

Bài nộp chỉ làm **một miền (phản hồi sinh viên) trên dữ liệu tiếng Việt**, ba tác giả (Đoàn Quang Thắng, Trương Công Lý, Phạm Nguyễn Thanh Nam). Tiêu đề làm việc:

> *LLM-assisted relabeling and aspect-aware multi-task learning for Vietnamese student feedback ABSA*

Đóng góp dự kiến:

1. Bộ nhãn 11 khía cạnh cho 16.175 câu UIT-VSFC, kèm quy trình gán nhãn và kết quả rà soát của người.
2. Bảng benchmark nhiều seed, cùng split và cùng độ đo cho các mô hình từ baseline đến PhoBERT.
3. Chẩn đoán lỗi theo khía cạnh và đánh giá PhoBERT Aspect-Query Attention: query attention cho cả hai nhánh, kèm ngưỡng theo khía cạnh. Mục này chỉ được nêu là cải thiện nếu vượt được dao động giữa các seed.

**Mạch bài:** chẩn đoán rồi sửa lỗi, không phải "đề xuất kiến trúc mới". Nếu mức tăng nhỏ, bài vẫn đứng được nhờ bộ dữ liệu, benchmark và phân tích lỗi. Lý do từng quyết định: `06-external-review-decisions.md`.

**Ngoài phạm vi:** đa miền, đa ngôn ngữ (M-ABSA, Coursera, Hotel, §7 bản thảo), baseline LLM và Jev, demo Flask, n8n, thay đổi kiến trúc mới ngoài danh sách thí nghiệm.

## Các cổng quyết định

| Cổng | Ngày | Điều kiện | Nếu không đạt |
|---|---|---|---|
| **G2** Tái hiện (**đạt 01/11**, lệch tối đa 0,35 điểm) | 04/11 | Trên dataset Kaggle cũ (16.124 mẫu) và β = 1,5, code mới đạt gần kết quả notebook: Task A và B cho seed 42 lệch tối đa 1 điểm | Dừng để tìm nguyên nhân; không chạy nhiều seed trên code chưa khớp |
| **G3** Ngân sách GPU (**đã đo thời gian, còn thiếu quota**) | 04/11 | Thời gian đo được đủ cho 21 lượt GPU của lõi P0 trong quota Kaggle | Bỏ P1 trước; nếu chỉ đủ 12–15 lượt, E6a chỉ là pilot báo cáo riêng |
| **G3b** Chốt kiến trúc | 06/11 | So sánh E2, E3, E5, E4 trên validation với 3 seed; khóa mô hình chính và cách chấm trước khi xem test | Áp dụng bảng diễn giải trong protocol; không thêm thành phần mới sau ngày này |
| **G4** Đóng băng số liệu | 09/11 | Có đủ P0 và file kết quả | Chỉ viết những gì đã có, không chạy thêm |
| **G5** Đọc duyệt | 11/11 | Có bản hoàn chỉnh 8 trang gửi người hướng dẫn | Rút gọn phân tích, vẫn gửi trong ngày |

## Lịch theo ngày

### Giai đoạn 1: Chuẩn bị (01–02/11)

| Ngày | Việc | Kết quả |
|---|---|---|
| CN 01/11 | Gửi email cho BTC (`01-fjcai2027-requirements.md`). Tạo tài khoản CMT, kiểm tra các trường khi nộp | Email đã gửi, biết các trường CMT yêu cầu |
| CN 01/11 | Họp nhóm: phân công, báo người hướng dẫn. | Có phân công |
| T2 02/11 | Lấy lại dữ liệu LLM gốc từ Google Sheet và số câu đã qua người rà soát | File dữ liệu và thống kê rà soát |
| T2 02/11 | Người rà soát duyệt `data/annotation/added_51_review.csv`, ưu tiên 6 câu test và 11 câu độ tin cậy thấp (**xong 01/11**, đổi 4 nhãn cảm xúc) | 51 nhãn đã được người duyệt |
| T2 02/11 | Tải `data/Final_Data` đã duyệt lên Kaggle thành phiên bản mới, đối chiếu hash (**xong 01/11**, `thngonquang/vku-feedback-v2`); đưa template lên Overleaf và biên dịch thử | Dữ liệu khớp; template chạy được |
| T2 02/11 | Bắt đầu tách code sang `src/` (**xong 01/11**) | Xem cấu trúc bên dưới |

### Giai đoạn 2: Code và tái hiện (02–04/11)

| Ngày | Việc | Kết quả |
|---|---|---|
| T3 03/11 | Hoàn thành data loader, preprocessing, độ đo và test đơn vị (**xong 01/11**) | Test `unittest` đạt |
| T3 03/11 | Chuyển BiLSTM-CNN và PhoBERT cross-attention (**xong 01/11**). | Hai mô hình chạy được |
| T4 04/11 | Chạy seed 42 trên Kaggle; đo thời gian và tỷ lệ cắt câu; pilot BamiBERT, kiểm tra giấy phép và phiên bản `transformers` (**xong 01/11**, xem `02-current-state-audit.md`). **Cổng G2, G3** | Khớp baseline; có ngân sách GPU |
| T4 04/11 | Khóa mô tả khía cạnh; cài head E5 và E4 (khoảng 50–100 dòng); chạy chẩn đoán lỗi trên validation (**xong 01/11**) | Ma trận nhầm; head chạy được |

### Giai đoạn 3: Thực nghiệm (04–09/11)

| Ngày | Việc | Kết quả |
|---|---|---|
| T5–T6 05–06/11 | Chạy E0 và PhoBERT E2, E3, E5, E4 (3 seed). **Cổng G3b** | Cấu hình khóa |
| T6–T7 06–07/11 | Hoàn tất lõi P0: E6a, E1 (3 seed); thêm seed đồng đều nếu còn quota | File kết quả P0 |
| T7 07/11 | Tính tỷ lệ sửa nhãn LLM từ file rà soát của nhóm | Số liệu rà soát có file nguồn |
| T7–CN 07–08/11 | Chạy P1 theo thứ tự E6b, E7, E8, chỉ khi P0 đã xong | File kết quả P1 |
| T2 09/11 | Tính ngưỡng theo khía cạnh, tổng hợp bảng, kiểm định, phân tích lỗi, hình attention. **Cổng G4** | Bảng và hình cuối |

### Giai đoạn 4: Viết bài (song song từ 04/11, tập trung 09–13/11)

| Ngày | Việc | Kết quả |
|---|---|---|
| 04–07/11 | Viết tiếng Anh: Dataset, Task, Annotation, Method (**bản nháp đầu xong 01/11**, `paper/main.tex`, chưa biên dịch) | Bốn mục đầu |
| 05–08/11 | Rà soát và kiểm tra tài liệu liên quan (khoảng 15–20 nguồn) (**24 nguồn đã đối chiếu thông tin xuất bản 01/11**; nên đọc toàn văn những nguồn chỉ mới đọc tóm tắt) | File `.bib` đã kiểm tra |
| T3 10/11 | Viết Results, Discussion, Introduction, Conclusion, Abstract | Bản đầy đủ đầu tiên |
| T4 11/11 | Rút về tối đa 8 trang. Gửi người hướng dẫn. **Cổng G5** | Bản gửi duyệt |
| T5 12/11 | Sửa theo góp ý; chạy kiểm tra trước khi nộp | Bản gần cuối |
| **T6 13/11** | **Nộp lên CMT**, tải về bản đã nộp và kiểm tra | Email xác nhận của CMT |
| 14–15/11 | Dự phòng; có thể sửa bản nộp nếu CMT còn mở | |

## Cấu trúc code

```text
Final/
├── src/absa/
│   ├── data.py, preprocess.py, metrics.py, aspects.py
│   ├── models/         # bilstm_cnn.py, multitask.py (E2, E3, E5, E4)
│   ├── train.py        # huấn luyện, chọn checkpoint bằng validation
│   ├── evaluate.py     # chọn ngưỡng, so sánh bootstrap; chỉ chấm test khi có --test
│   ├── run_jobs.py     # chạy các lượt song song, mỗi GPU một lượt
│   ├── summarize.py    # bảng nhiều seed, so sánh chính, kết luận theo bảng diễn giải; mặc định validation
│   └── baseline_tfidf.py, audit.py, diagnose.py, dataset_stats.py
├── scripts/build_kaggle_kernel.py, paper_tables.py
├── kaggle/             # notebook đã đẩy theo job
├── tests/
├── results/            # JSON và xác suất theo run (không lưu .pt)
└── paper/              # main.tex, references.bib, tables/; biên dịch trên Overleaf
```

Cấu hình nằm trong preset và tham số CLI, được ghi vào `result.json` của từng run. Job `p0` gồm 21 lượt, chỉ build được khi có `--dataset` và `--data-hash` của dữ liệu đã duyệt, và dừng nếu hash không khớp.

## Phân công

Nhóm tự gán tên vào từng vai trò.

| Vai trò | Trách nhiệm |
|---|---|
| ML | `src/`, Kaggle, kết quả thực nghiệm |
| Dữ liệu | Thống kê gán nhãn, điều khoản dữ liệu |
| Viết bài | Bản tiếng Anh, tài liệu tham khảo, biểu đồ |
| Điều phối | Theo dõi cổng, liên hệ BTC và người hướng dẫn, nộp CMT |

## Rủi ro

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| Không đủ thời gian (12 ngày) | Cao | Làm P0 trước; bỏ P1 đầu tiên khi trễ; các cổng cắt phạm vi; nộp sớm 2 ngày |
| E4 không vượt E2, E3 một cách có ý nghĩa | Trung bình | Áp dụng quy tắc diễn giải đã định trước; bài vẫn đứng được nhờ bộ dữ liệu, benchmark và chẩn đoán lỗi |
| BamiBERT không dùng được do giấy phép hoặc phiên bản thư viện | Trung bình | Kiểm tra 04/11; nếu không được, dùng ViSoBERT làm backbone thứ hai |
| Người rà soát bị neo theo nhãn LLM | Trung bình | Không có gán nhãn độc lập (nhóm quyết định 01/11); nêu rõ trong Limitations |
| Phản biện hỏi độ đồng thuận mà bài không có | Trung bình | Nêu rõ giới hạn; không trình bày tỷ lệ rà soát như độ đồng thuận |
| Không còn dữ liệu LLM gốc hoặc số liệu rà soát | Cao | Không có file nguồn thì không đưa con số vào bài; chỉ mô tả quy trình |
| Hết quota GPU Kaggle | Trung bình | Đo ở G3, giảm theo thứ tự ưu tiên; lưu kết quả theo từng run |
| BTC trả lời muộn về ẩn danh, ngôn ngữ | Trung bình | Áp dụng các mặc định đã chọn |
| Template lỗi hoặc thiếu gói khi biên dịch | Thấp | Biên dịch thử ngay ngày 02/11; có template Word dự phòng |
| Lỗi dịch thuật hoặc văn phong tiếng Anh | Trung bình | Dành ngày 12/11 để đọc lại; dùng đúng thuật ngữ từ các bài đã công bố |

## Kiểm tra trước khi nộp

- [ ] Dùng template của hội thảo, không sửa header, footer và định dạng
- [ ] 6–8 trang, kể cả tài liệu tham khảo, trong PDF cuối
- [ ] Viết hoàn toàn bằng tiếng Anh (hoặc theo trả lời của BTC)
- [ ] Ẩn tên, đơn vị, lời cảm ơn, link repo nếu ẩn danh
- [ ] Mỗi số trong bảng khớp một file trong `results/`
- [ ] Threshold và checkpoint đều chọn trên validation
- [ ] Mỗi khẳng định đều có thí nghiệm hoặc nguồn tương ứng
- [ ] Đã kiểm tra từng tài liệu tham khảo
- [ ] Cả ba tác giả đã đồng ý với bản cuối và có trên CMT, email chính xác
- [ ] Bản nộp chưa từng công bố hoặc đang nộp nơi khác
- [ ] Tải bản đã nộp về và kiểm tra font, hình, bảng

## Sau khi nộp

| Mốc | Việc |
|---|---|
| Đến 10/01/2027 | Chuẩn bị công bố dữ liệu |
| 10/01–01/02/2027 | Xử lý góp ý phản biện, bật thông tin tác giả và lời cảm ơn, nộp camera-ready |
| Trước 08/04/2027 | Đăng ký, nộp lệ phí, chuẩn bị bài trình bày |
| Sau hội thảo | Nếu được mời, viết bản mở rộng cho tạp chí |
