# Quyết định về góp ý bên ngoài (01/11/2026)

Góp ý đề xuất thay mean pooling bằng attention theo từng khía cạnh, kèm so sánh nhiều encoder, baseline LLM và Jev. Đã đối chiếu với code và một phần nguồn trích dẫn.

## Đối chiếu với code

| Góp ý nói | Thực tế |
|---|---|
| Mô hình dùng một vector mean pooling cho cả 11 khía cạnh, mốc 78,47 | Đúng với bản cũ. Bản hiện tại đã dùng 11 query với cross-attention cho nhánh cảm xúc, đạt 79,33; chỉ nhánh phát hiện còn mean pooling |
| Che loss cảm xúc gây lệch giữa huấn luyện và suy luận | Không ảnh hưởng F1 cặp: khía cạnh dự đoán sai đã là FP bất kể cảm xúc |
| Lịch 19 ngày từ 27/10 | Giai đoạn đó chưa có trong repo; còn 14 ngày tính từ 01/11 |
| Cổng 0,3 F1 trên một seed | Nhỏ hơn dao động dự kiến giữa các seed; thay bằng ít nhất 3 seed trên validation |
| Experience, curriculum, knowledge nhầm lẫn với nhau | Chưa có ma trận nhầm để kết luận; đo ở bước chẩn đoán |

## Quyết định

| Nội dung | Quyết định |
|---|---|
| Mạch bài "chẩn đoán rồi sửa lỗi" | **Dùng**, sau khi có số đo |
| Query attention cho nhánh phát hiện, query từ tên khía cạnh | **Dùng** (E4, E5) |
| Ngưỡng theo khía cạnh | **Dùng**, báo cáo tách khỏi lợi ích kiến trúc |
| Không dùng GPT-OSS làm baseline | **Dùng** |
| So sánh encoder | **Thu hẹp:** BamiBERT trong lõi; ViSoBERT, PhoBERT-base-v2, mBERT ở P1; không dùng encoder lớn (sửa theo nhận xét của Codex) |
| Mô hình đề xuất trên ít nhất hai backbone | **Dùng** |
| BERT-pair, head 4 lớp, một loss xử lý chồng lấn | **Để sau**, chỉ làm nếu còn thời gian (P2) |
| Jev, Qwen, Gemini, R-Drop, tăng cường dữ liệu, ViT5 | **Bỏ** cho bài này |
| Gán nhãn đối chứng | **Không làm:** nhóm quyết định 01/11, nêu giới hạn trong bài |

## Trạng thái trích dẫn

Trạng thái này chỉ cho biết số liệu trên trang nguồn có khớp hay không. Trước khi đưa vào `.bib` vẫn phải đối chiếu thông tin xuất bản.

**`paper/references.bib` (01/11):** 24 nguồn, lấy từ bản ghi DOI, arXiv và ACL Anthology, không gõ tay; riêng [2] đối chiếu với PDF. Mức đã đọc khác nhau:

- **Đọc toàn văn:** [2].
- **Đọc tóm tắt:** CAN, AC-MIMLLN, Sun et al., BamiBERT, NEU-ESC, Donhauser et al., Schroeder et al., Pangakis & Wolken.
- **Chỉ đối chiếu thông tin xuất bản:** các nguồn còn lại, gồm LSAN, Kastrati et al., Van Thin et al., EduPulse. Trước khi nộp nên đọc toàn văn các nguồn mà bài dùng để lập luận.

**Điều khoản UIT-VSFC (kiểm tra 01/11):** repo GitHub không có file giấy phép, README chỉ yêu cầu trích dẫn; các bản sao trên Hugging Face ghi `unknown`. Không thấy điều khoản cấm, nhưng cũng không có văn bản cho phép phân phối lại. Quyết định: khi công bố, chỉ phát hành **nhãn kèm vị trí câu trong UIT-VSFC**, không kèm văn bản câu, và trích dẫn bài gốc. Cách này không cần xin phép và không ảnh hưởng đến bản nộp.

**Sửa so với bản thảo cũ:** bản thảo ghi "Am" 88,95% cho [2], nhưng bài [2] không dùng tên độ đo này. Câu "bộ dữ liệu không công khai" được đổi thành "không tìm thấy bản công khai", vì [2] từng hứa công bố và tìm kiếm ngày 01/11 không thấy link.

| Nguồn | Trạng thái | Lưu ý khi dùng |
|---|---|---|
| Ton Nu Thi Sau, Do Phuoc Sang, Pham Thi Thu Trang, TNU JST 226(18):48–55, 2021, DOI 10.34238/tnu-jst.5101 | **Đã đọc bản PDF, thông tin khớp** | Bài [2]. Nguồn của tên 11 khía cạnh và baseline BiLSTM-CNN; không so số F1 trực tiếp |
| VLUE/CafeBERT (arXiv 2403.15882) | Số khớp | Cảm xúc câu trên UIT-VSFC, không phải ABSA 11 khía cạnh |
| BamiBERT (arXiv 2607.02259) | Số khớp | Như trên; có model trên Hugging Face, kèm giấy phép Qualcomm |
| AC-MIMLLN (arXiv 2010.02656) | Số khớp | Accuracy trên Rest14-hard, độ lệch chuẩn 3–6 điểm; không ngoại suy thành mức tăng F1 cặp |
| Sun et al. 2019 (arXiv 1903.09588) | Số khớp | 81,0 → 87,9 là Macro-F1 trên SentiHood |
| SCAPT (arXiv 2111.02194) | Số khớp (2,38 accuracy) | Là contrastive pre-training, không phải bằng chứng cho SupCon thêm vào loss |
| PhoBERT-base-v2 (Hugging Face) | Khớp | AGPL-3.0; khuyến nghị dùng cùng bộ tách từ RDRSegmenter |
| Vals AI về Jev | Số khớp | Không dùng trong bài |
| Liu et al. 2021 | Bài có thật, **chưa thấy** số 77,36 | Mở PDF trước khi trích |
| PACLIC 2022, PACLIC 2024 | **Link 404** | Tìm link chính thức; chưa dùng các số trong góp ý |
| Thin & Nguyen 2023 (ViT5) | **Không đọc được** số liệu | Đọc lại trước khi trích |
| LSAN, CAN, Huang et al. 2021, Schmitt et al., ASL, R-Drop, Panickssery et al., Wang et al. 2023, Simmering & Huoviala, ACM TALLIP 2023, ViSoBERT và các nguồn khác | **Chưa kiểm tra** | Đối chiếu trước khi dùng |

Không viết "đầu tiên" hay "chưa có công trình ACSA tiếng Việt nào" trong bài nếu chưa khảo sát có hệ thống. Mức mục tiêu 80–81 F1 chỉ là ước lượng của góp ý, không dùng làm cam kết.
