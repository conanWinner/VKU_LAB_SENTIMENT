# Quyết định phạm vi sau phản hồi Claude

Ngày: 01/11/2026  
Phạm vi: bài FJCAI 2027 và hai file nghiên cứu `07`–`08`.

> **Cập nhật 01/11/2026:** người dùng đã đồng ý đồng bộ. Protocol chính thức là `04-experiment-protocol.md`; file này chỉ để tham khảo. Hai điểm đã đổi so với nội dung bên dưới:
>
> 1. **51 câu đã được bổ sung** theo yêu cầu người dùng: `data/Final_Data` hiện có 16.175 mẫu, truy vết trong `data/annotation/added_51_review.csv`. Lý do thiếu ban đầu không còn rõ; người dùng cho rằng có thể do câu ngắn hoặc bị sót. Nhãn cần người duyệt và bài phải ghi rõ nguồn nhãn.
> 2. **"0,5" có hai nghĩa riêng:** mức chênh thực tiễn tối thiểu là 0,5 điểm F1; ngưỡng quyết định 0,5 chỉ để tham khảo.

## Quyết định đã nhận

| Hạng mục | Quyết định |
|---|---|
| Thí nghiệm lõi | Giữ 21 lượt GPU: PhoBERT E2–E5 (12), BamiBERT E2/E4 (6), BiLSTM-CNN (3), mỗi cấu hình 3 seed. TF-IDF là baseline CPU. |
| Seed bổ sung | Chỉ thêm khi đo xong còn quota; thêm đồng đều cho các cặp so sánh chính, không chỉ cho model thắng. |
| Tuning encoder | Giữ một recipe cố định để kịp hạn. Bài phải ghi đây là so sánh có kiểm soát, không phải bảng xếp hạng encoder sau tối ưu riêng. |
| Encoder P1 | Chỉ giữ ViSoBERT, PhoBERT-base-v2 và mBERT; không thêm encoder lớn. |
| Ngoài bài | GoiBERT, SEA-LION, HuTieuBERT, distillation, Qwen/QLoRA, retrieval few-shot, tăng dữ liệu và chuyển người rà soát. Head bốn trạng thái giữ ở P2/E10. |
| Loss/threshold | Giữ β = 1,0 và ngưỡng 0,5 như mốc tham khảo; không tự thay protocol `04`. Mọi threshold được chọn theo validation trước khi chấm test. |
| Kiểm soát ghi chép | Mỗi run ghi tỷ lệ truncation theo tokenizer, model revision/hash, split hash, preprocessing, seed, config và peak memory. |
| Nhãn đối chứng | Dùng tên **annotation audit**. Không gọi nhãn cũ là gold độc lập; báo rõ người rà soát, mẫu, bất đồng và giới hạn. |
| Mô tả aspect | Viết từ annotation guideline và khóa trước khi xem lỗi test; không dùng ví dụ test để tạo query hoặc định nghĩa lại aspect. |

## Đối chiếu 51 câu vắng mặt

So sánh văn bản giữa `data/{train,validation,test}.csv` và các file trong `data/Final_Data` bằng bộ đếm văn bản cho thấy:

| Split gốc | Số dòng gốc | Số nhóm văn bản trong `Final_Data` | Vắng mặt |
|---|---:|---:|---:|
| Train | 11.426 | 11.389 | 37 |
| Validation | 1.583 | 1.575 | 8 |
| Test | 3.166 | 3.160 | 6 |
| **Tổng** | **16.175** | **16.124** | **51** |

Không thấy văn bản được thêm hoặc thay thế khi đối chiếu exact text. Danh sách vắng mặt có nhiều câu rất ngắn hoặc ít thông tin, ví dụ `không có ạ !`, `tạm .`, `giáo viên .`, nhưng cũng có câu có thể còn nghĩa như `sức khỏe yếu .`, `công dụng nó là gì ?`, `điều này xảy ra ở mọi tiết học .`. Vì vậy hiện chỉ kết luận được **51 câu không xuất hiện trong dữ liệu đã relabel**, chưa kết luận được tiêu chí loại.

Bài báo phải trả lời được một trong hai cách:

1. Tìm lại log/prompt/guideline tạo `Final_Data` và công bố tiêu chí, số câu bị loại theo tiêu chí, người quyết định và thời điểm; hoặc
2. Nếu không còn bằng chứng, ghi là *unresolved filtering provenance*, báo 51 câu là chênh lệch giữa dataset gốc và subset relabel, không gọi là “câu vô nghĩa” và không tuyên bố đã relabel toàn bộ UIT-VSFC.

Không tự khôi phục, xóa hoặc sửa 51 câu trong lượt này. Nếu cần thay dataset, phải tạo manifest `original_split, original_id, new_id, keep_reason, evidence` và xin quyết định dữ liệu riêng.

## Các mục chưa xác nhận

- GPU, VRAM và quota Kaggle chưa được đọc trực tiếp trong tài khoản; không ghi “hai GPU 16 GB” như sự thật đã kiểm chứng.
- Tỷ lệ truncation theo tokenizer chưa được đo; chỉ có yêu cầu ghi nhận trong protocol run.
- Hash split/revision model chưa tồn tại vì chưa có run mới.
- `docs/04-experiment-protocol.md` chưa bị sửa trong lượt này; file 07 và 09 ghi quyết định phạm vi, không biến đề xuất thành kết quả đã chạy.
