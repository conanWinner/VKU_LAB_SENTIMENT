# Dàn ý bài báo

Bài viết tiếng Anh, tối đa 8 trang theo template hai cột của hội thảo. Số trang dưới đây là phân bổ dự kiến và cần điều chỉnh sau lần biên dịch đầu tiên.

**Tiêu đề làm việc:** *LLM-Assisted Relabeling and Aspect-Query Attention for Vietnamese Student Feedback ABSA*

Nếu E4 không tốt hơn rõ so với E2 và E3, đổi thành dạng *"A Multi-Aspect Dataset, Benchmark and Error Analysis for Vietnamese Student Feedback ABSA"*.

**Mạch bài:** bộ dữ liệu → chẩn đoán lỗi → sửa bằng biểu diễn riêng cho từng khía cạnh và ngưỡng theo khía cạnh → đánh giá nhiều seed.

| Mục | Trang | Nội dung | Lấy từ bản thảo |
|---|---|---|---|
| Abstract, Keywords | 0,3 | Bài toán, dữ liệu, phương pháp, **số kết quả thực tế**. Viết sau cùng | Viết lại; bản cũ chưa nêu số |
| I. Introduction | 0,6 | Nhu cầu; 4 chủ đề của UIT-VSFC chưa đủ chi tiết; bộ 11 khía cạnh của [2] chỉ có 5.010 câu ([2] hứa công bố; tìm lại trước khi viết là chưa công khai); 3 đóng góp | §1 |
| II. Related Work | 0,7 | Bộ dữ liệu phản hồi sinh viên; ABSA tiếng Việt; MTL và cơ chế attention theo khía cạnh; LLM hỗ trợ gán nhãn | §2, bổ sung nhiều |
| III. Dataset and Annotation | 1,7 | Hệ 11 khía cạnh (bảng rút gọn); quy trình LLM và người rà soát (1 hình); thống kê; 51 câu gán nhãn bổ sung và nguồn nhãn; **kết quả rà soát của người** | §3, §4, gộp lại |
| IV. Method | 1,2 | Định nghĩa bài toán; encoder; aspect query từ tên khía cạnh; hai nhánh dùng biểu diễn riêng; hàm loss; ngưỡng (1 hình) | §3.3, §5, sửa theo code thật |
| V. Experimental Setup | 0,6 | Mô hình so sánh, siêu tham số, chọn threshold, độ đo, seed | Mới |
| VI. Results and Discussion | 2,0 | Chẩn đoán lỗi; bảng tổng; ablation; so sánh encoder; F1 theo khía cạnh (1 hình); phân tích lỗi | §6, viết lại |
| VII. Conclusion | 0,2 | Tóm tắt, giới hạn (gồm neo theo nhãn LLM), hướng phát triển | §8 |
| Acknowledgment | Vài dòng | Những người góp phần gán nhãn, rà soát và hướng dẫn; tắt ở bản ẩn danh | Viết lại |
| References | 0,7 | 15–20 nguồn | §Tài liệu, mở rộng |

Bỏ khỏi bài: toàn bộ §7 (đa miền, đa ngôn ngữ); Bảng 4 (ví dụ tách dòng); các phương trình tập hợp (1)–(3), gộp thành một đoạn; ảnh chụp giao diện gán nhãn.

## Hình và bảng

| Loại | Nội dung | Nguồn |
|---|---|---|
| Hình 1 | Quy trình gán nhãn | Vẽ lại từ Hình 1 bản thảo |
| Hình 2 | Kiến trúc mô hình **thực sự được đánh giá** | Vẽ mới theo code |
| Hình 3 | F1 theo khía cạnh, kèm số mẫu train | `results/` |
| Bảng I | 11 khía cạnh, mô tả ngắn và số lượng | Bảng 3, 6 bản thảo |
| Bảng II | Thống kê train, validation, test | Tính lại từ `Final_Data` |
| Bảng III (tùy chọn) | Tỷ lệ chấp nhận và sửa nhãn; không có κ | Cần file rà soát |
| Bảng IV | Kết quả chính: Task A và B, trung bình ± độ lệch chuẩn | `results/` |
| Bảng V | Ablation (E2, E3, E5, E4; ngưỡng chung và theo khía cạnh) và so sánh trên backbone khác (E6a, E6b nếu có), ghi rõ cấu hình cố định; gộp một bảng nếu thiếu chỗ | `results/` |
| Hình 4 (tùy chọn) | Ma trận nhầm giữa các khía cạnh hoặc minh họa attention | `results/` |

## Bộ khung cho đóng góp

Chỉ giữ câu nào có kết quả tương ứng khi đến G4.

1. *We relabel the full UIT-VSFC corpus (16,175 sentences, 23,910 aspect–sentiment pairs) with an 11-aspect scheme adapted from [2], using an LLM-assisted human-in-the-loop workflow in which reviewers checked every sentence.*
2. *We provide a multi-seed benchmark from classical models to multi-task Transformers on a fixed split with consistent metrics.*
3. *We diagnose per-aspect errors of a shared-representation model and evaluate aspect-query attention for both aspect detection and sentiment classification, together with per-aspect thresholds, through an ablation study.*

Nếu cam kết công khai dữ liệu trong bài, cần xác nhận được phép phân phối và có link hoạt động. Khi ẩn danh, thay link bằng "available upon publication".

## Tài liệu tham khảo cần có

**Đã có trong bản thảo (kiểm tra lại thông tin):** UIT-VSFC (Nguyen et al., KSE 2018); Sau et al. (TNU JST, 2021); Tsoumakas & Katakis (2007). Bỏ M-ABSA vì không còn dùng.

**Cần bổ sung, có thể tra theo tên:**

- BERT (Devlin et al., NAACL 2019)
- PhoBERT (Nguyen & Nguyen, Findings of EMNLP 2020)
- SemEval ABSA (Pontiki et al., 2014, 2016)
- Multi-task learning (Caruana, 1997)
- Attention (Vaswani et al., 2017)

**Từ góp ý bên ngoài:** xem trạng thái từng nguồn trong `06-external-review-decisions.md`. Các nguồn đã khớp số liệu: VLUE/CafeBERT, BamiBERT, AC-MIMLLN, Sun et al. 2019. Cần đọc: LSAN và CAN cho attention theo nhãn, ViSoBERT nếu dùng trong E6b, Donhauser et al. 2026 (arXiv 2605.03624) cho chất lượng nhãn, NEU-ESC (arXiv 2506.23524) cho Related Work.

**Cần tìm:**

- ABSA tiếng Việt và ABSA trong giáo dục những năm gần đây
- MTL hoặc joint models cho ABSA
- Aspect-aware attention trong ACSA
- LLM hỗ trợ gán nhãn dữ liệu, đặc biệt là đánh giá chất lượng nhãn
- Công cụ tách từ dùng trong code (underthesea)

Không đưa nguồn nào vào `.bib` khi chưa mở được bài gốc hoặc trang xuất bản để đối chiếu tác giả, năm và nơi đăng.

## Bảng thuật ngữ

Thống nhất trước khi viết.

| Bản thảo | Trong bài |
|---|---|
| Khía cạnh | aspect category |
| Trạng thái cảm xúc | sentiment polarity |
| Phát hiện khía cạnh | aspect category detection (ACD) |
| Phát hiện khía cạnh kèm cảm xúc | aspect category sentiment detection, ghi rõ định nghĩa cặp |
| Học đa nhiệm, Phobert–MLT | multi-task learning; PhoBERT-MTL |
| `ky_nang_giang_day` | Teaching Skill |
| `kinh_nghiem` | Experience |
| `hanh_vi` | Behavior |
| `bai_tap` | Assignment |
| `cham_diem` | Grading |
| `cung_cap_tai_lieu` | Material Provision |
| `kien_thuc` | Knowledge |
| `chuong_trinh_hoc` | Curriculum |
| `thiet_bi_day_hoc` | Teaching Equipment |
| `de_xuat` | Suggestion |
| `noi_chung` | General |

Đã đối chiếu với [2]: bảng khía cạnh của [2] chỉ có tiếng Việt, phần tóm tắt tiếng Anh chỉ nêu "teacher behavior, teaching skills", nên tên tiếng Anh ở trên là do mình đặt.

Phần Dataset phải nêu ba khác biệt so với [2]: một câu góp ý được gán cả khía cạnh cụ thể và `de_xuat`; góp ý trung lập mang nhãn neutral; LLM nhận nhãn UIT-VSFC làm gợi ý. Bảng so sánh ở `02-current-state-audit.md`.
