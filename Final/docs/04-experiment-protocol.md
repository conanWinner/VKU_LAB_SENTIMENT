# Thiết kế thực nghiệm

**Đây là nguồn chính thức duy nhất của protocol.** `07`–`09` chỉ để tham khảo. Chốt tài liệu này trước khi chạy nhiều seed; nếu thay đổi sau đó, ghi lý do ở cuối file. Lý do chọn hoặc bỏ từng hướng nằm ở `06-external-review-decisions.md`.

## Câu hỏi nghiên cứu

| RQ | Câu hỏi | Thí nghiệm |
|---|---|---|
| RQ1 | Quy trình LLM và người rà soát tạo ra bộ nhãn như thế nào, và người đã sửa bao nhiêu? | Thống kê rà soát (không phải độ đồng thuận) |
| RQ2 | Lỗi của mô hình hiện tại tập trung ở đâu? | Chẩn đoán trên validation |
| RQ3 | Biểu diễn riêng cho từng khía cạnh có cải thiện so với vector câu dùng chung không? | E2, E3, E5, E4 |
| RQ4 | Ngưỡng riêng theo khía cạnh giúp thêm bao nhiêu? | Hậu xử lý trên các mô hình lõi |
| RQ5 | So với baseline và trên backbone khác thì thế nào? | E0, E1, E6a; E6b nếu còn quota |
| RQ6 | MTL có hơn huấn luyện riêng? Tăng dữ liệu có giúp không? | E7, E8 |

## Thiết lập chung

- **Dữ liệu:** chỉ dùng `data/Final_Data` (tiếng Việt, một miền), giữ nguyên split gốc: 16.175 mẫu, 23.910 cặp. 51 câu bổ sung phải được người duyệt trước run chính. Không dùng M-ABSA, Coursera, Hotel. Đọc từng file train, group theo `id`, rồi nối lại.
- **Bài toán:** 11 khía cạnh (multi-label), 3 cảm xúc cho mỗi khía cạnh có mặt.
- **Seed:** 42, 43, 44 cho mọi cấu hình. Nếu đo xong còn quota, thêm 45, 46 **đồng đều** cho các cặp so sánh chính, không chỉ cho mô hình thắng.
- **Chọn checkpoint:** `pair_f1_micro` trên validation, cho mọi mô hình. Không dùng `val_combined_f1` của notebook vì nó ghép `aspect_f1` với `sent_f1`, không phải F1 cặp.
- **Siêu tham số:** một cấu hình cố định cho mọi mô hình Transformer: 8 epoch, LR 2e-5, batch 16, độ dài 128, patience 2, dropout 0,3, trọng số cảm xúc `[1, 2, 1]`, `pos_weight` như notebook. Khi đổi encoder chỉ đổi encoder và tiền xử lý bắt buộc. Bài phải ghi đây là so sánh có kiểm soát, **không phải bảng xếp hạng encoder đã tối ưu riêng**. BiLSTM-CNN giữ cấu hình baseline của nó.
- **Trọng số loss:** cố định α = 1,0 và β = 1,0 cho mọi mô hình đa nhiệm, không dò. Giá trị 1,5 trong notebook chỉ dùng ở bước tái hiện (G2).
- **Tiền xử lý:** PhoBERT dùng `underthesea` để tái hiện code hiện tại; ghi phiên bản. RDRSegmenter do trang PhoBERT khuyến nghị được kiểm tra ở E13. BamiBERT nhận văn bản thô đã chuẩn hóa Unicode, không thêm dấu gạch dưới; pin `transformers<=5.5.0` theo model card. Với encoder khác, làm theo trang model. Đo tỷ lệ câu bị cắt theo từng tokenizer trước khi chạy.
- **Ghi kết quả:** mỗi run một JSON gồm hash split, commit, model ID và revision, phiên bản tokenizer và tiền xử lý, seed, config, tiêu chí checkpoint, kiểu và giá trị ngưỡng, độ đo validation và test, dự đoán theo id, thời gian chạy, bộ nhớ đỉnh. Lưu xác suất trên validation và test để tính ngưỡng và kiểm định mà không phải chạy lại.

**Về tập test:** các kết quả cũ (86,38; 87,10; 79,33...) đã được xem trong quá trình phát triển và chạy trên bộ 16.124 mẫu. Từ nay mọi lựa chọn chỉ dựa trên validation, và chỉ chấm test sau khi đã khóa cấu hình. Không bổ sung kiến trúc sau khi xem test.

## Mô hình đề xuất: PhoBERT Aspect-Query Attention (E4)

So với `AspectAwareBertMultiTaskABSA` hiện tại (E3), chỉ có hai thay đổi, được đo tách từng bước qua E3 → E5 → E4:

1. **Nhánh phát hiện khía cạnh** cũng dùng biểu diễn riêng của từng khía cạnh thay cho mean pooling: `V = cross_attention(Q, H)`, rồi một lớp tuyến tính cho mỗi khía cạnh (hoặc dùng chung trọng số, chốt khi code). Bước này là E5, query khởi tạo ngẫu nhiên.
2. **Query khởi tạo từ tên khía cạnh:** lấy mean pooling bằng chính encoder đang dùng cho mô tả tiếng Việt của khía cạnh, rồi vẫn cho học tiếp. Bước này là E4.

Mô tả khía cạnh viết từ annotation guideline và **khóa trước khi chạy chẩn đoán lỗi**; không dùng ví dụ từ validation hoặc test. Giữ nguyên cách che loss cảm xúc theo khía cạnh thật. Không thêm thành phần nào khác vào E4 trừ khi đổi protocol.

## Độ đo (`src/absa/metrics.py`)

| Tên | Định nghĩa |
|---|---|
| `aspect_{p,r,f1}_micro`, `aspect_f1_macro` | Task A: phát hiện khía cạnh |
| `pair_{p,r,f1}_micro` | **Task B:** một dự đoán chỉ đúng nếu đúng cả khía cạnh và cảm xúc. Đây là số tương ứng với Bảng 9 của bản thảo |
| `pair_f1_macro` | Trung bình theo 11 khía cạnh của F1 cặp |
| `sentiment_acc_on_gold`, `sentiment_f1_macro_on_gold` | Cảm xúc trên các khía cạnh có thật, tách khỏi lỗi phát hiện |
| `per_aspect` | P, R, F1 của Task A và Task B cho từng khía cạnh |

Test đơn vị tối thiểu:

- Loader không gộp `id` giữa các file train.
- Sai cảm xúc tính một FP và một FN cho F1 cặp.
- Thứ tự đầu ra không đổi điểm.
- Ví dụ nhỏ tính được bằng tay, rồi đối chiếu với kết quả notebook hiện tại.

## Ngưỡng

Tính sau huấn luyện từ xác suất đã lưu, không tốn thêm GPU. Quy tắc được định trước và khóa trên validation; test chỉ để báo cáo, **không dùng để chọn kiểu ngưỡng**.

| Kiểu | Cách chọn trên validation | Vai trò |
|---|---|---|
| Ngưỡng chung | Một giá trị từ 0,20 đến 0,70, bước 0,05, tối đa `pair_f1_micro` | **Kết quả chính khi so sánh mô hình** |
| Ngưỡng theo khía cạnh | Lần lượt cho từng khía cạnh, cùng lưới, tối đa F1 cặp của khía cạnh đó | Phân tích phụ, báo cáo cho mọi mô hình lõi, tách khỏi lợi ích kiến trúc |
| Ngưỡng 0,5 | Cố định | Chỉ để tham khảo và nối với kết quả cũ |

Validation chỉ có 43 mẫu `cham_diem`, nên ngưỡng của khía cạnh này dễ quá khớp.

## Danh sách thí nghiệm

| ID | Mô hình | Mục đích | Ưu tiên |
|---|---|---|---|
| E0 | TF-IDF + Linear SVM | Mốc đơn giản, chạy trên CPU | P0 |
| E1 | BiLSTM-CNN phỏng theo [2], khác biệt ghi ở `02-current-state-audit.md` | Baseline chính | P0 |
| E2 | PhoBERT MTL, mean pooling cho cả hai nhánh | Mô hình vector câu dùng chung | P0 |
| E3 | PhoBERT, query attention chỉ cho cảm xúc, query ngẫu nhiên (code hiện tại) | Đo đóng góp của attention cho nhánh cảm xúc | P0 |
| E5 | E3 + query attention cho nhánh phát hiện, query ngẫu nhiên | Đo đóng góp của attention cho nhánh phát hiện | P0 |
| E4 | **Đề xuất:** E5 + query khởi tạo từ tên khía cạnh | Đo đóng góp của khởi tạo theo ngữ nghĩa | P0 |
| E6a | E2 và E4 trên BamiBERT | Kiểm tra lợi ích trên backbone thứ hai | P0 |
| E6b | E4 trên ViSoBERT, PhoBERT-base-v2, mBERT | So sánh encoder dưới cấu hình cố định | P1 |
| E7 | Hai mô hình PhoBERT độc lập, một cho từng nhiệm vụ | MTL có hơn không | P1 |
| E8 | E1, E4 trên 5.000 câu lấy ngẫu nhiên từ train, cùng test | Tác động của quy mô dữ liệu | P1 |
| E9 | BERT-pair trên PhoBERT | Baseline theo từng khía cạnh; thời gian gấp khoảng 11 lần | P2 |
| E10 | Head 4 lớp mỗi khía cạnh (none/pos/neg/neu) | Ablation | P2 |
| E11 | E4 thêm ràng buộc trực giao giữa các attention | Một loss xử lý chồng lấn | P2 |
| E12 | E4 bỏ `pos_weight` | Tác động lên khía cạnh ít dữ liệu | P2 |
| E13 | E4 với RDRSegmenter | Tác động của bộ tách từ; trước tiên đo tỷ lệ câu bị tách khác nhau, chỉ huấn luyện nếu tỷ lệ này đáng kể | P2 |

**Khối lượng:** lõi P0 là **21 lượt GPU** (PhoBERT E2, E3, E5, E4: 12; E6a: 6; E1: 3), chưa tính pilot và lượt lỗi. P1 khoảng 21 lượt. Tính ngân sách theo thời gian đo thật ở G3.

**Nếu chỉ đủ 12–15 lượt:** giữ PhoBERT E2, E3, E5, E4 với 3 seed; E6a chỉ là pilot báo cáo riêng và thu hẹp kết luận về backbone.

**Về E6a:** chưa có E5 trên BamiBERT thì không kết luận riêng về khởi tạo query trên backbone này.

**Về E8:** **không dùng `data/paper_subset`** vì tập này chứa câu thuộc test của `Final_Data`. Cố định danh sách 5.000 câu bằng một seed riêng và lưu vào `results/`.

**Về BiLSTM-CNN:** dùng bản không Focal Loss. Đã đối chiếu với [2]: kiến trúc giống, nhưng tốc độ học, class weight, dropout và tiền xử lý khác, nên trong bài viết "adapted from [2]". **Không so số F1 với [2]** vì quy tắc gán nhãn khác nhau; chỉ so xu hướng theo khía cạnh ở mức định tính.

**Ngoài bài này:** GoiBERT, SEA-LION, HuTieuBERT, encoder lớn, distillation, Jev, Qwen/QLoRA, Gemini và các LLM zero/few-shot, retrieval few-shot, ViT5, R-Drop, tăng cường dữ liệu, contrastive pre-training, GCN, capsule, chuyển người rà soát. GPT-OSS không được dùng làm baseline trong mọi trường hợp vì nhãn ban đầu do chính nó sinh ra.

## So sánh thống kê

So sánh chính: E3–E2, E5–E3, E4–E5, E4–E2 và E4–baseline tốt nhất; trên BamiBERT là E4–E2. Hai mô hình của một so sánh phải dùng **cùng bộ seed**.

1. Báo cáo trung bình ± độ lệch chuẩn qua các seed, và mức chênh `pair_f1_micro` tính bằng điểm F1.
2. Khoảng tin cậy 95%: bootstrap có ghép cặp, ít nhất 1.000 lần lặp. Mỗi lần lấy lại **cùng chỉ số câu** cho hai mô hình, tính lại micro-F1 từ số đếm TP/FP/FN cho từng seed, rồi lấy trung bình mức chênh qua các seed. Không coi seed × câu là mẫu độc lập.
3. Nói rõ khoảng tin cậy này có điều kiện trên các seed đã chạy; với 3 seed, bất định giữa seed vẫn lớn.

**Mức chênh có ý nghĩa thực tiễn được định trước: 0,5 điểm F1.** Con số này khác với ngưỡng quyết định 0,5 ở trên.

| Kết quả so sánh | Cách viết trong bài |
|---|---|
| Khoảng tin cậy không chứa 0, mức chênh ≥ 0,5 điểm F1 | "Cải thiện" |
| Khoảng tin cậy không chứa 0, mức chênh < 0,5 điểm F1 | "Khác biệt nhỏ, ý nghĩa thực tiễn hạn chế" |
| Khoảng tin cậy chứa 0 | "Chưa đủ bằng chứng về khác biệt". **Không viết "tương đương"** vì chưa có kiểm định tương đương |

Cách trình bày mô hình trong bài:

| Kết quả | Cách trình bày |
|---|---|
| E4 cải thiện so với E2 | Trình bày E4 là mô hình đề xuất, nêu từng bước đóng góp qua E3, E5 |
| Cải thiện chỉ đến ở một bước | Trình bày đúng thành phần đó; mô hình chính là mô hình đơn giản nhất đạt mức này |
| Không có cải thiện | Trình bày là nghiên cứu chẩn đoán và benchmark; không gọi là cải tiến |
| Backbone giúp nhiều, E4 không hơn E2 trên BamiBERT | Lợi ích chủ yếu từ backbone; thu hẹp kết luận về attention |

Cổng G3b chỉ dùng validation với 3 seed để khóa cấu hình và mô hình chính, không dùng test và không dùng một seed.

## Chất lượng nhãn (RQ1)

**Nhóm quyết định không làm annotation audit (01/11/2026).** Lý do: nhóm đã dò toàn bộ dữ liệu và báo kết quả đạt trên 99%. Phiếu trong `data/annotation/audit/` được giữ nhưng không dùng.

Cách viết trong bài:

1. Mô tả quy trình: người rà soát kiểm tra từng câu, sửa hoặc xóa câu chưa đúng. Không có file so sánh nhãn LLM và nhãn cuối nên **không nêu con số "trên 99%" trong bài**; nếu tìm lại được file thì mới báo cáo tỷ lệ sửa.
2. **Không gọi con số này là độ đồng thuận, không viết κ và không gọi nhãn là gold độc lập.** Người rà soát đã thấy gợi ý của LLM nên có thể bị neo theo. Mức khớp với nhãn cảm xúc UIT-VSFC cũng không phải bằng chứng chất lượng, vì prompt đã đưa nhãn đó cho LLM làm gợi ý.
3. Phần Limitations nêu rõ: không có gán nhãn độc lập hai người, và hai rủi ro ở mục 2. Phân tích `de_xuat` chỉ được viết là giả thuyết.

Cũng cần ghi lại: tên chính xác của LLM, phiên bản, ngày chạy và prompt để đưa vào bài hoặc phần phụ lục dữ liệu.

**51 câu bổ sung:** nhóm cho biết đã xóa các câu này khi rà soát từng câu (phần lớn rất ngắn: 33/51 câu có tối đa 4 token). Ngày 01/11/2026 chúng được đưa lại với nhãn do Claude đề xuất và người duyệt theo `data/annotation/added_51_review.csv`. **Giữ trong dữ liệu**, vì bỏ câu sau khi đã rà nhãn là lọc theo nhãn và làm test dễ hơn; bỏ lại cũng phải tạo dataset mới và chạy lại 21 lượt. Bài phải ghi rõ nguồn nhãn này, không mô tả như đi qua cùng quy trình với các câu còn lại.

## Chẩn đoán và phân tích lỗi

**Chẩn đoán (RQ2), trên validation, trước khi khóa cấu hình:** dùng E3 seed 42.

- Với mỗi khía cạnh thật bị bỏ sót, đếm khía cạnh sai được dự đoán thêm trong cùng câu; tạo ma trận nhầm 11 × 11.
- Đặt cạnh đó tần suất đồng xuất hiện trong train, để phân biệt "nhầm" với "hay đi cùng nhau". Trong train, `kinh_nghiem` và `kien_thuc` chỉ cùng xuất hiện 87 câu.
- Tách lỗi theo FP, FN, sai cảm xúc cho từng khía cạnh.
- Kiểm tra đủ các khía cạnh yếu, không chỉ ba khía cạnh trong góp ý: kết quả test cũ cho `kinh_nghiem` 71,85, `noi_chung` 72,00, `chuong_trinh_hoc` 72,17, `kien_thuc` 74,95.

**Phân tích trên test sau khi khóa, cho bài báo:**

- F1 từng khía cạnh theo số mẫu train. Đã có ngoại lệ với giải thích theo độ phổ biến: `thiet_bi_day_hoc` cao dù ít mẫu.
- F1 câu một khía cạnh và câu nhiều khía cạnh; 1.485 câu train vừa có nhiều khía cạnh vừa có cảm xúc trái nhau là nhóm đáng xem riêng.
- Câu có từ đối lập ("nhưng", "tuy nhiên") và lỗi cảm xúc `neutral` (14,8% số cặp).
- Hình attention cho một số câu đại diện, nói rõ đây là minh họa, không phải giải thích nhân quả.
- Khoảng 5–8 ví dụ E2 sai và E4 đúng hoặc ngược lại; chọn ví dụ đại diện, không chỉ ví dụ có lợi.

## Thứ tự chạy khi hạn chế GPU

1. **Tái hiện (G2):** seed 42 của E3 (β = 1,5) và E1, trên dataset Kaggle cũ 16.124 mẫu để so đúng với notebook. Đo thời gian và pilot BamiBERT.
2. Khóa mô tả khía cạnh, chạy chẩn đoán trên validation.
3. E0, sau đó E2, E3, E5, E4 trên dữ liệu đã duyệt (seed 42–44). **Cổng G3b.**
4. E6a, E1 (seed 42–44).
5. Thêm seed 45, 46 đồng đều cho các cặp chính nếu còn quota.
6. E6b theo thứ tự ViSoBERT, PhoBERT-base-v2, mBERT; rồi E7, E8.
7. P2 chỉ khi đã xong tất cả ở trên và còn trước G4.

Lưu kết quả sau mỗi run để một phiên Kaggle bị ngắt không làm mất các run trước.

## Nhật ký thay đổi

| Ngày | Thay đổi | Lý do |
|---|---|---|
| 01/11/2026 | Đổi mô hình đề xuất sang query attention cho cả hai nhánh; thêm ngưỡng theo khía cạnh, so sánh encoder và chẩn đoán lỗi; đổi tiêu chí chọn checkpoint sang F1 cặp | Theo góp ý ngoài đã được kiểm tra, xem `06-external-review-decisions.md` |
| 01/11/2026 | Lõi 21 lượt, E5 lên P0, BamiBERT vào lõi và bỏ encoder lớn, β = 1,0, 3 seed đồng đều, khóa ngưỡng trên validation, mức chênh 0,5 điểm F1, sửa bootstrap, thay "tương đương" bằng "chưa đủ bằng chứng", đổi tên annotation audit, bổ sung 51 câu | Theo nhận xét của Codex đã được kiểm tra và người dùng đồng ý |
| 01/11/2026 | Hủy annotation audit; RQ1 chỉ còn thống kê rà soát; `--compare` mặc định chấm validation; E1 ghi là phỏng theo [2] | Nhóm quyết định; sửa lỗi code; đối chiếu với bài [2] |
