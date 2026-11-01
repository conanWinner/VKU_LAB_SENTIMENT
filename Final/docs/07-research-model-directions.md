# Nghiên cứu model mới và nhận xét cho bài FJCAI 2027

Ngày đối chiếu: **01/11/2026**. Skill: `research`, nhánh nghiên cứu tổng hợp (`fallback`).

> **Cập nhật 01/11/2026:** đây là tài liệu tham khảo; protocol chính thức là `04-experiment-protocol.md`, đã nhận các đề xuất phù hợp. Các số 16.124 mẫu trong file này là ảnh chụp trước khi bổ sung 51 câu; `data/Final_Data` hiện có 16.175 mẫu và 23.910 cặp. Các hướng ghi "ngoài bài" chỉ để tham khảo sau hạn nộp, kể cả khi đoạn đó vẫn viết dạng hướng dẫn chạy.

Tài liệu này bổ sung cho `01`–`06`, không tự thay thế protocol đã chốt. Các đề xuất dưới đây chưa được huấn luyện hoặc đo hiệu năng trong repo. “Mới” là thời điểm/cách tiếp cận có nguồn kiểm chứng, không đồng nghĩa với đóng góp khoa học mới của nhóm.

## 1. Khuyến nghị chính

**Sau phản hồi của Claude: tập trung vào aspect-query attention, giữ 21 lượt GPU lõi; chỉ bổ sung BamiBERT làm encoder mới.** Không mở nhánh LLM hoặc nghiên cứu model nhỏ trước hạn nộp. Các thay đổi dưới đây là khuyến nghị cập nhật; chưa sửa `04-experiment-protocol.md` hoặc code huấn luyện.

Phạm vi khuyến nghị thống nhất:

| Ưu tiên | Nội dung | Điều kiện |
|---|---|---|
| Lõi | PhoBERT E2/E3/E4/E5, BamiBERT E2/E4, BiLSTM-CNN; mỗi cấu hình 3 seed; TF-IDF trên CPU | Đo thời gian thực tế trước khi cam kết quota |
| P1 | ViSoBERT, PhoBERT-base-v2, mBERT | Chỉ sau lõi; ưu tiên thêm seed đồng đều cho các cặp so sánh chính nếu đủ quota |
| P2 / ngoài bài | Head bốn trạng thái ở P2; GoiBERT, SEA-LION, HuTieuBERT, Qwen/QLoRA, retrieval, distillation, augmentation và chuyển người rà soát ngoài bài này | Giữ nội dung khảo sát để tham khảo sau hạn nộp |

Giữ một cấu hình encoder cố định, đề nghị β = 1,0; báo cáo ngưỡng 0,5 để tham khảo bên cạnh phân tích ngưỡng đã định trước, không tự thay quy trình chọn ngưỡng của protocol. So sánh này kiểm tra phương pháp dưới cùng cấu hình, **không phải bảng xếp hạng encoder đã được tinh chỉnh tối ưu**. BiLSTM-CNN giữ cấu hình phù hợp baseline của nó, không áp LR của Transformer một cách máy móc.

Cần sửa trước khi chạy: E5 phải trở thành ablation bắt buộc nếu bài tuyên bố lợi ích của khởi tạo query bằng tên khía cạnh; BamiBERT không thuộc nhóm encoder lớn; chưa chứng minh khác biệt không có nghĩa là chứng minh tương đương.

## 2. Hội thảo và khoảng trống hiện tại

### 2.1. Yêu cầu đã kiểm tra lại

Trang gửi bài xác nhận 6–8 trang, đúng template IEEE do hội thảo cung cấp, nghiên cứu hoàn chỉnh có mô hình, thực nghiệm và thảo luận; hạn nộp **15/11/2026**. Trang này ghi tiếng Việt hoặc tiếng Anh, trong khi Thông báo số 2 yêu cầu hoàn toàn bằng tiếng Anh. Giữ quyết định viết tiếng Anh trong `01-fjcai2027-requirements.md`. “Phản biện kín” chưa đủ để suy ra bắt buộc double-blind; tên tác giả/header template vẫn là điểm cần BTC xác nhận. [S01][S02]

Hội thảo không yêu cầu dùng LLM lớn hoặc model phát hành mới nhất. Độ phù hợp chủ đề và thực nghiệm có thể kiểm chứng quan trọng hơn việc thêm nhiều thành phần. Chưa thể kết luận bài sẽ được nhận chỉ từ việc đáp ứng định dạng.

### 2.2. Bằng chứng từ repo

Đã đọc cả sáu file `docs`, mã trong `phobert_multitask.ipynb`, cấu hình `Server/server.py` và đếm lại `data/Final_Data` tại commit `e89f1ce`.

| Split | Mẫu theo `(file, id)` | Văn bản phân biệt | Dòng cặp aspect–sentiment |
|---|---:|---:|---:|
| Train | 11.389 | 11.388 | 16.883 |
| Validation | 1.575 | 1.575 | 2.312 |
| Test | 3.160 | 3.160 | 4.664 |
| Tổng | 16.124 | 16.123 | 23.859 |

Không thấy giao nhau của văn bản **khớp chính xác** giữa ba split. Đây chưa phải kiểm tra trùng gần giống hoặc kiểm tra contamination với corpus pre-training.

**Điểm cần sửa trong cách mô tả dữ liệu:** dataset card UIT-VSFC ghi 11.426/1.583/3.166, tổng 16.175 mẫu. Bộ hiện tại không có cùng số lượng; `05-paper-outline.md` chưa nên viết “relabel the full UIT-VSFC corpus” khi chưa giải thích quá trình lọc và ánh xạ. Cần manifest `original_split/original_id/new_id/keep_reason`; phân biệt 16.124 mẫu với 16.123 chuỗi khác nhau. Không tự xóa/gộp mẫu để ép khớp số liệu. [S03]

Đối chiếu bổ sung với `data/{train,validation,test}.csv` bằng số lần xuất hiện của từng văn bản: thiếu lần lượt **37/8/6**, tổng **51**, không có văn bản thêm hoặc sửa khác gốc. Đây là bằng chứng mẫu vắng mặt, chưa phải bằng chứng lý do loại. Không suy tiêu chí “câu ngắn/không nội dung” từ ví dụ; xem kiểm tra có thể chạy lại trong `09-review-scope-decisions.md`. License UIT-VSFC ghi `unknown`; cần làm rõ quyền với nhóm tác giả gốc trước khi công khai văn bản kèm bộ nhãn.

Notebook hiện có cross-attention cho sentiment, nhưng aspect detection vẫn mean pooling. `BETA` bên ngoài là 1,0, trong `forward` lại là 1,5. Các điểm về checkpoint cũ, human review và kết quả một seed được nêu trong audit `02`; lượt nghiên cứu này không huấn luyện lại hoặc kiểm định lại checkpoint. Demo hiện chưa phải bằng chứng mô hình đề xuất đã chạy đúng end-to-end.

## 3. Model mới: chọn theo vai trò

### 3.1. Bảng lựa chọn

`M` là triệu tham số; `B` là tỷ tham số. Kích thước không phải VRAM khi huấn luyện.

| Model / thời điểm nguồn | Vai trò | Điểm đáng thử | Giới hạn và ưu tiên |
|---|---|---|---|
| **BamiBERT**, 07/2026 | Encoder tiếng Việt, khoảng 102M | Raw text, không cần tách từ ngoài; rất gần cách dùng PhoBERT hiện tại | Ưu tiên cao; sửa phân loại “encoder lớn” trong E6. [S04][S05][S06] |
| **GoiBERT-S / GoiBERT-M**, 10/2026 | Encoder Việt nhỏ, 26,4M / 49,4M | Hướng giảm RAM/độ trễ; tokenizer cùng họ BamiBERT | Ngoài bài; checkpoint là backbone, benchmark còn dùng task distillation. [S06][S07] |
| **SEA-LION-ModernBERT-300M**, model card 03/2026 | Encoder hiện đại đa ngôn ngữ có tiếng Việt | Kiểm tra encoder khu vực so với encoder đơn ngữ | Ngoài bài; không lấy điểm embedding làm điểm ABSA. [S08] |
| SEA-LION-ModernBERT-600M | Đối chứng encoder lớn | Tham khảo hướng mở rộng | Ngoài bài; card trộn mô tả encoder/embedding. [S09] |
| **HuTieuBERT**, ACL 2026 | Hướng đưa cấu trúc hình thái vào attention | BMES boundary embeddings, learnable gate và attention bias | Ngoài bài; model card checkpoint truy xuất thất bại, chưa xác nhận cách nạp/license. [S10] |
| **Qwen3.5-4B** | Model sinh nhãn có cấu trúc | Đối chứng prompt và QLoRA, có chế độ tắt thinking | Ngoài bài vì thời gian/phạm vi; không thay trực tiếp vào `AutoModel` của E4. [S11] |
| **Qwen3.8-27B** | Đối chứng LLM lớn mới hơn | Trọng số và model card công khai, điều khiển thinking | Ngoài bài; mới hơn không chứng minh tốt hơn trên ABSA Việt. [S12] |

PhoBERT hiện tại phải giữ để nối với kết quả cũ. PhoBERT-v2/ViSoBERT vẫn hữu ích làm đối chứng, nhưng không nên trình bày là các model vừa ra năm 2026. Không cần thử tất cả dòng trong bảng.

### 3.2. BamiBERT: thay đổi nhỏ, giá trị thực nghiệm rõ

Bài gốc ngày 02/07/2026 mô tả encoder base 12 lớp, nhận raw text, context tối đa 2.048 token. Kết quả UIT-VSFC sentiment F1 83,41 so với PhoBERT 83,27 chỉ chênh nhỏ; ở Hotel sentiment classification BamiBERT còn kém PhoBERT/ViSoBERT. Vì vậy không thể suy ra model này chắc chắn tăng pair F1 cho bộ 11 khía cạnh. [S04]

Model ID: `Qualcomm-AI-Research/BamiBERT`. Model card hướng dẫn `AutoTokenizer`/`AutoModel`, ghi `transformers<=5.5.0`; license gồm BSD-3-Clause-Clear và Qualcomm Responsible AI License, intended use nghiên cứu/giáo dục. Cần pin revision và môi trường, không mặc định nâng Transformers mới nhất. [S05]

Cách chạy đề nghị: cùng dữ liệu, cùng metric; **E2-BamiBERT và E4-BamiBERT**. Chưa có E5-BamiBERT thì không kết luận riêng về semantic initialization trên cả hai backbone. Đưa raw Unicode đã chuẩn hóa vào tokenizer, không thêm dấu gạch dưới từ `underthesea`. Đo tỷ lệ câu bị cắt theo từng tokenizer và split ở độ dài đã khóa; chỉ dùng train/validation để quyết định chiều dài, không điều chỉnh từ test.

### 3.3. GoiBERT: tham khảo sau bài này

GoiBERT công bố 21/10, bản v2 ngày 22/10/2026. Paper dùng hai giai đoạn: distillation biểu diễn từ BamiBERT, rồi distillation theo nhiệm vụ từ CafeBERT bằng dự đoán out-of-fold — teacher dự đoán trên phần train mà chính nó chưa học. Đây là công trình đã có, không được gọi hai-stage distillation là sáng kiến mới của nhóm. [S06]

Bản `TwoheartsTechnologies/GoiBERT-S` tương ứng GoiBERT-base, hidden size 384; M tương ứng GoiBERT-large trong paper. S khuyến nghị tối đa 256 token vì vị trí xa hơn chưa được huấn luyện. Card ghi rõ checkpoint phát hành **chỉ là encoder**, các điểm benchmark còn dùng task distillation và layer-wise learning-rate decay. Community license có điều kiện thương mại và nghĩa vụ upstream; không đồng nhất với Apache/MIT. [S07]

Trong paper, tốc độ batch trên L40S tăng không đồng nghĩa `/predict` của Flask nhanh cùng tỷ lệ: phép đo chưa gồm tokenizer và task head. Thử fine-tune GoiBERT trực tiếp trước; chỉ distill bổ sung khi cần. Đặc biệt kiểm tra giao giữa split hiện tại và benchmark training text mà GoiBERT từng dùng, vì giữ sạch train/test nội bộ chưa đủ chứng minh sạch với dữ liệu pre-training của model. [S06][S07]

### 3.4. SEA-LION-ModernBERT: một hướng encoder thực sự khác

AI Singapore phát hành các encoder 300M/600M, có tiếng Việt trong 13 ngôn ngữ, tokenizer Gemma 3, context 8k và MIT. Dùng bản **encoder**, không nhầm `SEA-LION-ModernBERT-Embedding-*` đã tối ưu cho retrieval với backbone dự định fine-tune. [S08][S09]

Hướng nghiên cứu: liệu query theo khía cạnh có còn hữu ích với encoder khu vực hiện đại? Chạy E2/E4 trên 300M theo ngân sách tuning tương đương. Model đa ngôn ngữ không buộc bài phải trở thành nghiên cứu đa ngôn ngữ. Với câu phản hồi ngắn, chưa có lý do sử dụng context 8k; chỉ tăng chiều dài khi đo được truncation đáng kể.

### 3.5. HuTieuBERT: hướng nghiên cứu hình thái, chưa nhận là triển khai sẵn

Bài ACL 2026 đưa ranh giới hình vị vào token embeddings và các lớp attention đầu, có lợi ích lớn hơn ở tác vụ cú pháp. Bằng chứng đó không trực tiếp chứng minh lợi ích trên ACSA của nhóm. [S10]

Nếu mở rộng: so E4 với E4 có boundary feature, giữ nguyên head và số bước tối ưu; phân tích lỗi từ ghép, viết tắt, phương ngữ. Cần tái hiện đúng phương pháp gốc trước; đưa boundary vào query/key là một biến thể do nhóm đề xuất, chưa phải nội dung đã được nguồn xác nhận. Không chọn hướng này trước hạn nếu chưa lấy được preprocessing/code/checkpoint đầy đủ.

### 3.6. Qwen: ngoài phạm vi bài này

**Loại Qwen/QLoRA và retrieval few-shot khỏi bài này vì thời gian và phạm vi.** Góp ý trước đã hiểu sai lý do loại: vấn đề vòng lặp cùng nguồn nhãn áp dụng riêng cho GPT-OSS, không phải lý do loại toàn bộ LLM. Việc khảo sát model mới không đồng nghĩa mở lại nhánh đã loại. Nghiên cứu 2025 về zero/few-shot ASQP/TASD chỉ giữ làm tham khảo sau bài này. [S13]

Qwen3.5-4B là causal language model có vision encoder, 4B phần ngôn ngữ, Apache-2.0; thinking bật mặc định và có thể tắt. Qwen3.8-27B mới hơn, 27B phần ngôn ngữ và cũng có vision encoder. Không cần sử dụng ảnh/video cho dữ liệu hiện tại. Model card chưa cung cấp bằng chứng thắng trên bộ ABSA 11 khía cạnh này. [S11][S12]

Ước lượng số học **chỉ riêng trọng số ngôn ngữ**: 4B ở 4-bit khoảng 2 GB; 27B khoảng 13,5 GB. Đây không phải VRAM cần thiết: còn vision module nếu được nạp, trạng thái hybrid attention, activations, KV cache và adapter. Vì vậy không cam kết 27B huấn luyện được trên GPU 16 GB; cần đo 100–200 bước với đúng stack.

## 4. Hướng triển khai và phép thử đủ để đánh giá

Các mục dưới giữ lại thiết kế để tham khảo, **không phải danh sách triển khai trước hạn**: head bốn trạng thái H1 là E10/P2; H2–H4 ngoài bài; H5–H6 sau khi nộp. Mô tả khía cạnh phục vụ thí nghiệm lõi phải viết từ annotation guideline và khóa phiên bản trước khi xem lỗi; không lấy ví dụ test làm query.

### H1. Query theo mô tả khía cạnh + head bốn trạng thái

**Giả thuyết:** biểu diễn theo khía cạnh giảm nhầm nội dung/cảm xúc trong câu nhiều khía cạnh; đầu ra bốn trạng thái giúp liên kết có mặt và cảm xúc trong một phân phối.

```text
Văn bản -> encoder -> H (L x d)
Mô tả 11 khía cạnh -> query Q (11 x d)
Q, H -> attention -> V (11 x d)
V -> 11 phân phối: absent / negative / neutral / positive
```

Đây là sơ đồ đề xuất, không phải mô hình đã triển khai. Head bốn trạng thái chính là ý tưởng E10 đang để P2; attention theo khía cạnh và BERT-pair cũng có tiền lệ, nên không tuyên bố mới chỉ vì kết hợp chúng. [S17][S18]

Một cách xác định đầu ra, cho khía cạnh `a`:

```text
q_a = softmax(logits_a) trên {absent, negative, neutral, positive}
present_a = 1 - q_a[absent]
sentiment_a = argmax q_a[negative, neutral, positive]
phát cặp khi present_a >= threshold_a
```

Đối chứng phải tách hai thay đổi: (1) cùng E4, chỉ đổi head; (2) giữ head, chỉ đổi query khởi tạo. Nếu dùng argmax bốn lớp thay vì threshold, báo cáo như một decoder khác. Lớp `absent` nhiều hơn ba lớp còn lại, nên báo cáo mất cân bằng và precision/recall. Không giả định head bốn trạng thái khắc phục được mọi lỗi bỏ sót.

**Tách khỏi lõi:** E2/E3/E4/E5 trên PhoBERT và E2/E4 trên BamiBERT kiểm tra query attention với head hiện tại. H1 cần thêm E10 để kiểm tra riêng head bốn trạng thái, nên giữ P2; không tính H1 đã được kiểm tra bằng 21 lượt lõi.

### H2. QLoRA và sinh nhãn theo schema

**Giả thuyết:** model sinh được fine-tune trên đúng nhãn sẽ tốt hơn prompting thuần ở câu nhiều khía cạnh/cảm xúc đối lập. QLoRA là fine-tune adapter trên model lượng tử hóa 4-bit, đã có hỗ trợ trong PEFT; tính tương thích với riêng Qwen hybrid vẫn phải kiểm tra thực tế. [S14]

Thiết kế triển khai:

1. Serialize nhãn theo thứ tự cố định của 11 aspect; chỉ tối ưu loss phần câu trả lời, che prompt. `absent` khác `neutral`.
2. Đầu ra là object có đúng 11 key và mỗi value thuộc bốn trạng thái; chỉ chuyển thành set cặp khi tính F1. Cách này tránh thứ tự JSON array và cặp trùng làm sai metric.
3. QLoRA thử ban đầu: NF4 + double quantization, LoRA rank 8 hoặc 16, sequence budget tính cả prompt/mô tả/output; batch và gradient accumulation tùy GPU. Đây là cấu hình khởi đầu đề xuất, chưa đo.
4. Không sao chép danh sách `q_proj/v_proj` của model khác một cách máy móc: xem module của Qwen Gated DeltaNet, freeze vision nếu được nạp, kiểm tra đúng các tham số trainable. BF16 chỉ dùng nếu phần cứng hỗ trợ; có phương án FP16 tương thích.
5. Tắt thinking cho baseline chính; chỉ thêm thinking-on như ablation có đo thời gian. Khóa model revision, chat template, decoding, độ dài và seed.
6. Nếu engine hỗ trợ, dùng JSON Schema/grammar constrained decoding. vLLM hỗ trợ schema, nhưng schema đúng không đảm bảo nhãn đúng. Không loại các mẫu output lỗi/truncated khỏi denominator; báo invalid rate và quy tắc fallback cố định. [S15]

So sánh ba điều kiện: zero-shot; few-shot với ví dụ chỉ từ train; QLoRA supervised trên cùng train. Prompt/hyperparameter chọn trên validation, sau đó đánh giá **toàn bộ test** đã khóa. Nếu chỉ đo một mẫu nhỏ thì chỉ gọi pilot, không đặt chung bảng chính với F1 toàn test của encoder.

Không triển khai H2 trong bài này, kể cả khi còn quota sau lõi; ưu tiên hoàn thiện bằng chứng cho câu hỏi chính.

### H3. Distillation theo cặp khía cạnh–cảm xúc sang GoiBERT

**Giả thuyết:** học phân phối của teacher trên cùng 11 khía cạnh giữ được chất lượng khi giảm độ trễ. Kế thừa distillation và out-of-fold của GoiBERT; đóng góp phải nằm ở hiệu quả trên bộ nhãn mới và phân tích trade-off, không phải phát minh distillation. [S06]

Dùng encoder tốt nhất trên validation làm teacher. Student là GoiBERT-S/M đã pre-train, không tự distill lại hàng trăm triệu token. Nếu dùng head H1, teacher và student đều xuất `[batch, 11, 4]`; thử:

```text
loss = (1 - alpha) * CE(gold, student)
     + alpha * T^2 * KL(softmax(teacher_logits / T), softmax(student_logits / T))
```

Đây là công thức cấu hình đề xuất; chọn `alpha`, `T` trên validation. Luôn có student supervised không KD làm đối chứng. Teacher chỉ học train; không dùng test/nhãn đối chứng để distill. OOF tránh teacher học đúng mẫu nó đang sinh soft label, nhưng tốn thêm K lượt teacher; nếu dùng một teacher in-sample thì ghi rõ, không gọi là OOF và không giả là protocol của paper GoiBERT.

Cần đo pair F1, macro theo aspect, minority recall, dung lượng, peak RAM/VRAM, p50/p95 latency **gồm preprocessing** và throughput trên cùng máy. Lợi ích deployment có thể tồn tại dù F1 không vượt teacher.

### H4. Lấy ví dụ tương tự từ train rồi đưa cho LLM

Đây là hướng retrieval-augmented few-shot (truy hồi ví dụ để tạo prompt), không cần vector database cho 11 nghìn câu. Bắt đầu bằng TF-IDF + cosine đã đủ một đối chứng; chỉ thử embedding model sau khi có baseline.

Index chỉ chứa train. Cho cùng số ví dụ/cùng ngân sách token, so random few-shot với retrieval few-shot. Deduplicate ví dụ gần giống; không index validation/test. Giữ taxonomy và parser như H2. Nếu ích lợi chỉ nằm ở câu rất giống train, báo cáo riêng thay vì tuyên bố suy luận tốt hơn.

Đây là đề xuất kỹ thuật, chưa chứng minh mới về học thuật. Có thể ghép H4 vào pilot H2 rẻ hơn QLoRA, nhưng không làm đồng thời mọi nhánh.

### H5. LLM hỗ trợ tăng dữ liệu có kiểm soát

Nghiên cứu 2025 về ACSA đã dùng augmentation giữ ngữ nghĩa và confidence-weighted fine-tuning; chỉ “thêm câu bằng LLM” không đủ tạo tính mới. [S19]

Nếu chẩn đoán cho thấy lỗi ở khía cạnh ít mẫu, chỉ tạo biến thể từ train, giữ hoặc đổi nhãn theo quy tắc được người kiểm tra. So sánh: dữ liệu gốc; oversampling cùng số bước; augmentation cùng số bước. Giới hạn số câu bổ sung, lưu nguồn và tránh paraphrase test. Không tự cho rằng câu “giữ ngữ nghĩa” vẫn giữ toàn bộ aspect và polarity, đặc biệt khi sửa từ phủ định/“nhưng”. Ưu tiên sau H1/H2.

### H6. Chọn lọc dự đoán và chuyển người rà soát

Ứng dụng có thể đưa câu ít chắc chắn sang người xem lại. Chọn quy tắc trên validation, không dùng confidence LLM tự khai như xác suất đã được hiệu chỉnh. Với output bốn trạng thái, xem calibration từng aspect và risk–coverage trên câu; báo tỷ lệ bỏ sót cả câu không dự đoán aspect nào.

LLM fallback chỉ thêm sau khi đã so với phương án rẻ hơn là encoder abstain + người rà soát. Nếu đo cascade, báo cả hệ thống và từng tầng, tỷ lệ chuyển tầng, latency, chi phí; không chỉ chấm các câu dễ. Đây là hướng triển khai dài hơn, không bắt buộc cho bài FJCAI.

## 5. Nhận xét để đối chiếu protocol, chưa áp dụng vào file 04

| Điểm ở docs hiện tại | Nhận xét và thay đổi đề xuất |
|---|---|
| E4 đổi cả nhánh detection và semantic initialization; E5 chỉ P1 | Đưa E5 lên cùng mức ưu tiên E4 nếu tuyên bố cả hai đóng góp. E4–E3 không tách được hai hiệu ứng. |
| BamiBERT nằm trong “encoder lớn” | Chuyển thành encoder Việt base; bỏ nhánh encoder lớn khỏi bài. |
| Cùng hyperparameter cho mọi encoder | Nhận giới hạn để tiết kiệm lượt chạy: giữ cấu hình cố định và chỉ đổi preprocessing bắt buộc. Không trình bày thành bảng xếp hạng encoder tối ưu; tuning riêng là công việc sau. |
| Chọn beta bằng E4 rồi áp cho mọi baseline | Đề nghị cố định β = 1,0, không chọn theo lợi thế của E4. File 04 vẫn ghi lưới beta; cần đồng bộ khi nhóm chốt protocol, chưa tự sửa ở lượt này. |
| Chọn ngưỡng riêng nếu test xác nhận lợi ích | Chọn decoder trên validation; test chỉ đánh giá quyết định đã khóa. Có thể báo cả hai decoder như phân tích định trước, không chọn lại từ test. |
| CI chứa 0 thì viết “tương đương” | Viết “chưa đủ bằng chứng về khác biệt”. Muốn kết luận tương đương phải định trước biên tương đương và kiểm định phù hợp. |
| Chênh phải lớn hơn SD để gọi cải thiện | Đây không phải một kiểm định chuẩn. Báo effect size theo điểm F1, CI ghép cặp và ý nghĩa thực tiễn; không coi SD như ngưỡng p-value. |
| Bootstrap qua nhiều seed | Resample cùng sentence indices cho hai model, tái tính micro-F1 từ counts; không coi các seed × câu là mẫu độc lập. Báo rõ CI có điều kiện trên tập seed đã chạy; với 3 seed, bất định giữa seed vẫn lớn. |
| Chỉ tăng seed cho model thắng | Cho hai mô hình của so sánh chính cùng tập seed. Không so 5 seed model đề xuất với 1 seed baseline rồi kết luận ổn định hơn. |
| RDRSegmenter để P2 nhưng model chính là PhoBERT | Tái hiện underthesea trước, sau đó khóa preprocessing dự định dùng; segmenter là thành phần của model, cần ghi phiên bản và thời gian. |
| “full corpus” và số mẫu | Giải thích chênh với UIT-VSFC gốc và số nhóm `(file,id)`; không tự thay split đã dùng. |
| Lý do loại LLM | H2/H4 ngoài bài vì thời gian/phạm vi; riêng GPT-OSS có thêm vấn đề cùng nguồn nhãn. |

Chất lượng nhãn vẫn là điều kiện cần: gán nhãn độc lập, không xem LLM label; báo từng aspect và bất đồng. Một mẫu phân tầng lấy dư nhãn hiếm không đại diện tự nhiên cho tỷ lệ lỗi toàn bộ corpus: công bố cách lấy mẫu hoặc dùng trọng số nếu ước lượng toàn tập. Nếu không sửa test gốc, gọi đó là annotation audit; đừng gọi nhãn cũ thành gold độc lập. Nghiên cứu ABSA 2026 đã so nguồn nhãn bằng IAA và tác động lên mô hình, hỗ trợ hướng đánh giá này. [S16]

NEU-ESC là nghiên cứu giáo dục tiếng Việt liên quan gần, nhưng nhiệm vụ sentiment/topic khác 11-aspect pair prediction; đưa vào Related Work, không trộn accuracy của họ với F1 của nhóm hay gộp dataset khác taxonomy vào bài hiện tại. [S20]

## 6. Kế hoạch tối thiểu trước hạn

Ngân sách GPU chưa được xác nhận. Dưới đây là đề xuất phạm vi, không phải cam kết thời gian chạy.

| Cụm | Run huấn luyện chính nếu 3 seed | Mục đích |
|---|---:|---|
| PhoBERT E2/E3/E4/E5 | 12 | Tách attention cho detection khỏi semantic initialization |
| BamiBERT E2/E4 | 6 | Kiểm tra lợi ích E4 trên backbone khác |
| BiLSTM-CNN E1 | 3 | Đối chứng với nền tảng hiện tại |
| TF-IDF E0 | CPU, recipe cố định | Mốc rẻ, đơn giản |
| **Tổng lõi** | **21 run GPU** | Chưa tính tuning/pilot/run lỗi |

Chạy seed 42/43/44 cho từng cấu hình lõi. Nếu đo xong còn quota, thêm seed đồng đều cho các cặp so sánh chính; không chỉ tăng seed cho model thắng. P1 chỉ gồm ViSoBERT, PhoBERT-base-v2 và mBERT. Không mở GoiBERT/QLoRA. Không có E5-BamiBERT thì giới hạn tuyên bố semantic initialization ở PhoBERT. Pilot chỉ dùng lại làm run chính khi đúng toàn bộ cấu hình đã khóa.

Ước lượng sau khi đo: `tổng giờ GPU = tổng(run_count × thời gian một run đo được) + pilot + dự phòng`. **Chưa xác nhận loại/số GPU, VRAM hoặc quota Kaggle trong tài khoản**; không coi “hai GPU 16 GB” là tài nguyên đã có. Cần ghi phần cứng từ runtime thực và quota từ tài khoản; nhiều GPU không tự cộng VRAM thành một vùng nhớ chung.

1. **01–04/11:** khóa dữ liệu/metric/mô tả khía cạnh, tái hiện baseline, bắt đầu annotation audit; pilot PhoBERT/BamiBERT/BiLSTM-CNN để đo ngân sách.
2. **05–06/11:** chạy lõi với cấu hình cố định; chọn decoder trên validation theo protocol đã chốt. Nếu chậm, bỏ P1 trước.
3. **07–09/11:** chạy các seed còn lại, khóa số liệu, đánh giá test các cấu hình đã định; bootstrap và phân tích nhóm lỗi.
4. **10–13/11:** viết, rút 6–8 trang, tác giả duyệt; mục tiêu nộp sớm theo kế hoạch cũ. Không bổ sung kiến trúc sau khi xem test.

Nếu chỉ đủ 12–15 run, giữ PhoBERT E2/E3/E4/E5 ba seed, lấy BamiBERT làm pilot ghi riêng; thu hẹp tuyên bố cross-backbone. Không hy sinh kiểm tra nhãn để thêm model.

## 7. Cách triển khai nhỏ nhất khi bắt đầu viết code

Không cần tạo framework model registry. Tách loader/metric dùng chung trước; encoder lấy `config.hidden_size`, head độc lập với tên backbone. Dùng `nn.MultiheadAttention` sẵn có, mask padding đúng, tên aspect và mapping polarity cố định.

Mỗi run cần lưu: SHA-256 từng file split và danh sách file có thứ tự, code commit, model ID + revision SHA bất biến, tokenizer revision/preprocessing version, seed, config thực dùng, checkpoint criterion, decoder/threshold, dự đoán theo `(file,id)`, thời gian và peak memory. Với tài nguyên không có revision như embedding của BiLSTM-CNN, lưu hash file. Không dùng tên nhánh `main` thay revision cụ thể.

Tỷ lệ câu bị cắt = số mẫu có độ dài token trước cắt (gồm special tokens) vượt `max_length` / tổng mẫu của split. Đếm mỗi `(file,id)` một lần, không đếm theo dòng cặp; ghi tokenizer, preprocessing, `max_length`, tử số/mẫu số và tỷ lệ cho train/validation/test. Khóa chiều dài bằng train/validation trước khi báo test; chưa đo các tỷ lệ này trong lượt nghiên cứu.

Các kiểm tra tối thiểu lúc triển khai: loader không gộp `id` xuyên file; sai polarity tính một FP và một FN của pair F1; thứ tự output không đổi điểm; `neutral` không bị coi là absent; schema sai/truncated không bị bỏ khỏi đánh giá. Đây là danh sách công việc đề nghị, chưa phải tests đã chạy.

Đối với demo, chỉ thay `Server/server.py` sau khi có artifact chứa đúng encoder/head/tokenizer/threshold. Đo latency batch 1 và batch nhiều, warm-up trước, tách cold start; không dùng throughput backbone từ paper làm tốc độ demo.

## 8. Đóng góp có thể bảo vệ trước phản biện

Mạch bài khả thi: **nhãn 11 khía cạnh có annotation audit; baseline nhất quán; chẩn đoán lỗi; query attention được ablation; kiểm tra trên BamiBERT dưới cấu hình cố định**. Dùng mô hình đã có theo một thiết kế thực nghiệm tốt vẫn có giá trị, nhưng không gọi là kiến trúc hoàn toàn mới hoặc bảng xếp hạng encoder.

| Kết quả thực tế | Cách diễn giải |
|---|---|
| E4 tốt hơn E5/E3 trên nhiều seed | Có bằng chứng cho thành phần tương ứng, giới hạn trong dataset và backbone đã đo |
| BamiBERT tốt hơn PhoBERT, E4 không hơn E2 | Khác biệt gắn với backbone/preprocessing dưới cấu hình đã dùng; chưa chứng minh lợi ích attention hoặc ưu thế encoder sau tối ưu |
| CI chưa rõ hoặc nhãn ít đồng thuận | Nghiên cứu benchmark/chẩn đoán với giới hạn rõ; không chọn ví dụ đẹp để bù |

Tiêu đề tham khảo: *Evaluating Aspect-Query Attention for Fine-Grained Vietnamese Student Feedback Analysis*. Chỉ chọn sau khi biết phần nào có kết quả.

## 9. Nguồn và mức bằng chứng

Đã nhận nội dung của 20 nguồn S01–S20 trong lượt nghiên cứu ban đầu, với độ sâu đọc ghi riêng từng nguồn; không có nghĩa đã đọc toàn văn cả 20. “Primary” là nguồn tác giả/nhà phát hành; preprint và model card không tự động là đánh giá độc lập hay peer-reviewed. H1–H6 là thiết kế tham khảo, trạng thái phạm vi theo mục 1 và 4. Số URL truy xuất và nguồn trích dẫn được tách rõ ở file 08.

- **S01 — Primary, đọc trang:** [FJCAI 2027 — Hướng dẫn nộp bài](https://fjcai.org.vn/submission/).
- **S02 — Primary, đọc trang:** [FJCAI — Thông báo số 2](https://fjcai.org.vn/notifications/thong-bao-so-2/index.html).
- **S03 — Primary dataset card, đọc README:** [UIT-VSFC](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/blob/main/README.md). License trong card là unknown; không suy quyền phân phối từ việc tải được.
- **S04 — Primary preprint, đọc toàn văn:** [Nguyen et al., BamiBERT (2026)](https://arxiv.org/html/2607.02259v1).
- **S05 — Primary model card, đọc README:** [Qualcomm-AI-Research/BamiBERT](https://huggingface.co/Qualcomm-AI-Research/BamiBERT/blob/main/README.md).
- **S06 — Primary preprint, đọc các phần phương pháp/kết quả/giới hạn:** [Tran et al., GoiBERT v2 (2026)](https://arxiv.org/html/2610.30067v2).
- **S07 — Primary model card, đọc README:** [TwoheartsTechnologies/GoiBERT-S](https://huggingface.co/TwoheartsTechnologies/GoiBERT-S/blob/main/README.md).
- **S08 — Primary model card, đọc README:** [SEA-LION-ModernBERT-300M](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-300M/blob/main/README.md).
- **S09 — Primary model card, đọc README:** [SEA-LION-ModernBERT-600M](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-600M/blob/main/README.md).
- **S10 — Primary publisher, đọc abstract/metadata:** [Dinh et al., When Morphology Hides in Plain Sight (ACL 2026)](https://aclanthology.org/2026.acl-long.472/). Chưa tái hiện code HuTieuBERT.
- **S11 — Primary model card, đọc overview/serving/thinking:** [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/README.md).
- **S12 — Primary model card, đọc overview:** [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/README.md).
- **S13 — Primary preprint, đọc abstract/metadata:** [Hellwig et al., Do we still need Human Annotators? (2025)](https://arxiv.org/abs/2502.13044).
- **S14 — Primary documentation, đọc hướng dẫn:** [Hugging Face PEFT — Quantization](https://huggingface.co/docs/peft/developer_guides/quantization).
- **S15 — Primary documentation, đọc schema/serving:** [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/).
- **S16 — Primary preprint, đọc abstract/metadata:** [Donhauser et al., Annotation Quality in ABSA (2026)](https://arxiv.org/abs/2605.03624).
- **S17 — Primary publisher, đọc abstract/metadata:** [Sun et al., Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence (NAACL 2019)](https://aclanthology.org/N19-1035/).
- **S18 — Primary paper, đọc abstract/metadata:** [Li et al., Multi-Instance Multi-Label Learning Networks for ACSA (2020)](https://arxiv.org/abs/2010.02656).
- **S19 — Primary preprint, đọc abstract/metadata:** [Chai et al., Semantic-preserved Augmentation with Confidence-weighted Fine-tuning for ACSA (2025)](https://arxiv.org/abs/2506.07148).
- **S20 — Primary preprint, đọc abstract/metadata:** [Mai et al., NEU-ESC (2025)](https://arxiv.org/abs/2506.23524).

Không dùng số liệu từ Google AI Overview, link chỉ xuất hiện trong search, hoặc leaderboard không cùng task để kết luận. Không gọi khảo sát này là systematic review hoặc danh sách đầy đủ mọi model tính đến 01/11/2026.

## 10. Audit và giới hạn

Chi tiết URL, lỗi và cách đếm trong [08-research-source-audit.md](08-research-source-audit.md). Kết quả phân loại từ script của skill là `fallback`, không có tín hiệu specialist; người dùng đã xác định phạm vi và yêu cầu file Markdown nên không hỏi lại output.

Đã kiểm tra docs, một phần mã liên quan và số lượng/split dữ liệu; chưa huấn luyện model, tải trọng số, đo GPU/latency, xác nhận dataset Kaggle hay đọc lại toàn bộ bản thảo PDF. Vì vậy đây là **báo cáo nghiên cứu và thiết kế thử nghiệm**, chưa phải kết quả cải thiện F1. Không có code mô hình hoặc dữ liệu hiện có bị thay đổi trong lượt này.
