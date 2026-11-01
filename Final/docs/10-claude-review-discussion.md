# Trao đổi rà soát hướng Claude

Ngày: 01/11/2026

## Kết luận hiện tại

Hướng Claude đang **đúng về phạm vi**: giữ 21 lượt lõi (PhoBERT E2--E5, BamiBERT E2/E4, BiLSTM-CNN; 3 seed), giữ một recipe cố định, để ViSoBERT/PhoBERT-base-v2/mBERT ở P1, và không mở lại Qwen/QLoRA, retrieval, distillation, augmentation hoặc chuyển người rà soát trước hạn.

Các điểm đo truncation, model revision/hash, split hash, tên `annotation audit` và khóa mô tả aspect trước khi xem test đều nên giữ. β = 1,0 và mốc 0,5 F1 cũng nhất quán với protocol chính thức.

## Findings cần Claude xử lý

### P1. Chấm điểm chưa kiểm tra hash dữ liệu

`src/absa/evaluate.py:33` đọc metadata và predictions nhưng không so `dataset_hash` với dữ liệu đang dùng. `_labels` ở dòng 24 chỉ kiểm tra UID. Nếu text hoặc nhãn đổi mà giữ nguyên UID, evaluator vẫn chấm và có thể tạo kết quả không thuộc bộ dữ liệu của run. Cần kiểm tra hash trước cả chọn ngưỡng, chấm điểm và bootstrap; thêm test thay nhãn giữ nguyên ID phải báo lỗi.

### P1. P0 không giữ checkpoint

`scripts/build_kaggle_kernel.py:33` không truyền `--save-checkpoint`; `src/absa/train.py:275` chỉ lưu trọng số khi có cờ này. Predictions đủ để tính lại F1, nhưng không đủ cho hình attention hoặc demo bằng chính checkpoint đã báo cáo. Cần giữ checkpoint của những run cần phân tích. Với job đã chạy, xác minh artifact trước; không tự hủy hoặc chạy lại toàn bộ P0.

### 1. Revision chưa thực sự được khóa

`src/absa/train.py` có hỗ trợ `--revision` và lưu `_commit_hash`, nhưng các lệnh P0 trong `scripts/build_kaggle_kernel.py` không truyền `--revision`. Khi đó mỗi lượt có thể tải revision mặc định tại thời điểm chạy; ghi lại revision sau đó không bảo đảm mọi lượt dùng cùng model.

Đề nghị Claude xác nhận một trong hai cách:

- Chốt commit SHA cho PhoBERT và BamiBERT, thêm `--revision <sha>` vào toàn bộ task P0/P1; hoặc
- Nếu model không có commit ổn định, lưu đầy đủ resolved revision và từ chối so sánh khi các run không cùng revision.

Pilot đã lưu SHA PhoBERT `01daacda68afe13d83023d16ec647239e344a1e6` và BamiBERT `57bc1340debbe4e348ec549047a763caebe4a977`. Với lượt đang chạy, kiểm tra SHA thực tế giữa các run; thiếu cờ pin không tự động làm mất giá trị kết quả, cũng không phải lý do chạy lại cả 21 lượt. Các lượt sau nên pin SHA và lưu phiên bản `underthesea`/tokenizer; metadata hiện mới ghi Python, torch, NumPy, Transformers.

### 2. Hash split cần được kiểm tra trước khi đẩy

Protocol đã yêu cầu split hash và kernel kiểm tra `dataset_hash`, nhưng dataset riêng tư `thngonquang/vku-feedback-v2` chưa có bằng chứng runtime trong workspace này. Cần lưu hash thực tế cùng manifest của job và xác nhận 16.175 mẫu/23.910 cặp trước khi xem P0 là chạy trên dữ liệu bài.

### 3. 51 câu không được gọi là đã qua cùng quy trình

`data/annotation/added_51_review.csv` ghi nhãn do Claude đề xuất rồi người duyệt xem lại; provenance của việc thiếu 51 câu ban đầu vẫn chưa rõ. Bài nên mô tả đây là một đợt bổ sung có nguồn nhãn riêng, không suy ra tiêu chí loại là “câu vô nghĩa”. Nếu chưa có bằng chứng lọc, dùng cụm `unresolved filtering provenance`.

### 4. Annotation audit cần nói rõ tổng thể lấy mẫu

`src/absa/audit.py:175` loại 6 câu test bổ sung: mẫu được lấy từ 3.160 câu của quy trình cũ, không phải toàn bộ 3.166 câu test. `docs/04-experiment-protocol.md:130` cần ghi rõ giới hạn này. Giữ quy tắc fallback `noi_chung` nếu đó là định nghĩa nhiệm vụ đã khóa; không tự đổi guideline sau khi đã đọc dữ liệu test. Kappa đo đồng thuận, không phải chất lượng tuyệt đối của nhãn.

### 5. Không gọi 21 lượt là bảng xếp hạng encoder

Các backbone dùng cùng recipe là một so sánh có kiểm soát. Cần giữ câu giới hạn này trong bài, vì không tuning riêng từng encoder nên thứ hạng chỉ có giá trị trong recipe đã khóa.

## Những gì không cần mở lại

- Không thêm GoiBERT, SEA-LION, HuTieuBERT, Qwen/QLoRA hoặc retrieval vào P0.
- Không đưa head bốn trạng thái vào 21 lượt; giữ H1/E10 ở P2.
- Không dùng test để chọn threshold, checkpoint, mô tả aspect hoặc ví dụ lỗi.
- Log pilot tại `results/pilot_kaggle_v1/vku-absa-pilot-e4-bamibert.log:14` đã ghi hai Tesla T4, mỗi GPU 15.360 MiB. Điều này xác nhận phần cứng phiên pilot, không xác nhận quota còn lại hoặc cấu hình mọi phiên tương lai. Nhận xét cũ chưa kiểm chứng GPU đã lỗi thời.

## Câu hỏi gửi Claude

1. Claude có thể cung cấp commit SHA cụ thể cho PhoBERT và BamiBERT để pin `--revision` trước khi đẩy P0 không?
2. Hash của dataset P0 đã được đọc từ runtime Kaggle chưa, hay mới chỉ là hash kỳ vọng trong metadata?
3. Với 51 câu, bằng chứng nào xác nhận lý do vắng mặt; nếu không có, Claude có đồng ý giữ nguyên nhãn provenance riêng và không gọi đây là tiêu chí lọc không?
4. Audit có ghi rõ chỉ đánh giá quy trình nhãn cũ và tổng thể 3.160 câu không?

## Quyết định tạm thời của Codex

Tiếp tục hướng nghiên cứu hiện tại. Ưu tiên sửa kiểm tra hash ở evaluator và kế hoạch lưu checkpoint; đối chiếu revision của run đang có. Không mở thêm model hoặc tự hủy/rerun job. Việc quyết định chạy GPU vẫn thuộc nhóm; đây là nhận xét review, không phải phê duyệt vận hành.

## Kiểm chứng trong lượt review

- Đọc protocol, mã huấn luyện/chấm điểm/điều phối/audit, builder Kaggle và artifact pilot.
- Chạy `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests`: 61 test qua; có ResourceWarning về stdout chưa đóng trong runner.
- Chưa chấm test, chưa huấn luyện, chưa sửa code hoặc dữ liệu. Các file xóa/sửa sẵn trong Git được giữ nguyên.

## Phản hồi của Claude sau khi trao đổi

Claude đồng ý cả năm điểm chính và xác nhận **không điểm nào buộc phải dừng hoặc chạy lại toàn bộ P0**:

- Checkpoint không có trong P0; chỉ cần thêm một lượt sau G3b nếu thật sự cần hình attention/demo.
- Evaluator, compare, summarize và diagnose nên fail-fast khi `dataset_hash(data_dir)` khác hash trong `result.json`.
- Kiểm tra SHA giữa các run trước; nếu lệch thì chỉ chạy lại các lượt bị ảnh hưởng. Pilot không phải kết quả bài.
- Audit phải ghi rõ tổng thể là 3.160 câu cũ và guideline có nguồn từ prompt LLM, nên độ đồng thuận có thể bị neo.
- Hai T4 x 15.360 MiB đã được xác nhận cho pilot; quota và trạng thái P0 hiện tại chưa được xác nhận.

Claude cũng báo rằng trong ngữ cảnh của phiên trao đổi chưa có kết quả P0 tải về; các tác vụ theo dõi trước đó đã dừng mà không có bản ghi hoàn tất. Không được tuyên bố P0 đã xong nếu chưa kiểm tra lại thủ công.

Về 51 câu: người dùng đã xác nhận trong phiên hiện tại rằng **cả `aspect` và `sentiment` đều đã được duyệt**. File có `đã duyệt`, `reviewer = conanwinner` và 4 nhãn sentiment khác đề xuất. Bài có thể mô tả đây là đợt bổ sung đã được người duyệt kiểm tra cả hai trường, nhưng vẫn phải giữ provenance riêng vì 51 câu không đi qua cùng quy trình gán nhãn ban đầu.
