# CLAUDE.md

## Dự án

ABSA (phân tích cảm xúc theo khía cạnh) trên phản hồi sinh viên tiếng Việt: UIT-VSFC được gán nhãn lại theo 11 khía cạnh và 3 cảm xúc, với LLM hỗ trợ và người rà soát. Mô hình chính là PhoBERT học đa nhiệm.

**Mục tiêu hiện tại:** nộp bài cho FJCAI 2027, hạn **15/11/2026**, mục tiêu nội bộ **13/11/2026**. Bài viết bằng tiếng Anh, 6–8 trang, theo template IEEE của hội thảo.

**Phạm vi đã chốt:** chỉ một miền (phản hồi sinh viên) và chỉ dữ liệu tiếng Việt. Không làm đa miền, đa ngôn ngữ, và không dùng M-ABSA, Coursera, Hotel. Tác giả: Đoàn Quang Thắng, Trương Công Lý, Phạm Nguyễn Thanh Nam.

Trước khi làm việc liên quan đến bài báo hoặc thực nghiệm, đọc:

- `docs/01-fjcai2027-requirements.md`: yêu cầu hội thảo và các điểm cần xác nhận
- `docs/02-current-state-audit.md`: hiện trạng code, dữ liệu, số liệu và lỗi của bản thảo
- `docs/03-implementation-plan.md`: lịch theo ngày, cổng quyết định, rủi ro
- `docs/04-experiment-protocol.md`: **protocol chính thức duy nhất**; nếu tài liệu khác mâu thuẫn thì theo file này
- `docs/05-paper-outline.md`: dàn ý bài và phân bổ trang
- `docs/06-external-review-decisions.md`: quyết định về góp ý bên ngoài, trạng thái kiểm tra trích dẫn
- `docs/07-research-model-directions.md`, `docs/08-research-source-audit.md`, `docs/09-review-scope-decisions.md`: nhận xét và khảo sát của Codex, **chỉ để tham khảo**; các hướng ghi "ngoài bài" không triển khai trước hạn nộp

## Cấu trúc

Git root là thư mục cha `VKU_Lab_Sentiment/`, đường dẫn trong git có tiền tố `Final/`. Dự án này nằm trong `Final/`.

| Đường dẫn | Nội dung |
|---|---|
| `data/Final_Data/` | **Bộ dữ liệu chuẩn cho bài báo:** `train1..6.csv`, `validation.csv`, `test.csv`; cột `id,text,sentiment,aspect` |
| `data/annotation/added_51_review.csv` | 51 câu bổ sung ngày 01/11/2026, nhãn do Claude đề xuất và đã được người duyệt (đổi 4 nhãn cảm xúc); ghi rõ nguồn nhãn này trong bài |
| `data/paper_subset/` | Tập khoảng 5k lấy từ nhãn của chính dự án, **trùng câu với test của `Final_Data`**; **không phải dữ liệu của bài [2]**, không dùng để so sánh trên cùng test |
| `data/{train,validation,test}.{csv,json}`, `data/split/`, `data/v1.csv` | Dữ liệu gốc và trung gian, không dùng cho số liệu trong bài |
| `phobert_multitask.ipynb` | `AspectAwareBertMultiTaskABSA`: query attention chỉ cho nhánh cảm xúc (E3 trong protocol); mô hình đề xuất E4 mở rộng từ đây, chạy trên Kaggle |
| `mbert_multitask.ipynb`, `billstm_cnn.ipynb`, `16k-final-bilstm-cnn.ipynb` | Các mô hình so sánh |
| `mberrrrrrr.ipynb` | Thử nghiệm mBERT gộp thêm dữ liệu Coursera; **ngoài phạm vi**, không dùng số liệu cho bài |
| `Server/` | Demo Flask; dùng kiến trúc `BertMultiTaskABSA`, không có cross-attention |
| `Server/model/` | Checkpoint và bản thảo `kkkkk.pdf`; **không có trên git**, không xóa hoặc ghi đè |
| `For Train 6.json` | Workflow n8n gọi LLM để gán nhãn |
| `src/absa/`, `tests/` | Code chính cho mọi run mới: dữ liệu, tiền xử lý, độ đo, mô hình E1–E5, huấn luyện, chấm điểm, E0, chẩn đoán lỗi (`audit.py` còn giữ nhưng không dùng). Cấu hình nằm trong preset và tham số CLI, không có thư mục `configs/` |
| `scripts/build_kaggle_kernel.py`, `kaggle/` | Đóng gói `src/absa` vào một notebook riêng tư theo job; `kaggle/` giữ notebook đã đẩy |
| `results/` | `runs/`: run theo protocol. `g2_reproduction_kaggle_v1/`, `pilot_kaggle_v1/`: dữ liệu cũ 16.124 mẫu, **không dùng số cho bài** |
| `data/annotation/audit/` | Phiếu annotation audit. **Đã hủy audit ngày 01/11**, nhóm quyết định; giữ file, không dùng trong bài |
| `paper/` | Bản nháp LaTeX từ template hội thảo: `main.tex` (cờ `\anontrue`; chỗ chưa có số ghi `\TODO`), `references.bib` (lấy từ DOI, arXiv), `tables/` (sinh bằng `scripts/paper_tables.py`). Biên dịch trên Overleaf |

## Lệnh

```bash
source .venv/bin/activate                # Python 3.13
jupyter notebook --no-browser --port=8889
PYTHONPATH=src python -m unittest discover -s tests   # chạy test
cd Server && python server.py            # http://localhost:5000, cần model/best_model.pt
PYTHONPATH=src python -m absa.summarize results/runs --data-dir data/Final_Data  # bảng và so sánh trên validation

# Đẩy một job lên Kaggle (tốn quota GPU)
python scripts/build_kaggle_kernel.py <job> --owner thngonquang [--dataset OWNER/SLUG --data-hash HASH]
kaggle kernels push -p kaggle/<job> --accelerator NvidiaTeslaT4
```

Môi trường tính toán: **không huấn luyện trên máy local**, máy yếu; mọi lượt GPU chạy trên Kaggle bằng CLI, tài khoản `thngonquang`, 2 GPU T4. Dataset cho bài là `thngonquang/vku-feedback-v2` (riêng tư, hash `c13e760b…`). Dataset `conanwinner1/vku-feedback` là bản cũ 16.124 mẫu, thuộc tài khoản khác và chỉ đọc được. Máy chưa cài LaTeX; biên dịch bài trên Overleaf.

## Quy tắc bắt buộc về thực nghiệm

- **Mọi số trong bài phải truy được về một file trong `results/`** kèm commit, config và seed. Không chép số từ output notebook cũ vào bản nộp.
- **Không dùng test để chọn bất cứ gì.** Checkpoint, threshold và siêu tham số đều chọn trên validation; checkpoint chọn theo F1 cặp.
- **Chỉ có một định nghĩa độ đo.** Các notebook hiện dùng tên `sent_f1` cho hai đại lượng khác nhau. F1 cặp (aspect, sentiment) là độ đo Task B; F1 cảm xúc trên khía cạnh đúng là một đại lượng khác và phải đặt tên riêng.
- **`id` chỉ duy nhất trong từng file train.** Toàn bộ train có 1.999 id cho khoảng 11,4 nghìn câu. Phải group theo `id` theo từng file rồi mới nối; gộp file trước khi group sẽ trộn các câu khác nhau.
- **Không viết kết luận mà thực nghiệm không kiểm chứng**, ví dụ "giảm lan truyền lỗi" khi không có baseline pipeline. Cải thiện nhỏ hơn dao động giữa các seed thì không gọi là cải thiện.
- **Không tự tạo trích dẫn.** Mỗi tài liệu tham khảo phải được kiểm tra có thật và đúng thông tin trước khi đưa vào `paper/`.
- **Không gọi kết quả rà soát nhãn là độ đồng thuận.** Nhóm không làm gán nhãn độc lập. Người rà soát đã thấy nhãn LLM và không còn file so sánh, nên không nêu con số "trên 99%" trong bài.
- **Không dùng GPT-OSS làm baseline**, vì nhãn ban đầu do chính nó sinh ra.
- **Không so số F1 với bài [2]**, vì quy tắc gán nhãn khác nhau. Mức khớp với nhãn UIT-VSFC không phải bằng chứng chất lượng, vì prompt đã đưa nhãn đó cho LLM làm gợi ý.
- **Hai mô hình được so sánh phải dùng cùng bộ seed.** Chỉ viết "cải thiện" khi khoảng tin cậy ghép cặp không chứa 0 và mức chênh từ 0,5 điểm F1; khoảng tin cậy chứa 0 thì viết "chưa đủ bằng chứng về khác biệt", không viết "tương đương".

## Quy ước

- Trao đổi và tài liệu nội bộ bằng tiếng Việt; bài báo, tên biến và code bằng tiếng Anh.
- Chỉ commit khi được yêu cầu. Nhánh làm việc là `dev`.
- Không commit file `*.pt`, dữ liệu mới có thể chưa được phép phân phối, hoặc thông tin xác thực.
- Giữ bản ẩn danh được: thông tin tác giả, lời cảm ơn và link dữ liệu phải tắt được bằng một cờ trong LaTeX.
