# Audit nguồn nghiên cứu model và FJCAI

Ngày: 01/11/2026  
Mục đích: lưu dấu các URL đã mở để kiểm tra `docs/07-research-model-directions.md`.

## Nguồn chính

| ID | Nguồn | Đã kiểm tra | Dùng cho |
|---|---|---|---|
| S01 | [FJCAI submission](https://fjcai.org.vn/submission/) | 200; đọc trang | 6–8 trang, IEEE, CMT, hạn nộp |
| S02 | [FJCAI notification 2](https://fjcai.org.vn/notifications/thong-bao-so-2/index.html) | 200; đọc trang | Quy định bài tiếng Anh, mốc thời gian |
| S03 | [UIT-VSFC dataset card](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/blob/main/README.md) | 200; đọc README | Quy mô gốc, split, license unknown |
| S04 | [BamiBERT paper](https://arxiv.org/html/2607.02259v1) | 200; đọc toàn văn | Kiến trúc, benchmark, tokenizer |
| S05 | [BamiBERT model card](https://huggingface.co/Qualcomm-AI-Research/BamiBERT/blob/main/README.md) | 200; đọc README | Model ID, cách nạp, license |
| S06 | [GoiBERT paper](https://arxiv.org/html/2610.30067v2) | 200; đọc phương pháp/kết quả/giới hạn | Distillation, ABSA, throughput, giới hạn |
| S07 | [GoiBERT-S model card](https://huggingface.co/TwoheartsTechnologies/GoiBERT-S/blob/main/README.md) | 200; đọc README | 26,4M, 256 token, backbone-only, license |
| S08 | [SEA-LION ModernBERT 300M](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-300M/blob/main/README.md) | 200; đọc README | Encoder đa ngôn ngữ, 8k context, MIT |
| S09 | [SEA-LION ModernBERT 600M](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-600M/blob/main/README.md) | 200; đọc README | Đối chứng lớn hơn, deployment |
| S10 | [HuTieuBERT, ACL 2026](https://aclanthology.org/2026.acl-long.472/) | 200; đọc metadata/abstract | Boundary fusion, morphology-aware attention |
| S11 | [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/README.md) | 200; đọc overview/serving | 4B, Apache-2.0, vLLM/SGLang, thinking |
| S12 | [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/README.md) | 200; đọc overview | 27B, vision-language, deployment |
| S13 | [LLM prompting for ASQP](https://arxiv.org/abs/2502.13044) | 200; đọc abstract | Few-shot LLM không phải gold/baseline tương đương |
| S14 | [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization) | 200; đọc hướng dẫn | QLoRA/NF4/LoRA |
| S15 | [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/) | 200; đọc schema/serving | JSON schema, grammar, constrained decoding |
| S16 | [Annotation Quality in ABSA](https://arxiv.org/abs/2605.03624) | 200; đọc abstract | IAA và downstream impact của nguồn nhãn |
| S17 | [BERT-pair ABSA](https://aclanthology.org/N19-1035/) | 200; đọc metadata/abstract | Tiền lệ biến ABSA thành sentence-pair |
| S18 | [AC-MIMLLN](https://arxiv.org/abs/2010.02656) | 200; đọc abstract | Multi-instance/multi-label aspect-specific evidence |
| S19 | [Semantic-preserved ACSA augmentation](https://arxiv.org/abs/2506.07148) | 200; đọc abstract | Augmentation có ràng buộc ngữ nghĩa |
| S20 | [NEU-ESC](https://arxiv.org/abs/2506.23524) | 200; đọc abstract | Dataset giáo dục Việt gần chủ đề, khác taxonomy |

## Quy tắc bằng chứng

- Chỉ dùng số liệu benchmark khi bài/model card đã mở được trong phiên này.
- Số trên UIT-VSFC của BamiBERT/GoiBERT là task sentiment/topic hoặc benchmark ABSA khác; không chuyển thành pair F1 của `Final_Data`.
- Model card, preprint và leaderboard là bằng chứng mô tả; chưa phải tái lập độc lập.
- Qwen3.5/Qwen3.8 được giữ trong khảo sát model mới nhưng **ngoài phạm vi bài FJCAI này** vì thời gian/phạm vi; không coi là encoder supervised thay thế.
- Google AI Overview và kết quả tìm kiếm chỉ dùng để tìm URL, không dùng làm nguồn số liệu.

## Audit thực hiện

- Bằng chứng lưu tạm gồm 31 bản ghi: 8 URL tìm kiếm/catalog và 23 URL tài liệu trực tiếp.
- Trong 23 URL trực tiếp, 22 tải thành công và 1 URL HuTieuBERT README trả 404 sau hai lần thử. Bảng S01–S20 là 20 nguồn được trích dẫn; một số nguồn có cả bản `abs` và bản HTML/model card nên số nguồn trích dẫn không phải số URL.
- Các URL chính phản hồi HTTP 200 khi kiểm tra lại: FJCAI, arXiv, Hugging Face, ACL Anthology, PEFT và vLLM.
- Đã chạy `git diff --check` sau khi tạo tài liệu.
- Chưa chạy huấn luyện, chưa tải checkpoint vào repo, chưa đo GPU/latency.
- Nghiên cứu không thay đổi file model, dữ liệu hay checkpoint hiện có.
- Hai lần gọi web MCP không khả dụng; Google trả trang chuyển hướng JavaScript ở một số truy vấn. Lệnh fetch công khai phải chạy qua quyền mạng escalated. Logger toàn cục của skill không ghi được vì thư mục `/home/conanwinner/.research_sessions/` chỉ đọc; file này là audit cục bộ của phiên.
