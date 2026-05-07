# Run Server

## Để chạy server local:
- tạo folder /model trong /Server trước => Download best_model.pt từ drive
- Mở terminal ở thư mục /Server
- Tạo .venv bằng lệnh 
  python -m venv .venv
- Khởi động venv bằng lệnh 
  venv\Scripts\activate
- Cài đặt thư viện:
  pip install -r requirements.txt
- Chạy lệnh:
  python server.py

# Endpoint
## GET /health
URL: http://localhost:5000/health
Body: Không cần
Response: {"status": "ok", "threshold": 0.6, "model": "model/best_model.pt"}

## POST /predict
URL: http://localhost:5000/predict
Body (JSON):
{
    "text": "Giảng viên dạy rất hay nhưng phòng học quá nóng.",
}
- "text": Nội dung cần phân tích.

Response:
{
    "text": "Giảng viên dạy rất hay nhưng phòng học quá nóng.",
    "results": [
        {
            "category": "ky_nang_giang_day",
            "sentiment": "positive",
            "confidence": 0.84
        },
        {
            "category": "thiet_bi_day_hoc",
            "sentiment": "negative",
            "confidence": 0.71
        }
    ],
    "threshold": 0.6
}


