"""
Flask Server — PhoBERT Multi-Task ABSA
=======================================
Aspect-Based Sentiment Analysis server using PhoBERT.

Usage:
    python server.py                              # default: model_final.pt, port 5000
    python server.py --model best_model.pt        # custom model path
    python server.py --port 8080                  # custom port
    python server.py --model best_model.pt --port 8080

API:
    POST /predict
    Body (JSON): {"text": "Giảng viên rất nhiệt tình"}
    Response:    {"text": "...", "results": [...], "threshold": 0.5}
"""

import os
import sys
import re
import argparse
import unicodedata

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoModel, AutoTokenizer
from underthesea import word_tokenize


# ─────────────────────────────────────────────────────────────────────────────
# Config — phải giống hệt với notebook training
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME = 'vinai/phobert-base'
MODEL_PATH = 'model/best_model.pt'  # Đường dẫn hoặc URL của model (ví dụ: best_model.pt)
MAX_SEQ_LENGTH = 64
DEFAULT_THRESHOLD = 0.6

ASPECT_CATEGORIES = [
    'ky_nang_giang_day',    # 0
    'hanh_vi',              # 1
    'de_xuat',              # 2
    'bai_tap',              # 3
    'chuong_trinh_hoc',     # 4
    'kien_thuc',            # 5
    'kinh_nghiem',          # 6
    'cung_cap_tai_lieu',    # 7
    'thiet_bi_day_hoc',     # 8
    'cham_diem',            # 9
    'noi_chung',            # 10
]
NUM_ASPECTS = len(ASPECT_CATEGORIES)

POLARITIES = ['negative', 'neutral', 'positive']  # 0, 1, 2
NUM_POLARITIES = len(POLARITIES)
ID2POL = {i: p for i, p in enumerate(POLARITIES)}

IGNORE_INDEX = -100


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — giống notebook
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Tiền xử lý cho PhoBERT:
    1. Chuẩn hóa Unicode NFC
    2. Trim whitespace thừa, chuẩn hóa dấu câu đầu câu
    3. Word segmentation → underscore (giảng viên → giảng_viên)
    """
    text = unicodedata.normalize('NFC', str(text).strip())
    text = re.sub(r'^[\s.\-]+', '', text)
    text = word_tokenize(text, format='text')
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Model — giống hệt notebook
# ─────────────────────────────────────────────────────────────────────────────
def mean_pooling(last_hidden_state, attention_mask):
    """Mean pooling theo attention mask — ổn định hơn pooler_output."""
    mask = attention_mask.unsqueeze(-1).float()          # (B, L, 1)
    summed = (last_hidden_state * mask).sum(dim=1)       # (B, H)
    counts = mask.sum(dim=1).clamp(min=1e-9)             # (B, 1)
    return summed / counts                               # (B, H)


class BertMultiTaskABSA(nn.Module):
    def __init__(self, model_name, num_aspects, num_polarities,
                 pos_weight=None, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.aspect_head = nn.Linear(hidden, num_aspects)
        self.sentiment_head = nn.Linear(hidden, num_aspects * num_polarities)
        self.num_aspects = num_aspects
        self.num_polarities = num_polarities
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None

    def forward(self, input_ids, attention_mask,
                aspect_labels=None, sentiment_labels=None):
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask)

        # mean pooling thay vì pooler_output
        pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)

        asp_logits = self.aspect_head(pooled)
        sent_logits = self.sentiment_head(pooled).view(
            -1, self.num_aspects, self.num_polarities
        )

        loss = None
        if aspect_labels is not None and sentiment_labels is not None:
            loss_aspect = nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight
            )(asp_logits, aspect_labels)
            loss_sent = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)(
                sent_logits.view(-1, self.num_polarities),
                sentiment_labels.view(-1),
            )
            loss = loss_aspect + loss_sent

        return loss, asp_logits, sent_logits


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def predict(text: str, model, tokenizer, device, threshold=DEFAULT_THRESHOLD):
    """
    Predict aspect categories + sentiment cho một câu.

    Returns:
        list[dict]: mỗi dict có keys: category, sentiment, confidence
    """
    model.eval()
    processed = preprocess_text(text)
    enc = tokenizer(
        processed,
        max_length=MAX_SEQ_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )

    with torch.no_grad():
        _, asp_logits, sent_logits = model(
            enc['input_ids'].to(device),
            enc['attention_mask'].to(device),
        )

    asp_probs = torch.sigmoid(asp_logits).squeeze(0).cpu().numpy()
    sent_pred = torch.argmax(sent_logits, dim=-1).squeeze(0).cpu().numpy()

    results = []
    for i, (prob, cat) in enumerate(zip(asp_probs, ASPECT_CATEGORIES)):
        if prob >= threshold:
            results.append({
                'category': cat,
                'sentiment': ID2POL[int(sent_pred[i])],
                'confidence': round(float(prob), 4),
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Load model from checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_path: str, device: torch.device):
    """
    Load trained BertMultiTaskABSA model từ file .pt

    Args:
        model_path: đường dẫn tới file model (.pt)
        device: torch.device (cpu/cuda)
    Returns:
        model: BertMultiTaskABSA đã load weights
    """
    print(f'Loading PhoBERT encoder: {MODEL_NAME} ...')
    model = BertMultiTaskABSA(
        model_name=MODEL_NAME,
        num_aspects=NUM_ASPECTS,
        num_polarities=NUM_POLARITIES,
    )

    print(f'Loading trained weights: {model_path} ...')
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    print(f'[OK] Model loaded on {device}')
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────
def create_app(model, tokenizer, device):
    """Tạo Flask app với model đã load sẵn."""

    app = Flask(__name__)

    @app.route('/', methods=['GET'])
    def index():
        """Serve the sentiment analysis web interface."""
        return send_from_directory(app.static_folder, 'index.html')

    @app.route('/predict', methods=['GET', 'POST'])
    def api_predict():
        """
        Nhận JSON {"text": "..."} hoặc {"text": "...", "threshold": 0.6}
        Trả về danh sách aspect-sentiment.
        """
        if request.method == 'GET':
            return jsonify({
                'error': 'Use POST /predict with a JSON body containing "text".',
                'example': {
                    'text': 'Giảng viên dạy rất hay nhưng phòng học quá nóng.',
                },
            }), 405

        data = request.get_json(force=True, silent=True)
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in JSON body'}), 400

        text = data['text'].strip()
        if not text:
            return jsonify({'error': '"text" must not be empty'}), 400

        threshold = data.get('threshold', DEFAULT_THRESHOLD)

        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            return jsonify({'error': '"threshold" must be a number'}), 400

        results = predict(text, model, tokenizer, device, threshold)

        return jsonify({
            'text': text,
            'results': results,
            'threshold': threshold,
        })

    @app.route('/health', methods=['GET'])
    def health():
        """Health-check endpoint."""
        return jsonify({'status': 'ok', 'model': MODEL_NAME})

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='PhoBERT Multi-Task ABSA — Flask Server'
    )
    parser.add_argument(
        '--model', type=str, default=MODEL_PATH,
        help=f'Đường dẫn tới file checkpoint weights (.pt). Default: {MODEL_PATH}'
    )
    parser.add_argument(
        '--port', type=int, default=5000,
        help='Port cho Flask server. Default: 5000'
    )
    parser.add_argument(
        '--host', type=str, default='0.0.0.0',
        help='Host để bind. Default: 0.0.0.0'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Chạy Flask ở chế độ debug'
    )
    args = parser.parse_args()

    # ── Kiểm tra file model tồn tại ─────────────────────────────────────
    if not os.path.isfile(args.model):
        print(f'[ERROR] Model file not found: {args.model}')
        sys.exit(1)

    # ── Device ───────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Workaround cho transformers trong __main__ ───────────────────────
    if not hasattr(sys.modules['__main__'], '__file__'):
        dummy = os.path.abspath('dummy_fix.py')
        open(dummy, 'w').close()
        sys.modules['__main__'].__file__ = dummy

    # ── Load tokenizer ───────────────────────────────────────────────────
    print(f'Loading tokenizer: {MODEL_NAME} ...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Load model ───────────────────────────────────────────────────────
    model = load_model(args.model, device)

    # ── Start server ─────────────────────────────────────────────────────
    app = create_app(model, tokenizer, device)
    print(f'\nServer starting at http://{args.host}:{args.port}')
    print('   POST /predict  - Predict aspect + sentiment')
    print('   GET  /health   - Health check\n')
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
