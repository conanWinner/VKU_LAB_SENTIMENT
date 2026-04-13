#!/usr/bin/env python3
"""Generate the restructured bert_e2e_absa_kaggle.ipynb notebook."""
import json

def md(source):
    """Create a markdown cell."""
    if isinstance(source, str):
        source = source.split('\n')
    lines = []
    for i, line in enumerate(source):
        if i < len(source) - 1:
            lines.append(line + '\n')
        else:
            lines.append(line)
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code(source):
    """Create a code cell."""
    if isinstance(source, str):
        source = source.split('\n')
    lines = []
    for i, line in enumerate(source):
        if i < len(source) - 1:
            lines.append(line + '\n')
        else:
            lines.append(line)
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}

cells = []

# ── Title ──
cells.append(md("""\
# BERT E2E ABSA - Kaggle Training Notebook
Aspect-Based Sentiment Analysis sử dụng BERT fine-tuning.

> Notebook này tự chứa toàn bộ source code, chạy trực tiếp trong cells — không cần upload file `.py` riêng.
>
> **Tiếng Việt:** đổi `MODEL_NAME = 'vinai/phobert-base'` và `DO_LOWER_CASE = False`"""))

# ── 1. Dependencies ──
cells.append(md("## 1. Cài đặt dependencies"))
cells.append(code("!pip install transformers tensorboardX tqdm pytorch-crf -q"))

# ── 2. GPU ──
cells.append(md("## 2. Kiểm tra GPU"))
cells.append(code("""\
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('Memory:', round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2), 'GB')"""))

# ── 3. Seq Utils ──
cells.append(md("""\
## 3. Sequence Utility Functions
Các hàm chuyển đổi giữa các tagging schema (OT, BIO, BIEOS) và trích xuất targeted sentiment."""))

cells.append(code("""\
import numpy as np


def ot2bieos_ts(ts_tag_sequence):
    \"\"\"Convert OT tag sequence to BIEOS format.\"\"\"
    n_tags = len(ts_tag_sequence)
    new_ts_sequence = []
    prev_pos = '$$$'
    for i in range(n_tags):
        cur_ts_tag = ts_tag_sequence[i]
        if cur_ts_tag == 'O' or cur_ts_tag == 'EQ':
            new_ts_sequence.append('O')
            cur_pos = 'O'
        else:
            cur_pos, cur_sentiment = cur_ts_tag.split('-')
            if cur_pos != prev_pos:
                if i == n_tags - 1:
                    new_ts_sequence.append('S-%s' % cur_sentiment)
                else:
                    next_ts_tag = ts_tag_sequence[i + 1]
                    if next_ts_tag == 'O':
                        new_ts_sequence.append('S-%s' % cur_sentiment)
                    else:
                        new_ts_sequence.append('B-%s' % cur_sentiment)
            else:
                if i == n_tags - 1:
                    new_ts_sequence.append('E-%s' % cur_sentiment)
                else:
                    next_ts_tag = ts_tag_sequence[i + 1]
                    if next_ts_tag == 'O':
                        new_ts_sequence.append('E-%s' % cur_sentiment)
                    else:
                        new_ts_sequence.append('I-%s' % cur_sentiment)
        prev_pos = cur_pos
    return new_ts_sequence


def ot2bio_ts(ts_tag_sequence):
    \"\"\"Convert OT tag sequence to BIO format.\"\"\"
    new_ts_sequence = []
    n_tag = len(ts_tag_sequence)
    prev_pos = '$$$'
    for i in range(n_tag):
        cur_ts_tag = ts_tag_sequence[i]
        if cur_ts_tag == 'O':
            new_ts_sequence.append('O')
            cur_pos = 'O'
        else:
            cur_pos, cur_sentiment = cur_ts_tag.split('-')
            if cur_pos == prev_pos:
                new_ts_sequence.append('I-%s' % cur_sentiment)
            else:
                new_ts_sequence.append('B-%s' % cur_sentiment)
        prev_pos = cur_pos
    return new_ts_sequence


def bio2ot_ts(ts_tag_sequence):
    \"\"\"Convert BIO tag sequence back to OT format.\"\"\"
    new_ts_sequence = []
    n_tags = len(ts_tag_sequence)
    for i in range(n_tags):
        ts_tag = ts_tag_sequence[i]
        if ts_tag == 'O' or ts_tag == 'EQ':
            new_ts_sequence.append('O')
        else:
            pos, sentiment = ts_tag.split('-')
            new_ts_sequence.append('T-%s' % sentiment)
    return new_ts_sequence


def tag2ts(ts_tag_sequence):
    \"\"\"Convert BIEOS tag sequence to list of (begin, end, sentiment).\"\"\"
    n_tags = len(ts_tag_sequence)
    ts_sequence, sentiments = [], []
    beg, end = -1, -1
    for i in range(n_tags):
        ts_tag = ts_tag_sequence[i]
        eles = ts_tag.split('-')
        if len(eles) == 2:
            pos, sentiment = eles
        else:
            pos, sentiment = 'O', 'O'
        if sentiment != 'O':
            sentiments.append(sentiment)
        if pos == 'S':
            ts_sequence.append((i, i, sentiment))
            sentiments = []
        elif pos == 'B':
            beg = i
            if len(sentiments) > 1:
                sentiments = [sentiments[-1]]
        elif pos == 'E':
            end = i
            if end > beg > -1 and len(set(sentiments)) == 1:
                ts_sequence.append((beg, end, sentiment))
                sentiments = []
                beg, end = -1, -1
    return ts_sequence


print("✓ Sequence utility functions loaded.")"""))

# ── 4. Data Processing ──
cells.append(md("""\
## 4. Data Processing
Classes và hàm xử lý dữ liệu ABSA: đọc data, tokenize, tạo features, đánh giá kết quả."""))

cells.append(code("""\
import csv
import logging
import os
import sys
from io import open

logger = logging.getLogger(__name__)

SMALL_POSITIVE_CONST = 1e-4


class InputExample(object):
    \"\"\"A single training/test example for simple sequence classification.\"\"\"
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class SeqInputFeatures(object):
    \"\"\"A single set of features of data for the ABSA task.\"\"\"
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, evaluate_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.evaluate_label_ids = evaluate_label_ids


class DataProcessor(object):
    \"\"\"Base class for data converters for sequence classification data sets.\"\"\"
    def get_train_examples(self, data_dir):
        raise NotImplementedError()
    def get_dev_examples(self, data_dir):
        raise NotImplementedError()
    def get_test_examples(self, data_dir):
        raise NotImplementedError()
    def get_labels(self):
        raise NotImplementedError()


class ABSAProcessor(DataProcessor):
    \"\"\"Processor for the ABSA datasets.\"\"\"
    def get_train_examples(self, data_dir, tagging_schema):
        return self._create_examples(data_dir=data_dir, set_type='train', tagging_schema=tagging_schema)

    def get_dev_examples(self, data_dir, tagging_schema):
        return self._create_examples(data_dir=data_dir, set_type='dev', tagging_schema=tagging_schema)

    def get_test_examples(self, data_dir, tagging_schema):
        return self._create_examples(data_dir=data_dir, set_type='test', tagging_schema=tagging_schema)

    def get_labels(self, tagging_schema):
        if tagging_schema == 'OT':
            return []
        elif tagging_schema == 'BIO':
            return ['O', 'EQ', 'B-POS', 'I-POS', 'B-NEG', 'I-NEG', 'B-NEU', 'I-NEU']
        elif tagging_schema == 'BIEOS':
            return ['O', 'EQ', 'B-POS', 'I-POS', 'E-POS', 'S-POS',
                    'B-NEG', 'I-NEG', 'E-NEG', 'S-NEG',
                    'B-NEU', 'I-NEU', 'E-NEU', 'S-NEU']
        else:
            raise Exception("Invalid tagging schema %s..." % tagging_schema)

    def _create_examples(self, data_dir, set_type, tagging_schema):
        examples = []
        file = os.path.join(data_dir, "%s.txt" % set_type)
        class_count = np.zeros(3)
        with open(file, 'r', encoding='UTF-8') as fp:
            sample_id = 0
            for line in fp:
                sent_string, tag_string = line.strip().split('####')
                words = []
                tags = []
                for tag_item in tag_string.split(' '):
                    eles = tag_item.split('=')
                    if len(eles) == 1:
                        raise Exception("Invalid samples %s..." % tag_string)
                    elif len(eles) == 2:
                        word, tag = eles
                    else:
                        word = ''.join((len(eles) - 2) * ['='])
                        tag = eles[-1]
                    words.append(word)
                    tags.append(tag)
                if tagging_schema == 'BIEOS':
                    tags = ot2bieos_ts(tags)
                elif tagging_schema == 'BIO':
                    tags = ot2bio_ts(tags)
                guid = "%s-%s" % (set_type, sample_id)
                text_a = ' '.join(words)
                gold_ts = tag2ts(ts_tag_sequence=tags)
                for (b, e, s) in gold_ts:
                    if s == 'POS':
                        class_count[0] += 1
                    if s == 'NEG':
                        class_count[1] += 1
                    if s == 'NEU':
                        class_count[2] += 1
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=tags))
                sample_id += 1
        print("%s class count: %s" % (set_type, class_count))
        return examples


processors = {
    "laptop14": ABSAProcessor,
    "rest_total": ABSAProcessor,
    "rest_total_revised": ABSAProcessor,
    "rest14": ABSAProcessor,
    "rest15": ABSAProcessor,
    "rest16": ABSAProcessor,
}


print("✓ Data processing classes loaded.")"""))

cells.append(code("""\
def convert_examples_to_seq_features(examples, label_list, tokenizer,
                                     cls_token_at_end=False, pad_on_left=False, cls_token='[CLS]',
                                     sep_token='[SEP]', pad_token=0, sequence_a_segment_id=0,
                                     sequence_b_segment_id=1, cls_token_segment_id=1, pad_token_segment_id=0,
                                     mask_padding_with_zero=True):
    \"\"\"Feature extraction for sequence labeling.\"\"\"
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    max_seq_length = -1
    examples_tokenized = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = []
        labels_a = []
        evaluate_label_ids = []
        words = example.text_a.split(' ')
        wid, tid = 0, 0
        for word, label in zip(words, example.label):
            subwords = tokenizer.tokenize(word)
            tokens_a.extend(subwords)
            if label != 'O':
                labels_a.extend([label] + ['EQ'] * (len(subwords) - 1))
            else:
                labels_a.extend(['O'] * len(subwords))
            evaluate_label_ids.append(tid)
            wid += 1
            tid += len(subwords)
        assert tid == len(tokens_a)
        evaluate_label_ids = np.array(evaluate_label_ids, dtype=np.int32)
        examples_tokenized.append((tokens_a, labels_a, evaluate_label_ids))
        if len(tokens_a) > max_seq_length:
            max_seq_length = len(tokens_a)
    # count on the [CLS] and [SEP]
    max_seq_length += 2
    for ex_index, (tokens_a, labels_a, evaluate_label_ids) in enumerate(examples_tokenized):
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        labels = labels_a + ['O']
        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
            labels = labels + ['O']
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids
            labels = ['O'] + labels
            evaluate_label_ids += 1
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        label_ids = [label_map[label] for label in labels]

        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([0] * padding_length) + label_ids
            evaluate_label_ids += padding_length
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            label_ids = label_ids + ([0] * padding_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        features.append(
            SeqInputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             label_ids=label_ids,
                             evaluate_label_ids=evaluate_label_ids))
    print("maximal sequence length is", max_seq_length)
    return features


def match_ts(gold_ts_sequence, pred_ts_sequence):
    \"\"\"Calculate the number of correctly predicted targeted sentiment.\"\"\"
    tag2tagid = {'POS': 0, 'NEG': 1, 'NEU': 2}
    hit_count, gold_count, pred_count = np.zeros(3), np.zeros(3), np.zeros(3)
    for t in gold_ts_sequence:
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        gold_count[tid] += 1
    for t in pred_ts_sequence:
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        if t in gold_ts_sequence:
            hit_count[tid] += 1
        pred_count[tid] += 1
    return hit_count, gold_count, pred_count


def compute_metrics_absa(preds, labels, all_evaluate_label_ids, tagging_schema):
    \"\"\"Compute micro/macro F1-scores for ABSA task.\"\"\"
    if tagging_schema == 'BIEOS':
        absa_label_vocab = {'O': 0, 'EQ': 1, 'B-POS': 2, 'I-POS': 3, 'E-POS': 4, 'S-POS': 5,
                            'B-NEG': 6, 'I-NEG': 7, 'E-NEG': 8, 'S-NEG': 9,
                            'B-NEU': 10, 'I-NEU': 11, 'E-NEU': 12, 'S-NEU': 13}
    elif tagging_schema == 'BIO':
        absa_label_vocab = {'O': 0, 'EQ': 1, 'B-POS': 2, 'I-POS': 3,
                            'B-NEG': 4, 'I-NEG': 5, 'B-NEU': 6, 'I-NEU': 7}
    elif tagging_schema == 'OT':
        absa_label_vocab = {'O': 0, 'EQ': 1, 'T-POS': 2, 'T-NEG': 3, 'T-NEU': 4}
    else:
        raise Exception("Invalid tagging schema %s..." % tagging_schema)
    absa_id2tag = {}
    for k in absa_label_vocab:
        v = absa_label_vocab[k]
        absa_id2tag[v] = k
    n_tp_ts, n_gold_ts, n_pred_ts = np.zeros(3), np.zeros(3), np.zeros(3)
    ts_precision, ts_recall, ts_f1 = np.zeros(3), np.zeros(3), np.zeros(3)
    n_samples = len(all_evaluate_label_ids)
    class_count = np.zeros(3)
    for i in range(n_samples):
        evaluate_label_ids = all_evaluate_label_ids[i]
        pred_labels = preds[i][evaluate_label_ids]
        gold_labels = labels[i][evaluate_label_ids]
        assert len(pred_labels) == len(gold_labels)
        pred_tags = [absa_id2tag[label] for label in pred_labels]
        gold_tags = [absa_id2tag[label] for label in gold_labels]
        if tagging_schema == 'OT':
            gold_tags = ot2bieos_ts(gold_tags)
            pred_tags = ot2bieos_ts(pred_tags)
        elif tagging_schema == 'BIO':
            gold_tags = ot2bieos_ts(bio2ot_ts(gold_tags))
            pred_tags = ot2bieos_ts(bio2ot_ts(pred_tags))
        g_ts_sequence = tag2ts(ts_tag_sequence=gold_tags)
        p_ts_sequence = tag2ts(ts_tag_sequence=pred_tags)
        hit_ts_count, gold_ts_count, pred_ts_count = match_ts(gold_ts_sequence=g_ts_sequence,
                                                              pred_ts_sequence=p_ts_sequence)
        n_tp_ts += hit_ts_count
        n_gold_ts += gold_ts_count
        n_pred_ts += pred_ts_count
        for (b, e, s) in g_ts_sequence:
            if s == 'POS':
                class_count[0] += 1
            if s == 'NEG':
                class_count[1] += 1
            if s == 'NEU':
                class_count[2] += 1
    for i in range(3):
        n_ts = n_tp_ts[i]
        n_g_ts = n_gold_ts[i]
        n_p_ts = n_pred_ts[i]
        ts_precision[i] = float(n_ts) / float(n_p_ts + SMALL_POSITIVE_CONST)
        ts_recall[i] = float(n_ts) / float(n_g_ts + SMALL_POSITIVE_CONST)
        ts_f1[i] = 2 * ts_precision[i] * ts_recall[i] / (ts_precision[i] + ts_recall[i] + SMALL_POSITIVE_CONST)
    macro_f1 = ts_f1.mean()
    n_tp_total = sum(n_tp_ts)
    n_g_total = sum(n_gold_ts)
    print("class_count:", class_count)
    n_p_total = sum(n_pred_ts)
    micro_p = float(n_tp_total) / (n_p_total + SMALL_POSITIVE_CONST)
    micro_r = float(n_tp_total) / (n_g_total + SMALL_POSITIVE_CONST)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + SMALL_POSITIVE_CONST)
    scores = {'macro-f1': macro_f1, 'precision': micro_p, "recall": micro_r, "micro-f1": micro_f1}
    return scores


print("✓ Feature conversion & metrics functions loaded.")"""))

# ── 5. Model Architecture ──
cells.append(md("""\
## 5. Model Architecture
Các lớp mô hình ABSA: tagger configs, SAN, GRU, LSTM, và BertABSATagger."""))

cells.append(code("""\
import torch
import torch.nn as nn
from transformers import PreTrainedModel, BertModel, BertConfig
from torch.nn import CrossEntropyLoss
from torchcrf import CRF as TorchCRF


# ── Pretrained model archive (legacy, giữ cho tương thích) ──
BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin',
    'bert-large-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin',
    'bert-base-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin',
    'bert-base-multilingual-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin',
    'bert-base-multilingual-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin',
}


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertPreTrainedModel(PreTrainedModel):
    \"\"\"Custom BertPreTrainedModel for ABSA.\"\"\"
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__(*inputs, **kwargs)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


print("✓ BertPreTrainedModel loaded.")"""))

cells.append(code("""\
class TaggerConfig:
    def __init__(self):
        self.hidden_dropout_prob = 0.1
        self.hidden_size = 768
        self.n_rnn_layers = 1
        self.bidirectional = True


class SAN(nn.Module):
    \"\"\"Self Attention Network.\"\"\"
    def __init__(self, d_model, nhead, dropout=0.1):
        super(SAN, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(src2)
        src = self.norm(src)
        return src


class GRU(nn.Module):
    \"\"\"Customized GRU with layer normalization.\"\"\"
    def __init__(self, input_size, hidden_size, bidirectional=True):
        super(GRU, self).__init__()
        self.input_size = input_size
        if bidirectional:
            self.hidden_size = hidden_size // 2
        else:
            self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.Wxrz = nn.Linear(in_features=self.input_size, out_features=2*self.hidden_size, bias=True)
        self.Whrz = nn.Linear(in_features=self.hidden_size, out_features=2*self.hidden_size, bias=True)
        self.Wxn = nn.Linear(in_features=self.input_size, out_features=self.hidden_size, bias=True)
        self.Whn = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=True)
        self.LNx1 = nn.LayerNorm(2*self.hidden_size)
        self.LNh1 = nn.LayerNorm(2*self.hidden_size)
        self.LNx2 = nn.LayerNorm(self.hidden_size)
        self.LNh2 = nn.LayerNorm(self.hidden_size)

    def forward(self, x):
        def recurrence(xt, htm1):
            gates_rz = torch.sigmoid(self.LNx1(self.Wxrz(xt)) + self.LNh1(self.Whrz(htm1)))
            rt, zt = gates_rz.chunk(2, 1)
            nt = torch.tanh(self.LNx2(self.Wxn(xt)) + rt * self.LNh2(self.Whn(htm1)))
            ht = (1.0 - zt) * nt + zt * htm1
            return ht

        steps = range(x.size(1))
        bs = x.size(0)
        hidden = self.init_hidden(bs)
        input = x.transpose(0, 1)
        output = []
        for t in steps:
            hidden = recurrence(input[t], hidden)
            output.append(hidden)
        output = torch.stack(output, 0).transpose(0, 1)

        if self.bidirectional:
            output_b = []
            hidden_b = self.init_hidden(bs)
            for t in steps[::-1]:
                hidden_b = recurrence(input[t], hidden_b)
                output_b.append(hidden_b)
            output_b = output_b[::-1]
            output_b = torch.stack(output_b, 0).transpose(0, 1)
            output = torch.cat([output, output_b], dim=-1)
        return output, None

    def init_hidden(self, bs):
        h_0 = torch.zeros(bs, self.hidden_size).cuda()
        return h_0


class LSTM(nn.Module):
    \"\"\"Customized LSTM with layer normalization.\"\"\"
    def __init__(self, input_size, hidden_size, bidirectional=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        if bidirectional:
            self.hidden_size = hidden_size // 2
        else:
            self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.LNx = nn.LayerNorm(4*self.hidden_size)
        self.LNh = nn.LayerNorm(4*self.hidden_size)
        self.LNc = nn.LayerNorm(self.hidden_size)
        self.Wx = nn.Linear(in_features=self.input_size, out_features=4*self.hidden_size, bias=True)
        self.Wh = nn.Linear(in_features=self.hidden_size, out_features=4*self.hidden_size, bias=True)

    def forward(self, x):
        def recurrence(xt, hidden):
            htm1, ctm1 = hidden
            gates = self.LNx(self.Wx(xt)) + self.LNh(self.Wh(htm1))
            it, ft, gt, ot = gates.chunk(4, 1)
            it = torch.sigmoid(it)
            ft = torch.sigmoid(ft)
            gt = torch.tanh(gt)
            ot = torch.sigmoid(ot)
            ct = (ft * ctm1) + (it * gt)
            ht = ot * torch.tanh(self.LNc(ct))
            return ht, ct

        output = []
        steps = range(x.size(1))
        hidden = self.init_hidden(x.size(0))
        input = x.transpose(0, 1)
        for t in steps:
            hidden = recurrence(input[t], hidden)
            output.append(hidden[0])
        output = torch.stack(output, 0).transpose(0, 1)

        if self.bidirectional:
            hidden_b = self.init_hidden(x.size(0))
            output_b = []
            for t in steps[::-1]:
                hidden_b = recurrence(input[t], hidden_b)
                output_b.append(hidden_b[0])
            output_b = output_b[::-1]
            output_b = torch.stack(output_b, 0).transpose(0, 1)
            output = torch.cat([output, output_b], dim=-1)
        return output, None

    def init_hidden(self, bs):
        h_0 = torch.zeros(bs, self.hidden_size).cuda()
        c_0 = torch.zeros(bs, self.hidden_size).cuda()
        return h_0, c_0


print("✓ Tagger components loaded (TaggerConfig, SAN, GRU, LSTM).")"""))

cells.append(code("""\
class BertABSATagger(BertPreTrainedModel):
    \"\"\"BERT-based Aspect-Based Sentiment Analysis Tagger.\"\"\"
    def __init__(self, bert_config):
        super(BertABSATagger, self).__init__(bert_config)
        self.num_labels = bert_config.num_labels
        self.tagger_config = TaggerConfig()
        self.tagger_config.absa_type = bert_config.absa_type.lower()
        if bert_config.tfm_mode == 'finetune':
            self.bert = BertModel(bert_config)
        else:
            raise Exception("Invalid transformer mode %s!!!" % bert_config.tfm_mode)
        self.bert_dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        if bert_config.fix_tfm:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.tagger = None
        if self.tagger_config.absa_type == 'linear':
            penultimate_hidden_size = bert_config.hidden_size
        else:
            self.tagger_dropout = nn.Dropout(self.tagger_config.hidden_dropout_prob)
            if self.tagger_config.absa_type == 'lstm':
                self.tagger = LSTM(input_size=bert_config.hidden_size,
                                   hidden_size=self.tagger_config.hidden_size,
                                   bidirectional=self.tagger_config.bidirectional)
            elif self.tagger_config.absa_type == 'gru':
                self.tagger = GRU(input_size=bert_config.hidden_size,
                                  hidden_size=self.tagger_config.hidden_size,
                                  bidirectional=self.tagger_config.bidirectional)
            elif self.tagger_config.absa_type == 'tfm':
                self.tagger = nn.TransformerEncoderLayer(d_model=bert_config.hidden_size,
                                                         nhead=12,
                                                         dim_feedforward=4*bert_config.hidden_size,
                                                         dropout=0.1)
            elif self.tagger_config.absa_type == 'san':
                self.tagger = SAN(d_model=bert_config.hidden_size, nhead=12, dropout=0.1)
            elif self.tagger_config.absa_type == 'crf':
                self.tagger = TorchCRF(self.num_labels, batch_first=True)
            else:
                raise Exception('Unimplemented downstream tagger %s...' % self.tagger_config.absa_type)
            penultimate_hidden_size = self.tagger_config.hidden_size
        self.classifier = nn.Linear(penultimate_hidden_size, bert_config.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        tagger_input = outputs[0]
        tagger_input = self.bert_dropout(tagger_input)
        if self.tagger is None or self.tagger_config.absa_type == 'crf':
            logits = self.classifier(tagger_input)
        else:
            if self.tagger_config.absa_type == 'lstm':
                classifier_input, _ = self.tagger(tagger_input)
            elif self.tagger_config.absa_type == 'gru':
                classifier_input, _ = self.tagger(tagger_input)
            elif self.tagger_config.absa_type == 'san' or self.tagger_config.absa_type == 'tfm':
                tagger_input = tagger_input.transpose(0, 1)
                classifier_input = self.tagger(tagger_input)
                classifier_input = classifier_input.transpose(0, 1)
            else:
                raise Exception("Unimplemented downstream tagger %s..." % self.tagger_config.absa_type)
            classifier_input = self.tagger_dropout(classifier_input)
            logits = self.classifier(classifier_input)
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.tagger_config.absa_type != 'crf':
                loss_fct = CrossEntropyLoss()
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs
            else:
                loss = -self.tagger(logits, labels, mask=attention_mask.bool())
                outputs = (loss,) + outputs
        return outputs

    def viterbi_tags(self, logits, mask):
        \"\"\"Decode using torchcrf (replaces manual Viterbi).\"\"\"
        return self.tagger.decode(logits, mask=mask.bool())


print("✓ BertABSATagger loaded.")"""))

# ── 6. Config ──
cells.append(md("""\
## 6. Cấu hình
Chỉnh các tham số tại đây trước khi chạy."""))

cells.append(code("""\
import os

# ============================================================
# CẤU HÌNH CHÍNH - chỉnh tại đây
# ============================================================

# Dataset: 'laptop14', 'rest14', 'rest15', 'rest16'
TASK_NAME = 'laptop14'

# Model pretrained:
#   Tiếng Anh : 'bert-base-uncased'
#   Tiếng Việt: 'vinai/phobert-base' hoặc 'bert-base-multilingual-cased'
MODEL_NAME = 'bert-base-uncased'

# Tagger layer: 'linear', 'gru', 'san', 'tfm', 'crf'
ABSA_TYPE = 'linear'

# Tagging schema: 'BIEOS' (tốt nhất), 'BIO', 'OT'
TAGGING_SCHEMA = 'BIEOS'

# Số bước train tối đa
MAX_STEPS = 1500

# Lưu checkpoint mỗi N steps
SAVE_STEPS = 500

# Log loss mỗi N steps
LOG_STEPS = 100

# Batch size (giảm nếu OOM)
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 8

# Learning rate
LEARNING_RATE = 2e-5

# Seed
SEED = 42

# Đường dẫn data
# Kaggle: thay bằng '/kaggle/input/your-dataset/data/laptop14'
DATA_DIR = f'./data/{TASK_NAME}'

# Thư mục lưu model output
OUTPUT_DIR = f'bert-{ABSA_TYPE}-{TASK_NAME}-finetune'

# True nếu dùng uncased model (bert-base-uncased)
# False nếu dùng PhoBERT hoặc cased model
DO_LOWER_CASE = True

# ============================================================
print('Config:')
print(f'  Task      : {TASK_NAME}')
print(f'  Model     : {MODEL_NAME}')
print(f'  Tagger    : {ABSA_TYPE}')
print(f'  Schema    : {TAGGING_SCHEMA}')
print(f'  Max steps : {MAX_STEPS}')
print(f'  Data dir  : {DATA_DIR}')
print(f'  Output dir: {OUTPUT_DIR}')"""))

# ── 7. Imports & Setup ──
cells.append(md("## 7. Import và khởi tạo"))
cells.append(code("""\
import random
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertConfig, BertTokenizer, get_linear_schedule_with_warmup, WEIGHTS_NAME
from torch.optim import AdamW
from tensorboardX import SummaryWriter
from tqdm.notebook import tqdm, trange

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()
print(f'Using device: {device}, n_gpu: {n_gpu}')"""))

# ── 8. Init Model ──
cells.append(md("## 8. Khởi tạo model và tokenizer"))
cells.append(code("""\
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

processor = processors[TASK_NAME]()
label_list = processor.get_labels(TAGGING_SCHEMA)
num_labels = len(label_list)
print(f'Labels ({num_labels}): {label_list}')

config = BertConfig.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    finetuning_task=TASK_NAME,
    cache_dir='./cache'
)
config.absa_type = ABSA_TYPE
config.tfm_mode = 'finetune'
config.fix_tfm = 0

tokenizer = BertTokenizer.from_pretrained(
    MODEL_NAME,
    do_lower_case=DO_LOWER_CASE,
    cache_dir='./cache'
)

model = BertABSATagger.from_pretrained(
    MODEL_NAME,
    config=config,
    cache_dir='./cache'
)
model.to(device)
if n_gpu > 1:
    model = torch.nn.DataParallel(model)

print('Model loaded successfully.')"""))

# ── 9. Load Data ──
cells.append(md("## 9. Load và cache dữ liệu"))
cells.append(code("""\
def load_dataset(data_dir, task, tokenizer, mode='train'):
    cached_file = os.path.join(
        data_dir,
        f'cached_{mode}_{MODEL_NAME.split("/")[-1]}_{task}'
    )
    if os.path.exists(cached_file):
        print(f'Loading cached features: {cached_file}')
        features = torch.load(cached_file)
    else:
        print(f'Creating features from {data_dir}/{mode}.txt ...')
        proc = processors[task]()
        labels = proc.get_labels(TAGGING_SCHEMA)
        if mode == 'train':
            examples = proc.get_train_examples(data_dir, TAGGING_SCHEMA)
        elif mode == 'dev':
            examples = proc.get_dev_examples(data_dir, TAGGING_SCHEMA)
        else:
            examples = proc.get_test_examples(data_dir, TAGGING_SCHEMA)
        features = convert_examples_to_seq_features(
            examples=examples, label_list=labels, tokenizer=tokenizer,
            cls_token=tokenizer.cls_token, sep_token=tokenizer.sep_token,
            cls_token_segment_id=0, pad_on_left=False, pad_token_segment_id=0
        )
        torch.save(features, cached_file)

    all_input_ids    = torch.tensor([f.input_ids    for f in features], dtype=torch.long)
    all_input_mask   = torch.tensor([f.input_mask   for f in features], dtype=torch.long)
    all_segment_ids  = torch.tensor([f.segment_ids  for f in features], dtype=torch.long)
    all_label_ids    = torch.tensor([f.label_ids    for f in features], dtype=torch.long)
    all_evaluate_label_ids = [f.evaluate_label_ids  for f in features]

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset, all_evaluate_label_ids


train_dataset, _ = load_dataset(DATA_DIR, TASK_NAME, tokenizer, mode='train')
print(f'Train samples: {len(train_dataset)}')"""))

# ── 10. Training ──
cells.append(md("## 10. Training"))
cells.append(code("""\
train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=TRAIN_BATCH_SIZE
)

t_total = MAX_STEPS
num_train_epochs = MAX_STEPS // len(train_dataloader) + 1

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

print(f'Training for {t_total} steps (~{num_train_epochs} epochs)')"""))

cells.append(code("""\
os.makedirs(OUTPUT_DIR, exist_ok=True)
tb_writer = SummaryWriter(log_dir=OUTPUT_DIR)

global_step = 0
tr_loss, logging_loss = 0.0, 0.0
model.zero_grad()
set_seed(SEED)

for epoch in trange(num_train_epochs, desc='Epoch'):
    for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
        model.train()
        batch = tuple(t.to(device) for t in batch)
        inputs = {
            'input_ids':      batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'labels':         batch[3]
        }
        outputs = model(**inputs)
        loss = outputs[0]
        if n_gpu > 1:
            loss = loss.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        tr_loss += loss.item()

        optimizer.step()
        scheduler.step()
        model.zero_grad()
        global_step += 1

        if global_step % LOG_STEPS == 0:
            avg_loss = (tr_loss - logging_loss) / LOG_STEPS
            tb_writer.add_scalar('loss', avg_loss, global_step)
            tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
            print(f'  Step {global_step} | loss: {avg_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}')
            logging_loss = tr_loss

        if global_step % SAVE_STEPS == 0:
            ckpt_dir = os.path.join(OUTPUT_DIR, f'checkpoint-{global_step}')
            os.makedirs(ckpt_dir, exist_ok=True)
            m = model.module if hasattr(model, 'module') else model
            m.save_pretrained(ckpt_dir)
            print(f'  Checkpoint saved: {ckpt_dir}')

        if global_step >= MAX_STEPS:
            break
    if global_step >= MAX_STEPS:
        break"""))

cells.append(code("""\
# Lưu model cuối
m = model.module if hasattr(model, 'module') else model
m.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f'\\nTraining done. Model saved to: {OUTPUT_DIR}')
tb_writer.close()"""))

# ── 11. Evaluation ──
cells.append(md("## 11. Evaluation (Dev & Test)"))
cells.append(code("""\
def evaluate(model, tokenizer, data_dir, task, mode='test'):
    dataset, evaluate_label_ids = load_dataset(data_dir, task, tokenizer, mode=mode)
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=EVAL_BATCH_SIZE)

    eval_loss = 0.0
    preds, out_label_ids = None, None
    crf_logits, crf_mask = [], []

    model.eval()
    for batch in tqdm(dataloader, desc=f'Evaluating [{mode}]'):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'labels':         batch[3]
            }
            outputs = model(**inputs)
            tmp_loss, logits = outputs[:2]
            eval_loss += tmp_loss.mean().item()
            crf_logits.append(logits)
            crf_mask.append(batch[1])

        if preds is None:
            preds         = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds         = np.append(preds,         logits.detach().cpu().numpy(),          axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    m = model.module if hasattr(model, 'module') else model
    if m.tagger_config.absa_type != 'crf':
        preds = np.argmax(preds, axis=-1)
    else:
        crf_logits = torch.cat(crf_logits, dim=0)
        crf_mask   = torch.cat(crf_mask,   dim=0)
        preds = m.tagger.viterbi_tags(logits=crf_logits, mask=crf_mask)

    result = compute_metrics_absa(preds, out_label_ids, evaluate_label_ids, TAGGING_SCHEMA)
    result['eval_loss'] = eval_loss / len(dataloader)
    return result


print('=== Dev set ===')
dev_result = evaluate(model, tokenizer, DATA_DIR, TASK_NAME, mode='dev')
print(dev_result)

print('\\n=== Test set ===')
test_result = evaluate(model, tokenizer, DATA_DIR, TASK_NAME, mode='test')
print(test_result)"""))

# ── 12. Inference ──
cells.append(md("## 12. Inference - Thử dự đoán câu mới"))
cells.append(code("""\
if TAGGING_SCHEMA == 'BIEOS':
    ABSA_ID2TAG = {0:'O',1:'EQ',2:'B-POS',3:'I-POS',4:'E-POS',5:'S-POS',
                   6:'B-NEG',7:'I-NEG',8:'E-NEG',9:'S-NEG',
                   10:'B-NEU',11:'I-NEU',12:'E-NEU',13:'S-NEU'}
elif TAGGING_SCHEMA == 'BIO':
    ABSA_ID2TAG = {0:'O',1:'EQ',2:'B-POS',3:'I-POS',4:'B-NEG',5:'I-NEG',6:'B-NEU',7:'I-NEU'}
else:
    ABSA_ID2TAG = {0:'O',1:'EQ',2:'T-POS',3:'T-NEG',4:'T-NEU'}


def predict_sentence(text, model, tokenizer):
    model.eval()
    words = text.strip().split()
    tokens, evaluate_ids = [], []
    for i, word in enumerate(words):
        evaluate_ids.append(len(tokens) + 1)  # +1 cho [CLS]
        tokens.extend(tokenizer.tokenize(word))

    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    input_ids      = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)], dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs[0]

    m = model.module if hasattr(model, 'module') else model
    if m.tagger_config.absa_type != 'crf':
        pred_ids = np.argmax(logits.detach().cpu().numpy(), axis=-1)[0]
    else:
        pred_ids = m.tagger.viterbi_tags(logits=logits, mask=attention_mask)[0]

    pred_tags = [ABSA_ID2TAG[pred_ids[i]] for i in evaluate_ids]
    if TAGGING_SCHEMA == 'OT':
        pred_tags = ot2bieos_ts(pred_tags)
    elif TAGGING_SCHEMA == 'BIO':
        pred_tags = ot2bieos_ts(bio2ot_ts(pred_tags))

    results = []
    for beg, end, sentiment in tag2ts(pred_tags):
        aspect = ' '.join(words[beg:end+1])
        results.append((aspect, sentiment))
    return results


# Thử với câu mẫu
test_sentences = [
    "The battery life is great but the screen is terrible",
    "Great laptop that offers many great features",
    "The food was amazing but the service was slow",
]
for sent in test_sentences:
    result = predict_sentence(sent, model, tokenizer)
    print(f'Input : {sent}')
    print(f'Output: {result}')
    print()"""))

# ── 13. Load Checkpoint ──
cells.append(md("## 13. (Tùy chọn) Load checkpoint để evaluate lại"))
cells.append(code("""\
CHECKPOINT_DIR = f'{OUTPUT_DIR}/checkpoint-{MAX_STEPS}'

if os.path.exists(CHECKPOINT_DIR):
    print(f'Loading checkpoint from {CHECKPOINT_DIR}')
    model_ckpt = BertABSATagger.from_pretrained(CHECKPOINT_DIR)
    model_ckpt.to(device)
    result = evaluate(model_ckpt, tokenizer, DATA_DIR, TASK_NAME, mode='test')
    print('Test result:', result)
else:
    print(f'Checkpoint not found: {CHECKPOINT_DIR}')"""))

# ── Build notebook ──
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

output_path = "/home/conanwinner/Desktop/_CODE/VKU_Lab_Sentiment/Final/bert_e2e_absa_kaggle.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✓ Notebook written to {output_path}")
print(f"  Total cells: {len(cells)}")
