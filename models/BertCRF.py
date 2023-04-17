# -*- coding: utf-8 -*-
import json

import torch
from torch import nn
from torch.utils.data import Dataset
from torchcrf import CRF
from transformers import BertModel, BertTokenizer

MAX_LEN = 100
label2id = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC": 5, "I-LOC": 6}
tokenizer = BertTokenizer.from_pretrained("pytorch_bert")


class BertCRFModel(nn.Module):
    def __init__(self, bert_model_path, hidden_size, nun_labels, dropout_rate, is_biLSTM=False, LSTM_hidden_size=128,
                 LSTM_input_size=128):
        super(BertCRFModel, self).__init__()
        self.num_labels = nun_labels
        self.bert = BertModel.from_pretrained(bert_model_path)
        for name, param in self.bert.named_parameters():
            param.requires_grad = True
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        output_size = hidden_size
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.crf = CRF(num_tags=nun_labels, batch_first=True)
        self.criterion = nn.CrossEntropyLoss()
        self.is_biLSTM = is_biLSTM
        if self.is_biLSTM:
            self.biLSTM = nn.LSTM(LSTM_hidden_size, LSTM_input_size, num_layers=1, bidirectional=True, batch_first=True)
            output_size = LSTM_input_size * 2
        self.linear = nn.Linear(output_size, nun_labels)

    def forward(self, input_ids, labels, token_type_ids=None, input_mask=None):
        emissions = self.get_outputs(input_ids, token_type_ids, input_mask)
        loss = -1 * self.crf(emissions, labels, mask=input_mask)
        return loss

    def get_outputs(self, input_ids, token_type_ids=None, input_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask, return_dict=True)
        outputs = outputs["last_hidden_state"]
        if self.is_biLSTM:
            outputs, _ = self.biLSTM(outputs)
        outputs = self.dropout_layer(outputs)
        emissions = self.linear(outputs)
        return emissions

    def predict(self, input_ids, token_type_ids=None, input_mask=None):
        emissions = self.get_outputs(input_ids, token_type_ids, input_mask)
        return self.crf.decode(emissions, input_mask)


class MSRAData(Dataset):
    def __init__(self, path):
        self.data = self.load_data(path)

    def load_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.loads(f.read())
            d = [self.trans_data(_) for _ in d]
        return d

    def trans_data(self, data):
        """
        将data数据格式转换成比较方便处理的格式，减少训练时准备batch的时间
        :param data:
        :return:
        """
        text = data["text"]
        # labels = torch.zeros(len(text), len(text), dtype=torch.int64)
        labels = [0] * len(text)
        for ent in data["entities"]:
            labels[ent["start"]] = label2id["B-" + ent["type"]]
            for i in range(ent["start"] + 1, ent["end"]):
                labels[i] = label2id["I-" + ent["type"]]
        if len(labels) >= MAX_LEN:
            labels = labels[:MAX_LEN]
        else:
            labels += [0] * (MAX_LEN - len(labels))
        labels = [0] + labels + [0]
        return text, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def collate_fn(batch):
    texts = [_[0] for _ in batch]
    labels = [_[1] for _ in batch]
    labels = torch.LongTensor(labels)
    tokens = tokenizer.batch_encode_plus(texts, return_tensors="pt", truncation=True, max_length=MAX_LEN + 2,
                                         padding='max_length')
    return (tokens["input_ids"], tokens["token_type_ids"], tokens["attention_mask"], labels)
