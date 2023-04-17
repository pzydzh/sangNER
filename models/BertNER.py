# -*- coding: utf-8 -*-
import json

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer

MAX_LEN = 100
label2id = {"O": 0, "PER": 1, "LOC": 2, "ORG": 3}
tokenizer = BertTokenizer.from_pretrained("pytorch_bert")


class BertNERModel(nn.Module):
    def __init__(self, bert_model_path, hidden_size, nun_labels):
        super(BertNERModel, self).__init__()
        self.num_labels = nun_labels
        self.bert = BertModel.from_pretrained(bert_model_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.linear = nn.Linear(hidden_size, nun_labels)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs):
        outputs = self.bert(inputs, return_dict=True)
        outputs = outputs["last_hidden_state"]
        outputs = self.linear(outputs)
        return outputs

    def predict(self, token_ids):
        labels = self.forward(token_ids)
        labels = nn.functional.softmax(labels, dim=1)
        labels = torch.argmax(labels, dim=2, keepdim=False)
        return labels


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
        labels = [0] * len(text)
        for ent in data["entities"]:
            labels[ent["start"]] = label2id[ent["type"]]
            for i in range(ent["start"] + 1, ent["end"]):
                labels[i] = label2id[ent["type"]]
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
