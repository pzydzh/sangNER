# -*- coding: utf-8 -*-
import os

import torch
from transformers import BertTokenizer

from models.BertCRF import BertCRFModel

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    device = torch.device("cpu")

model_path = "bert_CRF_NER.ckpt"

id2label = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-LOC", 4: "I-LOC", 5: "B-ORG", 6: "I-ORG"}
HIDDEN_SIZE = 768
NUM_LABELS = 7
bert_path = "pytorch_bert"
model = BertCRFModel(bert_path, HIDDEN_SIZE, NUM_LABELS, dropout_rate=0.5)

tokenizer = BertTokenizer.from_pretrained(bert_path)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


def get_entities(query):
    token_ids = tokenizer.encode(query, return_tensors="pt")  # 这里需要注意，token_ids中包含了特殊token
    labels = model.predict(token_ids)[0][1:-1]  # 这里对标签去除特殊token对应的标签

    entities, start = [], False
    for i, label in enumerate(labels):
        if label > 0:  # 实体的标签都大于0
            if id2label[label].startswith("B"):
                start = True
                entities.append([[i], id2label[label][2:]])  # 这里取到类别之后，需要去除前两个字符，B-和I-
            elif start:
                entities[-1][0].append(i)
            else:
                start = False
        else:
            start = False

    res = [(query[index[0]:index[-1] + 1], index[0], index[-1] + 1, label) for index, label in entities]
    return res


print(get_entities("饶世红在深圳开心的加班"))
