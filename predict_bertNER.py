# -*- coding: utf-8 -*-
import os

import torch
from transformers import BertTokenizer

from models.BertNER import BertNERModel

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    device = torch.device("cpu")

model_path = "bert_CRF_NER.ckpt"

id2label = {0: "O", 1: "PER", 2: "LOC", 3: "ORG"}
HIDDEN_SIZE = 768
NUM_LABELS = 4
bert_path = "pytorch_bert"
model = BertNERModel(bert_path, HIDDEN_SIZE, NUM_LABELS)

tokenizer = BertTokenizer.from_pretrained(bert_path)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


def get_entities(query):
    token_ids = tokenizer.encode(query, return_tensors="pt")  # 这里需要注意，token_ids中包含了特殊token
    labels = model.predict(token_ids)[0].cpu().numpy()[1:-1]  # 这里对标签去除特殊token对应的标签

    entities = []
    for i, label in enumerate(labels):
        if label > 0:  # 实体的标签都大于0
            if not entities or entities[-1][-1] != id2label[label]:
                entities.append([[i], id2label[label]])
            elif entities[-1][-1] == id2label[label]:
                entities[-1][0].append(i)

        else:
            continue

    res = [(query[index[0]:index[-1] + 1], index[0], index[-1] + 1, label) for index, label in entities]
    return res


print(get_entities("张勇在深圳开心的加班"))
