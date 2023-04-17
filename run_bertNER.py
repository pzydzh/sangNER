# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.BertNER import BertNERModel, MSRAData, collate_fn

base_dir = os.path.dirname(__file__)
BATCH_SIZE = 4
HIDDEN_SIZE = 768
NUM_LABELS = 4
EPOCHS = 2
LR = 0.01

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    device = torch.device("cpu")

bert_path = "pytorch_bert"
test_data_path = os.path.join(base_dir, "data/MSRA/msra_test_bio.json")
train_data_path = os.path.join(base_dir, "data/MSRA/msra_train_bio.json")
test_dataset = MSRAData(test_data_path)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         num_workers=0,
                         collate_fn=collate_fn)
train_dataset = MSRAData(train_data_path)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=0,
                          collate_fn=collate_fn)

model = BertNERModel(bert_path, HIDDEN_SIZE, NUM_LABELS)
model.train()

bert_lr = 5e-5
other_lr = 1e-3
bert_params = []
other_params = []
for name, param in model.named_parameters():
    if name.startswith("bert"):
        bert_params.append(param)
    else:
        other_params.append(param)
parameters = [{"params": bert_params, "lr": bert_lr}, {"params": other_params, "lr": other_lr}]

optimizer = Adam(parameters, lr=LR)
criterion = nn.CrossEntropyLoss()


def multihead_loss(logits, labels, eval=False):
    """
    直接采用sklearn提供的方法进行计算。这里传递进来的logits和labels都是tensor类型的
    :param logits:
    :param labels:
    :return:
    """
    if not eval:
        logits = torch.argmax(logits, dim=-1)
    logits = logits.view(size=(-1,)).cpu()
    labels = labels.view(size=(-1,)).cpu()
    # [None, 'binary' (default), 'micro', 'macro', 'samples', 'weighted']
    average = "macro"
    precision = precision_score(logits, labels, average=average)
    recall = recall_score(logits, labels, average=average)
    f1 = f1_score(logits, labels, average=average)
    accuracy = accuracy_score(logits, labels)
    res = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }
    return res


def evaluate(data_loader):
    model.eval()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch_data in data_loader:
            input_ids = batch_data[0].to(device)
            labels = batch_data[-1].to(device)
            outputs = model(input_ids=input_ids)
            outputs = outputs.view(-1, NUM_LABELS)
            labels = labels.view(-1)
            labels = labels.data.cpu().numpy()

            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)
    predict_all = torch.Tensor(predict_all)
    labels_all = torch.Tensor(labels_all)
    res = multihead_loss(predict_all, labels_all, True)
    return res


dev_best_f1 = 0
for epoch in range(EPOCHS):
    with tqdm(train_loader) as tt:
        for idx, batch_data in enumerate(tt):
            input_ids = batch_data[0].to(device)
            labels = batch_data[-1].to(device)
            outputs = model(input_ids=input_ids)
            model.zero_grad()
            loss = criterion(outputs.permute(0, 2, 1), labels)
            loss.backward()
            optimizer.step()

            res = multihead_loss(outputs, labels)
            recall, precision, f1, accuracy = res["recall"], res["precision"], res["f1"], res["accuracy"]
            tt.set_postfix(loss=round(loss.item(), 3),
                           recall=round(recall.item(), 3),
                           precision=round(precision.item(), 3),
                           f1=round(f1.item(), 3),
                           accuracy=round(accuracy.item(), 3))

        res = evaluate(test_loader)
        dev_recall, dev_precision, dev_f1, dev_acc = res["recall"], res["precision"], res["f1"], res["accuracy"]
        if dev_f1 > dev_best_f1:
            dev_best_f1 = dev_f1
            torch.save(model.state_dict(), "bert_CRF_NER.ckpt")

        print(f"epoch:{epoch:4}, Dev F1:{dev_f1:>5.2}, Dev recall: {dev_recall:>5.2}, Dev precision: {dev_precision:>5.2}")
