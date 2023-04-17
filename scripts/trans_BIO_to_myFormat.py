# -*- coding: utf-8 -*-
import json
import os

from tqdm import tqdm

from utils.data_op import write_file

base_dir = os.path.dirname(os.path.dirname(__file__))


def load_data(path, add_example_id=False, add_entity_id=False):
    datas = []
    entity_count = 0
    entity2id = {}
    with open(path, "r", encoding="utf-8") as f:
        sentences = f.read().split("\n\n")
        for sent_index, sentence in enumerate(tqdm(sentences)):
            sentence = sentence.split("\n")
            data = {"text": "", "entity_list": []}
            if add_example_id:
                data["id"] = sent_index + 1
            entity = {"type": "", "value": "", "index": []}
            last_label = "O"
            for i, token in enumerate(sentence):
                if token:
                    item, label = token.split("\t")
                    if not label:
                        continue
                    if last_label != "O" and label == "O":
                        if add_entity_id:
                            if entity["value"] in entity2id:
                                entity["id"] = entity2id[entity["value"]]
                            else:
                                entity_count += 1
                                entity["id"] = entity_count
                                entity2id[entity["value"]] = entity_count
                        entity["index"].append(i)
                        data["entity_list"].append(entity)
                        entity = {"type": "", "value": "", "index": []}
                    if label == "O":
                        last_label = "O"
                    else:
                        last_label = label[2:]
                    if label[0] == "B":
                        entity["type"] = label[2:]
                        entity["value"] += item
                        entity["index"].append(i)
                    elif label[0] == "I":
                        entity["value"] += item
                    data["text"] += item
            if entity["type"]:
                data["entity_list"].append(entity)
            if data["text"]:
                datas.append(json.dumps(data, ensure_ascii=False, indent=4))
        save_path = os.path.split(path)
        save_file = os.path.splitext(save_path[1])[0] + "_myformat.txt"
        write_file(datas, os.path.join(save_path[0], save_file))


if __name__ == "__main__":
    train_path = os.path.join(base_dir, "data/MSRA/msra_train_bio.txt")
    test_path = os.path.join(base_dir, "data/MSRA/msra_test_bio.txt")
    load_data(test_path, add_example_id=True, add_entity_id=True)
