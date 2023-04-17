# -*- coding: utf-8 -*-
import os
import json


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.loads(f.read())
        return d


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=4))


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [_.strip() for _ in lines]
        return lines


def write_file(data, path):
    with open(path, "w", encoding="utf-8") as f:
        data = [_.strip()+"\n" for _ in data if _]
        if data:
            data[-1] = data[-1][:-1]
        f.writelines(data)