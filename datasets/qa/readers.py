import abc
import json
import logging
import os
from typing import Dict, List


def read_dureader_rubost(input_files, **kwargs):
    if isinstance(input_files, str):
        input_files = [input_files]
    for input_file in input_files:
        with open(input_file, mode="rt", encoding="utf-8") as fin:
            info = json.load(fin)
        for d in info["data"]:
            paragraphs = d["paragraphs"]
            for p in paragraphs:
                context = p["context"]
                for qa in p["qas"]:
                    question = qa["question"]
                    _id = qa["id"]
                    if not qa["answers"]:
                        continue
                    answer = qa["answers"][0]["text"]
                    instance = {"context": context, "question": question, "answer": answer, "id": _id}
                    yield instance


def read_dureader_checklist(input_files, **kwargs):
    if isinstance(input_files, str):
        input_files = [input_files]
    for input_file in input_files:
        with open(input_file, mode="rt", encoding="utf-8") as fin:
            info = json.load(fin)
        for data in info["data"]:
            for p in data["paragraphs"]:
                title = p["title"]
                context = p["context"]
                for qa in p["qas"]:
                    if qa["is_impossible"]:
                        continue
                    question = qa["question"]
                    answer = qa["answers"][0]["text"]
                    instance = {"context": title + context, "question": question, "answer": answer, "id": qa["id"]}
                    yield instance
