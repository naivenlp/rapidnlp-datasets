import json
import logging
import os


def read_jsonl_files(input_files, sequence_key="sequence", label_key="label", pair_key="pair", id_key="id", **kwargs):
    if isinstance(input_files, str):
        input_files = [input_files]
    for f in input_files:
        with open(f, mode="rt", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                instance = {
                    "sequence": data[sequence_key],
                    "label": data[label_key],
                }
                if pair_key and pair_key in data:
                    instance.update({"pair": data[pair_key]})
                if id_key in data:
                    instance.update({"id": data[id_key]})
                yield instance
