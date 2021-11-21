import json
import logging
import os


def read_jsonl_files(input_files, sequence_key="sequence", **kwargs):
    if isinstance(input_files, str):
        input_files = [input_files]
    for f in input_files:
        with open(f, mode="rt", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                instance = {"sequence": data[sequence_key]}
                yield instance
