import json
import logging
import os


def read_jsonl_files_for_unsup_simcse(input_files, sequence_key="sequence", **kwargs):
    if isinstance(input_files, str):
        input_files = [input_files]
    for f in input_files:
        if not os.path.exists(f):
            logging.warning("File %d does not exist, skipped.")
            continue
        with open(f, mode="rt", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                instance = {"sequence": data[sequence_key]}
                yield instance


def read_jsonl_files_for_supervised_simcse(input_files, sequence_key="sequence", pos_sequence_key="pos_sequence", **kwargs):
    if isinstance(input_files, str):
        input_files = [input_files]
    for f in input_files:
        if not os.path.exists(f):
            logging.warning("File %d does not exist, skipped.")
            continue
        with open(f, mode="rt", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                instance = {"sequence": data[sequence_key], "pos_sequence": data[pos_sequence_key]}
                yield instance


def read_jsonl_files_for_hardneg_simcse(
    input_files, sequence_key="sequence", pos_sequence_key="pos_sequence", neg_sequence_key="neg_sequence", **kwargs
):
    if isinstance(input_files, str):
        input_files = [input_files]
    for f in input_files:
        if not os.path.exists(f):
            logging.warning("File %d does not exist, skipped.")
            continue
        with open(f, mode="rt", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                instance = {
                    "sequence": data[sequence_key],
                    "pos_sequence": data[pos_sequence_key],
                    "neg_sequence": data[neg_sequence_key],
                }
                yield instance
