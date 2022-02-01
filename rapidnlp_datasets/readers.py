import json
import logging
import os
import re


def read_conll_files(input_files, sep="\t", feature_column=0, label_column=1, **kwargs):
    if isinstance(input_files, str):
        input_files = [input_files]
    for f in input_files:
        if not os.path.exists(f) or not os.path.isfile(f):
            logging.warning("File %d does not exist, skipped.", f)
            continue
        feature, label = []
        with open(f, mode="rt", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    yield {"feature": feature, "label": label}
                    feature, label = [], []
                    continue
                parts = re.split(sep, line)
                feature.append(parts[feature_column])
                label.append(parts[label_column])
        if feature and label:
            yield {"feature": feature, "label": label}


def read_tsv_files(input_files, sep="\t", skip_rows=0, **kwargs):
    if isinstance(input_files, str):
        input_files = [input_files]
    for f in input_files:
        if not os.path.exists(f):
            logging.warning("File %s does not exist, skipped.", f)
            continue
        with open(f, mode="rt", encoding="utf-8") as fin:
            rows = skip_rows
            while rows > 0:
                fin.readline()
                rows -= 1
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(sep)
                yield parts


def read_csv_files(input_files, sep=",", **kwargs):
    for parts in read_tsv_files(input_files, sep=sep, **kwargs):
        yield parts


def read_jsonl_files(input_files, **kwargs):
    if isinstance(input_files, str):
        input_files = [input_files]
    for f in input_files:
        if not os.path.exists(f):
            logging.warning("File %s does not exist, skipped.", f)
            continue
        with open(f, mode="rt", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                yield data


def read_text_files(input_files, **kwargs):
    if isinstance(input_files, str):
        input_files = [input_files]
    for f in input_files:
        if not os.path.exists(f):
            logging.warning("File %s does not exist, skipped.", f)
            continue
        with open(f, mode="rt", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                yield line
