import json
import logging
import os
import re


def read_conll_files(input_files, sep="[\\s\t]+", **kwargs):
    if isinstance(input_files, str):
        input_files = [input_files]
    for f in input_files:
        if not os.path.exists(f):
            logging.warning("File %s does not exist, skipped.", f)
            continue
        feature, label = [], []
        with open(f, mode="rt", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    instance = {"feature": feature, "label": label}
                    yield instance
                    feature, label = [], []
                parts = re.split(sep, line)
                if len(parts) != 2:
                    continue
                feature.append(parts[0].strip())
                label.append(parts[1].strip())

        if feature and label:
            instance = {"feature": feature, "label": label}
            yield instance


def read_jsonl_files(input_files, feature_key="feature", label_key="label", **kwargs):
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
                instance = {"feature": data[feature_key], "label": data[label_key]}
                yield instance
