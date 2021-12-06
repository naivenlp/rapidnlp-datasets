import abc
import json
import logging
import os
import re


class AbstractFileReader(abc.ABC):
    """Abstract file reader"""

    @abc.abstractmethod
    def read_files(self, input_files, **kwargs):
        raise NotImplementedError()


class JsonlFileReader(AbstractFileReader):
    """Jsonl file reader"""

    def __init__(self) -> None:
        super().__init__()

    def read_files(self, input_files, **kwargs):
        if isinstance(input_files, str):
            input_files = [input_files]
        for f in input_files:
            if not os.path.exists(f) or not os.path.isfile(f):
                logging.warning("File %d does not exist, skipped.", f)
                continue
            with open(f, mode="rt", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    instance = self._parse_instance(data, **kwargs)
                    if not instance:
                        continue
                    yield instance

    @abc.abstractmethod
    def _parse_instance(self, data, **kwargs):
        raise NotImplementedError()


class CsvFileReader(AbstractFileReader):
    """CSV file reader"""

    def __init__(self) -> None:
        super().__init__()

    def read_files(self, input_files, **kwargs):
        if isinstance(input_files, str):
            input_files = [input_files]
        for f in input_files:
            if not os.path.exists(f) or not os.path.isfile(f):
                logging.warning("File %d does not exist, skipped.", f)
                continue
            with open(f, mode="rt", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    instance = self._parse_instance(line, **kwargs)
                    if not instance:
                        continue
                    yield instance

    @abc.abstractmethod
    def _parse_instance(self, line, sep=",", **kwargs):
        raise NotImplementedError()


class ConllFileReader(AbstractFileReader):
    """CONLL file reader"""

    def read_files(self, input_files, feature_column=0, label_column=1, sep="\t", **kwargs):
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
                        instance = self._parse_instance(feature, label, **kwargs)
                        yield instance
                        feature, label = [], []
                        continue
                    parts = re.split(sep, line)
                    feature.append(parts[feature_column])
                    label.append(parts[label_column])
            if feature and label:
                instance = self._parse_instance(feature, label, **kwargs)
                yield instance

    @abc.abstractmethod
    def _parse_instance(self, feature, label, **kwargs):
        raise NotImplementedError()
