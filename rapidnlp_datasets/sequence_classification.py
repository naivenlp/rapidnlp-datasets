import re
from collections import namedtuple
from typing import Iterator

from tokenizers import BertWordPieceTokenizer

from rapidnlp_datasets.readers import CsvFileReader, JsonlFileReader

ExampleForSequenceClassification = namedtuple(
    "ExampleForSequenceClassification",
    ["tokens", "input_ids", "segment_ids", "attention_mask", "label"],
)


class JsonlFileReaderForSequenceClassification(JsonlFileReader):
    """Jsonl file reader for sequence classification"""

    def _parse_instance(
        self, data, sequence_column="sequence", label_column="label", with_pair=False, pair_column="pair", **kwargs
    ):
        instance = {
            "label": data[label_column],  # label is str, convert to in later
            "sequence": data[sequence_column],
        }
        if with_pair and pair_column in data:
            instance["pair"] = data[pair_column]
        return instance


class CsvFileReaderForSequenceClassification(CsvFileReader):
    """Csv file reader for sequene classification"""

    def _parse_instance(
        self, line, sep=",", sequence_column=0, label_column=-1, with_pair=False, pair_column=2, **kwargs
    ):
        parts = re.split(sep, line)
        instance = {
            "label": parts[label_column],
            "sequence": parts[sequence_column],
        }
        if with_pair:
            instance["pair"] = parts[pair_column]
        return instance


class ExampleParserForSequenceClassification:
    """Parse instance to ExampleForSequenceClassification"""

    def __init__(self, vocab_file, label_to_id=None, do_lower_case=True, **kwargs) -> None:
        self.tokenizer = BertWordPieceTokenizer.from_file(vocab_file, lowercase=do_lower_case)
        if label_to_id is None:
            label_to_id = lambda x: int(x)
        self.label_to_id = label_to_id

    def parse_instance(self, instance, add_special_tokens=True, **kwargs) -> ExampleForSequenceClassification:
        sequence, pair = instance["sequence"], instance.get("pair", None)
        encoding = self.tokenizer.encode(sequence, pair=pair, add_special_tokens=add_special_tokens)
        example = ExampleForSequenceClassification(
            tokens=encoding.tokens,
            input_ids=encoding.ids,
            segment_ids=encoding.type_ids,
            attention_mask=encoding.attention_mask,
            label=self.label_to_id(instance["label"]),
        )
        return example

    def parse_instances(
        self, instances, add_special_tokens=True, **kwargs
    ) -> Iterator[ExampleForSequenceClassification]:
        for instance in instances:
            example = self.parse_instance(instance, add_special_tokens=add_special_tokens, **kwargs)
            if not example:
                continue
            yield example
