from collections import namedtuple
from typing import Iterator

from tokenizers import BertWordPieceTokenizer

from .charlevel_tokenizer import BertCharLevelTokenizer
from .parsers import AbstractExampleParser
from .readers import ConllFileReader, CsvFileReader, JsonlFileReader

ExampleForTokenClassification = namedtuple(
    "ExampleForTokenClassification", ["tokens", "input_ids", "segment_ids", "attention_mask", "label_ids"]
)


class JsonlFileReaderForTokenClassification(JsonlFileReader):
    """Jsonl file reader for token classification"""

    def _parse_instance(self, data, feature_column="feature", label_column="label", **kwargs):
        instance = {
            "features": data[feature_column],
            "labels": data[label_column],
        }
        return instance


class ConllFileReaderForTokenClassification(ConllFileReader):
    """CONLL file reader for token classification"""

    def _parse_instance(self, feature, label, **kwargs):
        instance = {
            "features": feature,
            "labels": label,
        }
        return instance


class ExampleParserForTokenClassification(AbstractExampleParser):
    """Parse instance to ExampleForTokenClassification"""

    def __init__(self, vocab_file, label_to_id, tokenization="bert-charlevel", do_lower_case=True, **kwargs) -> None:
        super().__init__()
        self.label_to_id = label_to_id
        assert tokenization in ["bert-charlevel", "bert-wordpiece"]
        if tokenization == "bert-charlevel":
            self.tokenizer = BertCharLevelTokenizer.from_file(vocab_file, lowercase=do_lower_case)
        if tokenization == "bert-wordpiece":
            self.tokenizer = BertWordPieceTokenizer.from_file(vocab_file, lowercase=do_lower_case)

    def parse_instance(self, instance, **kwargs) -> ExampleForTokenClassification:
        tokens = instance["features"]
        input_ids = [self.tokenizer.token_to_id(x) for x in tokens]
        segment_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)
        e = ExampleForTokenClassification(
            tokens=tokens,
            input_ids=input_ids,
            segment_ids=segment_ids,
            attention_mask=attention_mask,
            label_ids=[self.label_to_id(x) for x in instance["labels"]],
        )
        return e

    def parse_instances(self, instances, **kwargs) -> Iterator[ExampleForTokenClassification]:
        for instance in instances:
            e = self.parse_instance(instance, **kwargs)
            if not e:
                continue
            yield e
