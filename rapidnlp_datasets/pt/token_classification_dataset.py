from typing import List

from rapidnlp_datasets.token_classification import (
    ConllFileReaderForTokenClassification,
    ExampleForTokenClassification,
    ExampleParserForTokenClassification,
    JsonlFileReaderForTokenClassification,
)

from . import utils
from .dataset import PTDataset


class DatasetForTokenClassification(PTDataset):
    """Dataset for token classification in PyTorch"""

    @classmethod
    def from_jsonl_files(cls, input_files, vocab_file, label_to_id, tokenization="bert-wordpiece", **kwargs):
        reader = JsonlFileReaderForTokenClassification()
        instances = reader.read_files(input_files, **kwargs)
        return cls.from_instances(instances, vocab_file, label_to_id, tokenization=tokenization, **kwargs)

    @classmethod
    def from_conll_files(cls, input_files, vocab_file, label_to_id, tokenization="bert-wordpiece", **kwargs):
        reader = ConllFileReaderForTokenClassification()
        instances = reader.read_files(input_files, **kwargs)
        return cls.from_instances(instances, vocab_file, label_to_id, tokenization=tokenization, **kwargs)

    @classmethod
    def from_instances(cls, instances, vocab_file, label_to_id, tokenization="bert-wordpiece", **kwargs):
        parser = ExampleParserForTokenClassification(vocab_file, label_to_id, tokenization=tokenization, **kwargs)
        examples = []
        for e in parser.parse_instances(instances, **kwargs):
            if not e:
                continue
            examples.append(e)
        return cls.from_examples(examples, **kwargs)

    @classmethod
    def from_examples(cls, examples: List[ExampleForTokenClassification], max_sequence_length=512, **kwargs):
        examples = [e for e in examples if len(e.input_ids) <= max_sequence_length]
        return cls(examples=examples, max_sequence_length=max_sequence_length, **kwargs)

    def __init__(self, examples: List[ExampleForTokenClassification], **kwargs) -> None:
        super().__init__(**kwargs)
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        e = self.examples[index]
        features = {
            "input_ids": e.input_ids,
            "segment_ids": e.segment_ids,
            "attention_mask": e.attention_mask,
        }
        labels = {
            "label_ids": e.label_ids,
        }
        return features, labels

    @property
    def batch_padding_collate(self):
        """Padding sequence to max length in a batch"""

        def _collate_fn(batch):
            max_length = max([len(x["input_ids"]) for x, _ in batch])
            padded_features = utils.padding_features(
                features=[x for x, _ in batch], pad_id=self.pad_id, max_length=max_length
            )
            padded_labels = utils.padding_labels(
                labels=[y for _, y in batch], pad_id=self.pad_id, max_length=max_length
            )
            return padded_features, padded_labels

        return _collate_fn

    @property
    def fixed_padding_collate(self):
        """Padding sequence to a fixed max length"""

        def _collate_fn(batch):
            max_length = self.max_sequence_length
            padded_features = utils.padding_features(
                features=[x for x, _ in batch], pad_id=self.pad_id, max_length=max_length
            )
            padded_labels = utils.padding_labels(
                labels=[y for _, y in batch], pad_id=self.pad_id, max_length=max_length
            )
            return padded_features, padded_labels

        return _collate_fn
