from typing import List

import torch
from rapidnlp_datasets.sequence_classification import (
    CsvFileReaderForSequenceClassification,
    ExampleForSequenceClassification,
    ExampleParserForSequenceClassification,
    JsonlFileReaderForSequenceClassification,
)


def _padding_features(features, pad_id=0, max_length=512):
    padded_features = {}
    for feature in features:
        for k in feature.keys():
            if k not in padded_features:
                padded_features[k] = []
            value = feature[k]
            padded_value = value + [pad_id] * (max_length - len(value))
            padded_features[k].append(padded_value)
    # to tensor
    return {k: torch.LongTensor(v) for k, v in padded_features.items()}


class DatasetForSequenceClassification(torch.utils.data.Dataset):
    """Dataset for sequence classification in PyTorch"""

    @classmethod
    def from_jsonl_files(cls, input_files, vocab_file, label_to_id=None, **kwargs):
        reader = JsonlFileReaderForSequenceClassification()
        instances = reader.read_files(input_files, **kwargs)
        return cls.from_instances(instances, vocab_file, label_to_id=label_to_id, **kwargs)

    @classmethod
    def from_csv_files(cls, input_files, vocab_file, label_to_id=None, sep=",", **kwargs):
        reader = CsvFileReaderForSequenceClassification()
        instances = reader.read_files(input_files, sep=sep, **kwargs)
        return cls.from_instances(instances, vocab_file, label_to_id=label_to_id, **kwargs)

    @classmethod
    def from_tsv_files(cls, input_files, vocab_file, label_to_id=None, sep="\t", **kwargs):
        reader = CsvFileReaderForSequenceClassification()
        instances = reader.read_files(input_files, sep=sep, **kwargs)
        return cls.from_instances(instances, vocab_file, label_to_id=label_to_id, **kwargs)

    @classmethod
    def from_instances(
        cls,
        instances,
        vocab_file,
        label_to_id=None,
        max_sequence_length=512,
        do_lower_case=True,
        add_special_tokens=True,
        **kwargs
    ):
        parser = ExampleParserForSequenceClassification(
            vocab_file,
            label_to_id=label_to_id,
            max_sequence_length=max_sequence_length,
            do_lower_case=do_lower_case,
            **kwargs
        )
        examples = []
        for e in parser.parse_instances(instances, add_special_tokens=add_special_tokens, **kwargs):
            examples.append(e)
        return cls.from_examples(examples, **kwargs)

    @classmethod
    def from_examples(
        cls, examples: List[ExampleForSequenceClassification], pad_id=0, max_sequence_length=512, **kwargs
    ):
        return cls(examples=examples, pad_id=pad_id, max_sequence_length=max_sequence_length, **kwargs)

    def __init__(
        self, examples: List[ExampleForSequenceClassification], pad_id=0, max_sequence_length=512, **kwargs
    ) -> None:
        super().__init__()
        self.examples = examples
        self.pad_id = pad_id
        self.max_sequence_length = max_sequence_length

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
            "label": e.label,
        }
        return features, labels

    @property
    def batch_padding_collate(self):
        """Padding sequence to max length in a batch."""

        def _collate_fn(batch):
            max_length = max([len(x["input_ids"]) for x, _ in batch])
            features = _padding_features([x for x, _ in batch], pad_id=self.pad_id, max_length=max_length)
            labels = {"label": torch.LongTensor([y["label"] for _, y in batch])}
            return features, labels

        return _collate_fn

    @property
    def fixed_padding_collate(self):
        """Padding sequence to fixed length."""

        def _collate_fn(batch):
            features = _padding_features([x for x, _ in batch], pad_id=self.pad_id, max_length=self.max_sequence_length)
            labels = {"label": torch.LongTensor([y["label"] for _, y in batch])}
            return features, labels

        return _collate_fn
