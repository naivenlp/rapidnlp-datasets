from typing import List

import torch
from rapidnlp_datasets.question_answering import (
    ExampleForQuestionAnswering,
    ExampleParserForQuestionAnswering,
    JsonlFileReaderForQuestionAnswering,
    read_dureader_checklist,
    read_dureader_rubost,
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


def _padding_labels(labels):
    padded_labels = {
        "start_positions": torch.LongTensor([x["start_positions"] for x in labels]),
        "end_positions": torch.LongTensor([x["end_positions"] for x in labels]),
    }
    return padded_labels


class DatasetForQuestionAnswering(torch.utils.data.Dataset):
    """Dataset for question answering in PyTorch"""

    @classmethod
    def from_dureader_robust(cls, input_files, vocab_file, tokenization="bert-wordpiece", **kwargs):
        instances = read_dureader_rubost(input_files, **kwargs)
        return cls.from_instances(instances, vocab_file, tokenization=tokenization, **kwargs)

    @classmethod
    def from_dureader_checklist(cls, input_files, vocab_file, tokenization="bert-charlevel", **kwargs):
        instances = read_dureader_checklist(input_files, **kwargs)
        return cls.from_instances(instances, vocab_file, tokenization=tokenization, **kwargs)

    @classmethod
    def from_jsonl_files(cls, input_files, vocab_file, tokenization="bert-wordpiece", **kwargs):
        reader = JsonlFileReaderForQuestionAnswering()
        instances = reader.read_files(input_files, **kwargs)
        return cls.from_instances(instances, vocab_file, tokenization=tokenization, **kwargs)

    @classmethod
    def from_instances(cls, instances, vocab_file, tokenization="bert-wordpiece", **kwargs):
        parser = ExampleParserForQuestionAnswering(vocab_file, tokenization=tokenization, **kwargs)
        examples = []
        for e in parser.parse_instances(instances, **kwargs):
            if not e:
                continue
            examples.append(e)
        return cls.from_examples(examples, **kwargs)

    @classmethod
    def from_examples(cls, examples: List[ExampleForQuestionAnswering], pad_id=0, max_sequence_length=512, **kwargs):
        examples = [e for e in examples if len(e.input_ids) <= max_sequence_length]
        return cls(examples=examples, pad_id=pad_id, max_sequence_length=max_sequence_length, **kwargs)

    def __init__(
        self, examples: List[ExampleForQuestionAnswering], pad_id=0, max_sequence_length=512, **kwargs
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
            "start_positions": e.start_positions,
            "end_positions": e.end_positions,
        }
        return features, labels

    @property
    def batch_padding_collate(self):
        """Padding sequence to max length in batch"""

        def _collate_fn(batch):
            max_length = max([len(x["input_ids"]) for x, _ in batch])
            features = _padding_features([x for x, _ in batch], pad_id=self.pad_id, max_length=max_length)
            labels = _padding_labels([y for _, y in batch])
            return features, labels

        return _collate_fn

    @property
    def fixed_padding_collate(self):
        """Padding sequence to fixed length"""

        def _collate_fn(batch):
            max_length = self.max_sequence_length
            features = _padding_features([x for x, _ in batch], pad_id=self.pad_id, max_length=max_length)
            labels = _padding_labels([y for _, y in batch])
            return features, labels

        return _collate_fn
