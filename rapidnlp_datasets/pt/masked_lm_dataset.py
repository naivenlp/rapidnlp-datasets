import logging
from typing import List

from rapidnlp_datasets.masked_lm import (
    CsvFileReaderForMaskedLanguageModel,
    ExampleForMaskedLanguageModel,
    ExampleParserForMaskedLanguageModel,
    JsonlFileReaderForMaskedLanguageModel,
    TextFileReaderForMaskedLanguageModel,
)

from . import utils
from .dataset import PTDataset


class DatasetForMaskedLanguageModel(PTDataset):
    """Dataset for mlm in PyTorch"""

    @classmethod
    def from_jsonl_files(cls, input_files, vocab_file, **kwargs):
        reader = JsonlFileReaderForMaskedLanguageModel()
        instances = reader.read_files(input_files, **kwargs)
        return cls.from_instances(instances, vocab_file, **kwargs)

    @classmethod
    def from_text_files(cls, input_files, vocab_file, **kwargs):
        reader = TextFileReaderForMaskedLanguageModel()
        instances = reader.read_files(input_files, **kwargs)
        return cls.from_instances(instances, vocab_file, **kwargs)

    @classmethod
    def from_csv_files(cls, input_files, vocab_file, **kwargs):
        reader = CsvFileReaderForMaskedLanguageModel()
        instances = reader.read_files(input_files, **kwargs)
        return cls.from_instances(instances, vocab_file, **kwargs)

    @classmethod
    def from_instances(cls, instances, vocab_file, max_sequence_length=512, **kwargs):
        parser = ExampleParserForMaskedLanguageModel(vocab_file, **kwargs)
        examples = []
        for e in parser.parse_instances(instances, max_sequence_length=max_sequence_length, **kwargs):
            if not e:
                continue
            examples.append(e)
        return cls.from_examples(examples, max_sequence_length=max_sequence_length, **kwargs)

    @classmethod
    def from_examples(cls, examples: List[ExampleForMaskedLanguageModel], max_sequence_length=512, **kwargs):
        examples = [e for e in examples if len(e.input_ids) <= max_sequence_length]
        logging.info("Load %d examples: ", len(examples))
        return cls(examples=examples, max_sequence_length=max_sequence_length, **kwargs)

    def __init__(self, examples: List[ExampleForMaskedLanguageModel], **kwargs):
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
            "masked_ids": e.masked_ids,
            "masked_pos": e.masked_pos,
        }
        return features, labels

    @property
    def batch_padding_collate(self):
        """Padding sequence to max length in a batch"""

        def _collate_fn(batch):
            max_length = max([len(x) for x, _ in batch])
            padded_features = utils.padding_features([x for x, _ in batch], pad_id=self.pad_id, max_length=max_length)
            padded_labels = utils.padding_labels([y for _, y in batch], pad_id=self.pad_id, max_length=max_length)
            return padded_features, padded_labels

        return _collate_fn

    @property
    def fixed_padding_collate(self):
        """Padding sequence to a fixed length"""

        def _collate_fn(batch):
            max_length = self.max_sequence_length
            padded_features = utils.padding_features([x for x, _ in batch], pad_id=self.pad_id, max_length=max_length)
            padded_labels = utils.padding_labels([y for _, y in batch], pad_id=self.pad_id, max_length=max_length)
            return padded_features, padded_labels

        return _collate_fn
