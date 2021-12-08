from typing import List

from rapidnlp_datasets.simcse import (
    CsvFileReaderForSimCSE,
    ExampleForSimCSE,
    ExampleParserForSimCSE,
    JsonlFileReaderForSimCSE,
)

from . import utils
from .dataset import PTDataset


class DatasetForSimCSE(PTDataset):
    """Dataset for SimCSE in PyTorch"""

    @classmethod
    def from_jsonl_files(
        cls,
        input_files,
        vocab_file,
        sequence_column="sequence",
        with_pos_sequence=False,
        with_neg_sequence=False,
        pos_sequence_column="pos_sequence",
        neg_sequence_column="neg_sequence",
        **kwargs
    ):
        reader = JsonlFileReaderForSimCSE(
            with_pos_sequence=with_pos_sequence,
            with_neg_sequence=with_neg_sequence,
            sequence_column=sequence_column,
            pos_sequence_column=pos_sequence_column,
            neg_sequence_column=neg_sequence_column,
            **kwargs
        )
        instances = reader.read_files(input_files, **kwargs)
        return cls.from_instances(
            instances, vocab_file, with_pos_sequence=with_pos_sequence, with_neg_sequence=with_neg_sequence, **kwargs
        )

    @classmethod
    def from_csv_files(
        cls,
        input_files,
        vocab_file,
        sep=",",
        sequence_column=0,
        with_pos_sequence=False,
        with_neg_sequence=False,
        pos_sequence_column=1,
        neg_sequence_column=2,
        **kwargs
    ):
        reader = CsvFileReaderForSimCSE(
            with_pos_sequence=with_pos_sequence,
            with_neg_sequence=with_neg_sequence,
            sequence_column=sequence_column,
            pos_sequence_column=pos_sequence_column,
            neg_sequence_column=neg_sequence_column,
            **kwargs
        )
        instances = reader.read_files(input_files, sep=sep, **kwargs)
        return cls.from_instances(
            instances, vocab_file, with_pos_sequence=with_pos_sequence, with_neg_sequence=with_neg_sequence, **kwargs
        )

    @classmethod
    def from_instances(
        cls, instances, vocab_file, with_pos_sequence=False, with_neg_sequence=False, add_special_tokens=True, **kwargs
    ):
        parser = ExampleParserForSimCSE(
            vocab_file, with_pos_sequence=with_pos_sequence, with_neg_sequence=with_neg_sequence, **kwargs
        )
        examples = []
        for e in parser.parse_instances(instances, add_special_tokens=add_special_tokens, **kwargs):
            if not e:
                continue
            examples.append(e)
        return cls.from_examples(
            examples, with_pos_sequence=with_pos_sequence, with_neg_sequence=with_neg_sequence, **kwargs
        )

    @classmethod
    def from_examples(
        cls,
        examples: List[ExampleForSimCSE],
        with_pos_sequence=False,
        with_neg_sequence=False,
        max_sequence_length=512,
        **kwargs
    ):
        valid_examples = []
        for e in examples:
            if len(e.input_ids) > max_sequence_length:
                continue
            if with_pos_sequence and len(e.pos_input_ids) > max_sequence_length:
                continue
            if with_neg_sequence and len(e.neg_input_ids) > max_sequence_length:
                continue
            valid_examples.append(e)

        return cls(
            examples=valid_examples, with_pos_sequence=with_pos_sequence, with_neg_sequence=with_neg_sequence, **kwargs
        )

    def __init__(
        self, examples: List[ExampleForSimCSE], with_pos_sequence=False, with_neg_sequence=False, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.examples = examples
        self.with_pos_sequence = with_pos_sequence
        self.with_neg_sequence = with_neg_sequence

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        e = self.examples[index]
        features = {
            "input_ids": e.input_ids,
            "segment_ids": e.segment_ids,
            "attention_mask": e.attention_mask,
        }
        if self.with_pos_sequence:
            features.update(
                {
                    "pos_input_ids": e.pos_input_ids,
                    "pos_segment_ids": e.pos_segment_ids,
                    "pos_attention_mask": e.pos_attention_mask,
                }
            )
        if self.with_neg_sequence:
            features.update(
                {
                    "neg_input_ids": e.neg_input_ids,
                    "neg_segment_ids": e.neg_segment_ids,
                    "neg_attention_mask": e.neg_attention_mask,
                }
            )
        return features, None

    @property
    def batch_padding_collate(self):
        """Padding sequence to max length in batch"""

        def _collate_fn(batch):
            max_length = 0
            max_length = max([max_length] + [len(x["input_ids"]) for x, _ in batch])
            if self.with_pos_sequence:
                max_length = max([max_length] + [len(x["pos_input_ids"]) for x, _ in batch])
            if self.with_neg_sequence:
                max_length = max([max_length] + [len(x["neg_input_ids"]) for x, _ in batch])
            padded_features = utils.padding_features([x for x, _ in batch], pad_id=self.pad_id, max_length=max_length)
            return padded_features, None

        return _collate_fn

    @property
    def fixed_padding_collate(self):
        """Padding sequence to a fixed length"""

        def _collate_fn(batch):
            max_length = self.max_sequence_length
            padded_features = utils.padding_features([x for x, _ in batch], pad_id=self.pad_id, max_length=max_length)
            return padded_features, None

        return _collate_fn
