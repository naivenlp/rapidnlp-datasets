import abc
import logging
from typing import Dict, List

import tensorflow as tf
from smile_datasets import utils
from smile_datasets.dataset import AbcDataset
from tokenizers import BertWordPieceTokenizer

from .example import ExampleForQuestionAnswering
from .parsers import ParserForQuestionAnswering
from .readers import read_dureader_checklist, read_dureader_rubost, read_jsonl_files


class DatasetForQuestionAnswering(abc.ABC):
    """ """

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __getitem__(self, index) -> ExampleForQuestionAnswering:
        raise NotImplementedError()

    def __iter__(self) -> ExampleForQuestionAnswering:
        for idx in range(len(self)):
            yield self[idx]

    def save_tfrecord(self, output_files, **kwargs):
        """Save examples to tfrecord"""

        def _encoding(example: ExampleForQuestionAnswering):
            feature = {
                "input_ids": utils.int64_feature([int(x) for x in example.input_ids]),
                "segment_ids": utils.int64_feature([int(x) for x in example.segment_ids]),
                "attention_mask": utils.int64_feature([int(x) for x in example.attention_mask]),
                "start": utils.int64_feature([int(example.start)]),
                "end": utils.int64_feature([int(example.end)]),
            }
            return feature

        utils.save_tfrecord(iter(self), _encoding, output_files, **kwargs)


class DatapipeForQuestionAnswering(AbcDataset):
    """ """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    def from_tfrecord_files(cls, input_files, num_parallel_calls=None, buffer_size=None, **kwargs) -> tf.data.Dataset:
        dataset = utils.read_tfrecord_files(input_files, **kwargs)
        num_parallel_calls = num_parallel_calls or utils.AUTOTUNE
        buffer_size = buffer_size or utils.AUTOTUNE
        # parse examples
        features = {
            "input_ids": tf.io.VarLenFeature(tf.int64),
            "segment_ids": tf.io.VarLenFeature(tf.int64),
            "attention_mask": tf.io.VarLenFeature(tf.int64),
            "start": tf.io.VarLenFeature(tf.int64),
            "end": tf.io.VarLenFeature(tf.int64),
        }
        dataset = dataset.map(
            lambda x: tf.io.parse_example(x, features),
            num_parallel_calls=num_parallel_calls,
        ).prefetch(buffer_size)
        dataset = dataset.map(
            lambda x: (
                tf.cast(tf.sparse.to_dense(x["input_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["segment_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["attention_mask"]), tf.int32),
                tf.cast(tf.squeeze(tf.sparse.to_dense(x["start"])), tf.int32),
                tf.cast(tf.squeeze(tf.sparse.to_dense(x["end"])), tf.int32),
            ),
            num_parallel_calls=num_parallel_calls,
        ).prefetch(buffer_size)
        # do transformation
        d = cls(**kwargs)
        return d(dataset, **kwargs)

    @classmethod
    def from_dureader_robust(cls, input_files, tokenizer=None, vocab_file=None, **kwargs) -> tf.data.Dataset:
        instances = read_dureader_rubost(input_files, **kwargs)
        return cls.from_instances(instances, tokenizer=tokenizer, vocab_file=vocab_file, **kwargs)

    @classmethod
    def from_dureader_checklist(cls, input_files, tokenizer=None, vocab_file=None, **kwargs) -> tf.data.Dataset:
        instances = read_dureader_checklist(input_files, **kwargs)
        return cls.from_instances(instances, tokenizer=tokenizer, vocab_file=vocab_file, **kwargs)

    @classmethod
    def from_jsonl_files(cls, input_files, tokenizer=None, vocab_file=None, **kwargs) -> tf.data.Dataset:
        instances = read_jsonl_files(input_files, **kwargs)
        return cls.from_instances(instances, tokenizer=tokenizer, vocab_file=vocab_file, **kwargs)

    @classmethod
    def from_instances(
        cls, instances: List[Dict], tokenizer: BertWordPieceTokenizer = None, vocab_file=None, **kwargs
    ) -> tf.data.Dataset:
        """Build tf.data.Dataset from json instances.

        Args:
            instances: List instance of dict, each instance contains keys `context`, `question`, `answer` and `id`
            tokenizer: Tokenizer used to tokenize text
            vocab_file: The vocab path to build tokenizer. The `tokenizer` or `vocab_file` must be provided!

        Returns:
            Instance of tf.data.Dataset, can be used to fit to tf.keras.Model directly.
        """
        examples = []
        parser = ParserForQuestionAnswering(tokenizer=tokenizer, vocab_file=vocab_file, **kwargs)
        for instance in instances:
            e = parser.parse(instance, **kwargs)
            if not e:
                continue
            examples.append(e)
        logging.info("Read %d examples in total.", len(examples))
        return cls.from_examples(examples, **kwargs)

    @classmethod
    def from_dataset(cls, dataset: DatasetForQuestionAnswering, **kwargs) -> tf.data.Dataset:
        examples = []
        for idx in range(len(dataset)):
            examples.append(dataset[idx])
        return cls.from_examples(examples, **kwargs)

    @classmethod
    def from_examples(cls, examples: List[ExampleForQuestionAnswering], **kwargs) -> tf.data.Dataset:
        d = cls(**kwargs)

        def _to_dataset(x, dtype=tf.int32):
            x = tf.ragged.constant(x, dtype=dtype)
            d = tf.data.Dataset.from_tensor_slices(x)
            d = d.map(lambda x: x)
            return d

        dataset = tf.data.Dataset.zip(
            (
                _to_dataset(x=[e.input_ids for e in examples], dtype=tf.int32),
                _to_dataset(x=[e.segment_ids for e in examples], dtype=tf.int32),
                _to_dataset(x=[e.attention_mask for e in examples], dtype=tf.int32),
                _to_dataset(x=[e.start for e in examples], dtype=tf.int32),
                _to_dataset(x=[e.end for e in examples], dtype=tf.int32),
            )
        )

        return d(dataset, **kwargs)

    def _filter(self, dataset: tf.data.Dataset, max_sequence_length=512, **kwargs) -> tf.data.Dataset:
        dataset = dataset.filter(lambda a, b, c, x, y: tf.size(a) <= max_sequence_length)
        return dataset

    def _to_dict(self, dataset: tf.data.Dataset, start_key="start", end_key="end", **kwargs) -> tf.data.Dataset:
        dataset = dataset.map(
            lambda a, b, c, x, y: ({"input_ids": a, "segment_ids": b, "attention_mask": c}, {start_key: x, end_key: y}),
            num_parallel_calls=kwargs.get("num_parallel_calls", utils.AUTOTUNE),
        ).prefetch(kwargs.get("buffer_size", utils.AUTOTUNE))
        return dataset

    def _batch_padding(self, dataset: tf.data.Dataset, **kwargs) -> tf.data.Dataset:
        pad_id = tf.constant(kwargs.get("pad_id", 0), dtype=tf.int32)
        # fmt: off
        padded_shapes = kwargs.get("padded_shapes", ([None, ], [None, ], [None, ], [], []))
        padding_values = kwargs.get("padding_values", (pad_id, pad_id, pad_id, None, None))
        # fmt: on
        dataset = utils.batching_and_padding(dataset, padded_shapes, padding_values, **kwargs)
        return dataset

    def _fixed_padding(self, dataset: tf.data.Dataset, **kwargs) -> tf.data.Dataset:
        maxlen = tf.constant(kwargs.get("max_sequence_length", 0), dtype=tf.int32)
        pad_id = tf.constant(kwargs.get("pad_id", 0), dtype=tf.int32)
        # fmt: off
        padded_shapes = kwargs.get("padded_shapes", ([maxlen, ], [maxlen, ], [maxlen, ], [], []))
        padding_values = kwargs.get("padding_values", (pad_id, pad_id, pad_id, None, None))
        # fmt: on
        dataset = utils.batching_and_padding(dataset, padded_shapes, padding_values, **kwargs)
        return dataset

    def _bucket_padding(self, dataset: tf.data.Dataset, **kwargs) -> tf.data.Dataset:
        pad_id = tf.constant(kwargs.get("pad_id", 0), dtype=tf.int32)
        # fmt: off
        padded_shapes = ([None, ], [None, ], [None, ], [], [])
        padding_values = (pad_id, pad_id, pad_id, None, None)
        # fmt: on
        dataset = utils.bucketing_and_padding(
            dataset,
            bucket_fn=lambda a, b, c, x, y: tf.size(a),
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            **kwargs,
        )
        return dataset
