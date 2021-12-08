import abc
import logging
from typing import Dict, List

import tensorflow as tf
from rapidnlp_datasets.question_answering import (
    ExampleForQuestionAnswering,
    ExampleParserForQuestionAnswering,
    JsonlFileReaderForQuestionAnswering,
    read_dureader_checklist,
    read_dureader_rubost,
)

from . import utils
from .dataset import TFDataset


class TFDatasetForQuestionAnswering(TFDataset):
    """Dataset for question answering in TensorFlow."""

    def save_tfrecord(self, output_files, **kwargs):
        """Save examples to tfrecord"""

        def _encoding(example: ExampleForQuestionAnswering):
            feature = {
                "input_ids": utils.int64_feature([int(x) for x in example.input_ids]),
                "segment_ids": utils.int64_feature([int(x) for x in example.segment_ids]),
                "attention_mask": utils.int64_feature([int(x) for x in example.attention_mask]),
                "start_positions": utils.int64_feature([int(example.start_positions)]),
                "end_positions": utils.int64_feature([int(example.end_positions)]),
            }
            return feature

        if not self.examples:
            logging.warning("self.examples is empty or None, skippped")
            return
        utils.save_tfrecord(self.examples, _encoding, output_files, **kwargs)

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
            "start_positions": tf.io.VarLenFeature(tf.int64),
            "end_positions": tf.io.VarLenFeature(tf.int64),
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
                tf.cast(tf.squeeze(tf.sparse.to_dense(x["start_positions"])), tf.int32),
                tf.cast(tf.squeeze(tf.sparse.to_dense(x["end_positions"])), tf.int32),
            ),
            num_parallel_calls=num_parallel_calls,
        ).prefetch(buffer_size)
        # do transformation
        d = cls(**kwargs)
        return d(dataset, **kwargs)

    @classmethod
    def from_dureader_robust(cls, input_files, vocab_file, tokenization="bert-wordpiece", **kwargs) -> tf.data.Dataset:
        instances = read_dureader_rubost(input_files, **kwargs)
        return cls.from_instances(instances, vocab_file, tokenization=tokenization, **kwargs)

    @classmethod
    def from_dureader_checklist(
        cls, input_files, vocab_file, tokenization="bert-wordpiece", **kwargs
    ) -> tf.data.Dataset:
        instances = read_dureader_checklist(input_files, **kwargs)
        return cls.from_instances(instances, vocab_file, tokenization=tokenization, **kwargs)

    @classmethod
    def from_jsonl_files(cls, input_files, vocab_file, tokenization="bert-wordpiece", **kwargs) -> tf.data.Dataset:
        reader = JsonlFileReaderForQuestionAnswering()
        instances = reader.read_files(input_files, **kwargs)
        return cls.from_instances(instances, vocab_file, tokenization=tokenization, **kwargs)

    @classmethod
    def from_instances(
        cls, instances: List[Dict], vocab_file, tokenization="bert-wordpiece", **kwargs
    ) -> tf.data.Dataset:
        examples = []
        parser = ExampleParserForQuestionAnswering(vocab_file, tokenization=tokenization, **kwargs)
        for e in parser.parse_instances(instances, **kwargs):
            if not e:
                continue
            examples.append(e)
        if not examples:
            logging.warning("examples is empty or null, skipped to build dataset.")
            return None
        logging.info("Read %d examples in total.", len(examples))
        return cls.from_examples(examples, **kwargs)

    @classmethod
    def from_examples(
        cls, examples: List[ExampleForQuestionAnswering], max_sequence_length=512, return_self=False, **kwargs
    ) -> tf.data.Dataset:
        examples = [e for e in examples if len(e.input_ids) <= max_sequence_length]
        d = cls(examples=examples, **kwargs)

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
                _to_dataset(x=[e.start_positions for e in examples], dtype=tf.int32),
                _to_dataset(x=[e.end_positions for e in examples], dtype=tf.int32),
            )
        )
        # do transformation
        if return_self:
            return d(dataset, **kwargs), d
        return d(dataset, **kwargs)

    def _filter(self, dataset: tf.data.Dataset, do_filter=True, max_sequence_length=512, **kwargs) -> tf.data.Dataset:
        if not do_filter:
            return dataset
        dataset = dataset.filter(lambda a, b, c, x, y: tf.size(a) <= max_sequence_length)
        return dataset

    def _to_dict(
        self, dataset: tf.data.Dataset, to_dict=True, start_key="start", end_key="end", **kwargs
    ) -> tf.data.Dataset:
        num_parallel_calls = kwargs.get("num_parallel_calls", utils.AUTOTUNE)
        if not to_dict:
            dataset = dataset.map(
                lambda a, b, c, x, y: ((a, b, c), (x, y)),
                num_parallel_calls=num_parallel_calls,
            )
            return dataset
        dataset = dataset.map(
            lambda a, b, c, x, y: ({"input_ids": a, "segment_ids": b, "attention_mask": c}, {start_key: x, end_key: y}),
            num_parallel_calls=num_parallel_calls,
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
        maxlen = tf.constant(kwargs.get("max_sequence_length", 512), dtype=tf.int32)
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
