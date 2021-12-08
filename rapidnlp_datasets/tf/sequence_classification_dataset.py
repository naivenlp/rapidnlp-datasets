import abc
import logging
from typing import Dict, List

import tensorflow as tf
from rapidnlp_datasets.sequence_classification import (
    CsvFileReaderForSequenceClassification,
    ExampleForSequenceClassification,
    ExampleParserForSequenceClassification,
    JsonlFileReaderForSequenceClassification,
)

from . import utils
from .dataset import TFDataset


class TFDatasetForSequenceClassifiation(TFDataset):
    """Datapipe for sequence classification"""

    def save_tfrecord(self, output_files, **kwargs):
        """Save examples to tfrecord"""

        def _encode(example: ExampleForSequenceClassification):
            feature = {
                "input_ids": utils.int64_feature([int(x) for x in example.input_ids]),
                "segment_ids": utils.int64_feature([int(x) for x in example.segment_ids]),
                "attention_mask": utils.int64_feature([int(x) for x in example.attention_mask]),
                "label": utils.int64_feature([int(example.label)]),
            }
            return feature

        if not self.examples:
            logging.warning("self.examples is empty or None, skipped.")
            return
        utils.save_tfrecord(self.examples, _encode, output_files, **kwargs)

    @classmethod
    def from_tfrecord_files(
        cls, input_files, num_parallel_calls=None, buffer_size=None, return_self=False, **kwargs
    ) -> tf.data.Dataset:
        num_parallel_calls = num_parallel_calls or utils.AUTOTUNE
        buffer_size = buffer_size or utils.AUTOTUNE
        # read files
        dataset = utils.read_tfrecord_files(input_files, **kwargs)
        # parse example
        features = {
            "input_ids": tf.io.VarLenFeature(tf.int64),
            "segment_ids": tf.io.VarLenFeature(tf.int64),
            "attention_mask": tf.io.VarLenFeature(tf.int64),
            "label": tf.io.VarLenFeature(tf.int64),
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
                tf.cast(tf.squeeze(tf.sparse.to_dense(x["label"])), tf.int32),
            ),
            num_parallel_calls=num_parallel_calls,
        ).prefetch(buffer_size)
        # do transformation
        d = cls(**kwargs)
        if return_self:
            return d(dataset, **kwargs), d
        return d(dataset, **kwargs)

    @classmethod
    def from_jsonl_files(cls, input_files, vocab_file, label_to_id=None, **kwargs) -> tf.data.Dataset:
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
        cls, instances: List[Dict], vocab_file, label_to_id=None, add_special_tokens=True, do_lower_case=True, **kwargs
    ) -> tf.data.Dataset:
        examples = []
        parser = ExampleParserForSequenceClassification(
            vocab_file, label_to_id=label_to_id, do_lower_case=do_lower_case, **kwargs
        )
        for e in parser.parse_instances(instances, add_special_tokens=add_special_tokens, **kwargs):
            if not e:
                continue
            examples.append(e)
        return cls.from_examples(examples, **kwargs)

    @classmethod
    def from_examples(
        cls, examples: List[ExampleForSequenceClassification], max_sequence_length=512, return_self=False, **kwargs
    ) -> tf.data.Dataset:
        examples = [e for e in examples if len(e.input_ids) <= max_sequence_length]
        if not examples:
            logging.warning("examples is empty or null, skipped to build dataset.")
            return None

        # zip examples to dataset
        def _to_dataset(x, dtype=tf.int32):
            x = tf.ragged.constant(x, dtype=dtype)
            d = tf.data.Dataset.from_tensor_slices(x)
            d = d.map(lambda x: x)
            return d

        dataset = tf.data.Dataset.zip(
            (
                _to_dataset([e.input_ids for e in examples]),
                _to_dataset([e.segment_ids for e in examples]),
                _to_dataset([e.attention_mask for e in examples]),
                _to_dataset([e.label for e in examples]),
            )
        )
        # do transformation
        d = cls(examples=examples, **kwargs)
        if return_self:
            return d(dataset, **kwargs), d
        return d(dataset, **kwargs)

    def _filter(self, dataset: tf.data.Dataset, do_filter=True, max_sequence_length=512, **kwargs) -> tf.data.Dataset:
        if not do_filter:
            return dataset
        dataset = dataset.filter(lambda a, b, c, y: tf.size(a) <= max_sequence_length)
        return dataset

    def _to_dict(self, dataset: tf.data.Dataset, to_dict=True, **kwargs) -> tf.data.Dataset:
        num_parallel_calls = kwargs.get("num_parallel_calls", utils.AUTOTUNE)
        if not to_dict:
            dataset = dataset.map(
                lambda a, b, c, y: ((a, b, c), y),
                num_parallel_calls=num_parallel_calls,
            )
            return dataset
        label_key = kwargs.get("label_key", "label")
        dataset = dataset.map(
            lambda a, b, c, y: ({"input_ids": a, "segment_ids": b, "attention_mask": c}, {label_key: y}),
            num_parallel_calls=num_parallel_calls,
        )
        return dataset

    def _fixed_padding(self, dataset: tf.data.Dataset, **kwargs) -> tf.data.Dataset:
        maxlen = tf.constant(kwargs.get("max_sequence_length", 512), dtype=tf.int32)
        pad_id = tf.constant(kwargs.get("pad_id", 0), dtype=tf.int32)
        # fmt: off
        padded_shapes = kwargs.get("padded_shapes", ([maxlen, ], [maxlen, ], [maxlen, ], []))
        padding_values = kwargs.get("padding_values", (pad_id, pad_id, pad_id, None))
        # fmt: on
        dataset = utils.batching_and_padding(dataset, padded_shapes, padding_values, **kwargs)
        return dataset

    def _batch_padding(self, dataset: tf.data.Dataset, **kwargs) -> tf.data.Dataset:
        pad_id = tf.constant(kwargs.get("pad_id", 0), dtype=tf.int32)
        # fmt: off
        padded_shapes = kwargs.get("padded_shapes", ([None, ], [None, ], [None, ], []))
        padding_values = kwargs.get("padding_values", (pad_id, pad_id, pad_id, None))
        # fmt: on
        dataset = utils.batching_and_padding(dataset, padded_shapes, padding_values, **kwargs)
        return dataset

    def _bucket_padding(self, dataset: tf.data.Dataset, **kwargs) -> tf.data.Dataset:
        pad_id = tf.constant(kwargs.get("pad_id", 0), dtype=tf.int32)
        # fmt: off
        padded_shapes = ([None, ], [None, ], [None, ], [])
        padding_values = (pad_id, pad_id, pad_id, None)
        # fmt: on
        dataset = utils.bucketing_and_padding(
            dataset,
            bucket_fn=lambda a, b, c, y: tf.size(a),
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            **kwargs,
        )
        return dataset
