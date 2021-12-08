import logging
from typing import List

import tensorflow as tf
from rapidnlp_datasets.token_classification import (
    ConllFileReaderForTokenClassification,
    ExampleForTokenClassification,
    ExampleParserForTokenClassification,
    JsonlFileReaderForTokenClassification,
)

from . import utils
from .dataset import TFDataset


class TFDatasetForTokenClassification(TFDataset):
    """Dataset for token classification in TensorFlow"""

    def save_tfrecord(self, output_files, **kwargs):
        """Save examples in tfrecord format"""

        def _encode(example: ExampleForTokenClassification):
            feature = {
                "input_ids": utils.int64_feature([int(x) for x in example.input_ids]),
                "segment_ids": utils.int64_feature([int(x) for x in example.segment_ids]),
                "attention_mask": utils.int64_feature([int(x) for x in example.attention_mask]),
                "label_ids": utils.int64_feature([int(x) for x in example.label_ids]),
            }
            return feature

        utils.save_tfrecord(self.examples, _encode, output_files, **kwargs)

    @classmethod
    def from_conll_files(cls, input_files, vocab_file, label_to_id, sep="\t", **kwargs) -> tf.data.Dataset:
        reader = ConllFileReaderForTokenClassification()
        instances = reader.read_files(input_files, sep=sep, **kwargs)
        return cls.from_instances(instances, vocab_file, label_to_id, **kwargs)

    @classmethod
    def from_jsonl_files(cls, input_files, vocab_file, label_to_id, **kwargs) -> tf.data.Dataset:
        reader = JsonlFileReaderForTokenClassification()
        instances = reader.read_files(input_files, **kwargs)
        return cls.from_instances(instances, vocab_file, label_to_id, **kwargs)

    @classmethod
    def from_tfrecord_files(cls, input_files, num_parallel_calls=None, buffer_size=None, **kwargs) -> tf.data.Dataset:
        dataset = utils.read_tfrecord_files(input_files, **kwargs)
        num_parallel_calls = num_parallel_calls or utils.AUTOTUNE
        buffer_size = buffer_size or utils.AUTOTUNE
        # parse example
        features = {
            "input_ids": tf.io.VarLenFeature(tf.int64),
            "segment_ids": tf.io.VarLenFeature(tf.int64),
            "attention_mask": tf.io.VarLenFeature(tf.int64),
            "label_ids": tf.io.VarLenFeature(tf.int64),
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
                tf.cast(tf.sparse.to_dense(x["label_ids"]), tf.int32),
            ),
            num_parallel_calls=num_parallel_calls,
        ).prefetch(buffer_size)
        # do transformation
        d = cls(**kwargs)
        return d(dataset, **kwargs)

    @classmethod
    def from_instances(
        cls, instances, vocab_file, label_to_id, tokenization="bert-wordpiece", **kwargs
    ) -> tf.data.Dataset:
        parser = ExampleParserForTokenClassification(vocab_file, label_to_id, tokenization=tokenization, **kwargs)
        examples = []
        for e in parser.parse_instances(instances, **kwargs):
            if not e:
                continue
            examples.append(e)
        return cls.from_examples(examples, **kwargs)

    @classmethod
    def from_examples(
        cls, examples: List[ExampleForTokenClassification], return_self=False, verbose=True, **kwargs
    ) -> tf.data.Dataset:
        if not examples:
            logging.warning("examples is empty or null, skipped to build dataset.")
            return None

        if verbose:
            n = min(5, len(examples))
            for i in range(n):
                logging.info("Showing NO.%d example: %s", i, examples[i])

        # parse examples to dataset
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
                _to_dataset([e.label_ids for e in examples]),
            )
        )
        # do transformation
        d = cls(examples=examples, **kwargs)
        if return_self:
            return d(dataset, **kwargs), d
        return d(dataset, **kwargs)

    def _filter(self, dataset: tf.data.Dataset, do_filer=True, max_sequence_length=512, **kwargs) -> tf.data.Dataset:
        if not do_filer:
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
        label_key = kwargs.get("label_key", "label_ids")
        dataset = dataset.map(
            lambda a, b, c, y: ({"input_ids": a, "segment_ids": b, "attention_mask": c}, {label_key: y}),
            num_parallel_calls=num_parallel_calls,
        ).prefetch(kwargs.get("buffer_size", utils.AUTOTUNE))
        return dataset

    def _fixed_padding(self, dataset: tf.data.Dataset, pad_id=0, max_sequence_length=512, **kwargs) -> tf.data.Dataset:
        maxlen = tf.constant(max_sequence_length, dtype=tf.int32)
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        padded_shapes = kwargs.get("padded_shapes", ([maxlen, ], [maxlen, ], [maxlen, ], [maxlen, ]))
        padding_values = kwargs.get("padding_values", (pad_id, pad_id, pad_id, pad_id))
        # fmt: on
        dataset = utils.batching_and_padding(dataset, padded_shapes, padding_values, **kwargs)
        return dataset

    def _batch_padding(self, dataset: tf.data.Dataset, pad_id=0, **kwargs) -> tf.data.Dataset:
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        padded_shapes = kwargs.get("padded_shapes", ([None, ], [None, ], [None, ], [None, ]))
        padding_values = kwargs.get("padding_values", (pad_id, pad_id, pad_id, pad_id))
        # fmt: on
        dataset = utils.batching_and_padding(dataset, padded_shapes, padding_values, **kwargs)
        return dataset

    def _bucket_padding(self, dataset: tf.data.Dataset, pad_id=0, **kwargs) -> tf.data.Dataset:
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        padded_shapes = kwargs.get("padded_shapes", ([None, ], [None, ], [None, ], [None, ]))
        padding_values = kwargs.get("padding_values", (pad_id, pad_id, pad_id, pad_id))
        # fmt: on
        dataset = utils.bucketing_and_padding(
            dataset,
            bucket_fn=lambda a, b, c, y: tf.size(a),
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            **kwargs,
        )
        return dataset
