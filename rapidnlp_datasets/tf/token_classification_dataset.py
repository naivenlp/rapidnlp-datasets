import logging

import tensorflow as tf

from . import utils
from .dataset import TFDataset


class TFDatasetForTokenClassification(TFDataset):
    """Dataset for token classification in TensorFlow"""

    def __init__(self, examples=None, **kwargs) -> None:
        super().__init__(examples, **kwargs)
        self.input_ids = kwargs.pop("input_ids", "input_ids")
        self.token_type_ids = kwargs.pop("token_type_ids", "token_type_ids")
        self.attention_mask = kwargs.pop("attention_mask", "attention_mask")
        self.labels = kwargs.pop("labels", "labels")

    @classmethod
    def from_tfrecord_files(cls, input_files, **kwargs) -> tf.data.Dataset:
        dataset = utils.read_tfrecord_files(input_files, **kwargs)
        d = cls(examples=None, **kwargs)
        # parse example
        features = {
            d.input_ids: tf.io.VarLenFeature(tf.int64),
            d.token_type_ids: tf.io.VarLenFeature(tf.int64),
            d.attention_mask: tf.io.VarLenFeature(tf.int64),
            d.labels: tf.io.VarLenFeature(tf.int64),
        }
        dataset = dataset.map(
            lambda x: tf.io.parse_example(x, features),
            num_parallel_calls=utils.AUTOTUNE,
        ).prefetch(utils.AUTOTUNE)
        dataset = dataset.map(
            lambda x: (
                tf.cast(tf.sparse.to_dense(x[d.input_ids]), tf.int32),
                tf.cast(tf.sparse.to_dense(x[d.token_type_ids]), tf.int32),
                tf.cast(tf.sparse.to_dense(x[d.attention_mask]), tf.int32),
                tf.cast(tf.sparse.to_dense(x[d.labels]), tf.int32),
            ),
            num_parallel_calls=utils.AUTOTUNE,
        ).prefetch(utils.AUTOTUNE)
        # do transformation
        return d(dataset, **kwargs)

    def parse_examples_to_dataset(self):
        if not self.examples:
            logging.info("self.examples is empty or None, skipped.")
            return None
        input_ids, token_type_ids, attention_mask, labels = [], [], [], []
        for e in self.examples:
            input_ids.append(e.input_ids)
            token_type_ids.append(e.token_type_ids)
            attention_mask.append(e.attention_mask)
            labels.append(e.label_ids)

        # parse examples to dataset
        def _to_dataset(x, dtype=tf.int32):
            x = tf.ragged.constant(x, dtype=dtype)
            d = tf.data.Dataset.from_tensor_slices(x)
            d = d.map(lambda x: x)
            return d

        dataset = tf.data.Dataset.zip(
            (
                _to_dataset(input_ids),
                _to_dataset(token_type_ids),
                _to_dataset(attention_mask),
                _to_dataset(labels),
            )
        )
        return dataset

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
        dataset = dataset.map(
            lambda a, b, c, y: ({self.input_ids: a, self.token_type_ids: b, self.attention_mask: c}, {self.labels: y}),
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
