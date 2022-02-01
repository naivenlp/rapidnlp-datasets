import logging

import tensorflow as tf

from . import utils
from .dataset import TFDataset


class TFDatasetForQuestionAnswering(TFDataset):
    """Dataset for question answering in TensorFlow."""

    def __init__(
        self,
        examples=None,
        input_ids="input_ids",
        token_type_ids="token_type_ids",
        attention_mask="attention_mask",
        start_positions="start_positions",
        end_positions="end_positions",
        **kwargs
    ) -> None:
        super().__init__(examples, **kwargs)
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.start_positions = start_positions
        self.end_positions = end_positions

    def parse_examples_to_dataset(self):
        input_ids, token_type_ids, attention_mask = [], [], []
        start_positions, end_positions = [], []
        for x in self.examples:
            x = x._asdict()
            input_ids.append(x[self.input_ids])
            token_type_ids.append(x[self.token_type_ids])
            attention_mask.append(x[self.attention_mask])
            start_positions.append(x[self.start_positions])
            end_positions.append(x[self.end_positions])

        def _to_dataset(x, dtype=tf.int32):
            x = tf.ragged.constant(x, dtype=dtype)
            d = tf.data.Dataset.from_tensor_slices(x)
            d = d.map(lambda x: x)
            return d

        dataset = tf.data.Dataset.zip(
            (
                _to_dataset(input_ids, dtype=tf.int32),
                _to_dataset(token_type_ids, dtype=tf.int32),
                _to_dataset(attention_mask, dtype=tf.int32),
                _to_dataset(start_positions, dtype=tf.int32),
                _to_dataset(end_positions, dtype=tf.int32),
            )
        )
        return dataset

    @classmethod
    def from_tfrecord_files(cls, input_files, **kwargs) -> tf.data.Dataset:
        d = cls(
            input_ids=kwargs.pop("input_ids", "input_ids"),
            token_type_ids=kwargs.pop("token_type_ids", "token_type_ids"),
            attention_mask=kwargs.pop("attention_mask", "attention_mask"),
            start_positions=kwargs.pop("start_positions", "start_positions"),
            end_positions=kwargs.pop("end_positions", "end_positions"),
            **kwargs,
        )
        dataset = utils.read_tfrecord_files(input_files, **kwargs)
        # parse examples
        features = {
            d.input_ids: tf.io.VarLenFeature(tf.int64),
            d.token_type_ids: tf.io.VarLenFeature(tf.int64),
            d.attention_mask: tf.io.VarLenFeature(tf.int64),
            d.start_positions: tf.io.VarLenFeature(tf.int64),
            d.end_positions: tf.io.VarLenFeature(tf.int64),
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
                tf.cast(tf.squeeze(tf.sparse.to_dense(x[d.start_positions])), tf.int32),
                tf.cast(tf.squeeze(tf.sparse.to_dense(x[d.end_positions])), tf.int32),
            ),
            num_parallel_calls=utils.AUTOTUNE,
        ).prefetch(utils.AUTOTUNE)
        return d(dataset, **kwargs)

    def _filter(self, dataset: tf.data.Dataset, do_filter=True, max_sequence_length=512, **kwargs) -> tf.data.Dataset:
        if not do_filter:
            return dataset
        dataset = dataset.filter(lambda a, b, c, x, y: tf.size(a) <= max_sequence_length)
        return dataset

    def _to_dict(self, dataset: tf.data.Dataset, to_dict=True, **kwargs) -> tf.data.Dataset:
        num_parallel_calls = kwargs.get("num_parallel_calls", utils.AUTOTUNE)
        if not to_dict:
            dataset = dataset.map(
                lambda a, b, c, x, y: ((a, b, c), (x, y)),
                num_parallel_calls=num_parallel_calls,
            )
            return dataset
        dataset = dataset.map(
            lambda a, b, c, x, y: (
                {self.input_ids: a, self.token_type_ids: b, self.attention_mask: c},
                {self.start_positions: x, self.end_positions: y},
            ),
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
