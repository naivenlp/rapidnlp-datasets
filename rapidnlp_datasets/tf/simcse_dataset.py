import logging
from typing import List

import tensorflow as tf

from . import utils
from .dataset import TFDataset


class TFDatasetForSimCSE(TFDataset):
    """Dataset for SimCSE in tensorflow"""

    def __init__(self, examples, with_positive_sequence=False, with_negative_sequence=False, **kwargs) -> None:
        super().__init__(examples, **kwargs)
        self.examples = examples
        self.with_positive_sequence = with_positive_sequence
        self.with_negative_sequence = with_negative_sequence

    @classmethod
    def from_tfrecord_files(cls, input_files, with_positive_sequence=False, with_negative_sequence=False, **kwargs):
        num_parallel_calls = kwargs.get("num_parallel_calls", utils.AUTOTUNE)
        buffer_size = kwargs.get("buffer_size", utils.AUTOTUNE)

        def _collect_features():
            seq_features = {
                "input_ids": tf.io.VarLenFeature(tf.int64),
                "token_type_ids": tf.io.VarLenFeature(tf.int64),
                "attention_mask": tf.io.VarLenFeature(tf.int64),
            }
            pos_features, neg_features = {}, {}
            if with_positive_sequence:
                pos_features = {
                    "pos_input_ids": tf.io.VarLenFeature(tf.int64),
                    "pos_token_type_ids": tf.io.VarLenFeature(tf.int64),
                    "pos_attention_mask": tf.io.VarLenFeature(tf.int64),
                }
            if with_negative_sequence:
                neg_features = {
                    "neg_input_ids": tf.io.VarLenFeature(tf.int64),
                    "neg_token_type_ids": tf.io.VarLenFeature(tf.int64),
                    "neg_attention_mask": tf.io.VarLenFeature(tf.int64),
                }
            seq_features.update(pos_features)
            seq_features.update(neg_features)
            return seq_features

        def _parse_fn(x):
            seq_values = (
                tf.cast(tf.sparse.to_dense(x["input_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["token_type_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["attention_mask"]), tf.int32),
            )
            pos_values, neg_values = (), ()
            if with_positive_sequence:
                pos_values = (
                    tf.cast(tf.sparse.to_dense(x["pos_input_ids"]), tf.int32),
                    tf.cast(tf.sparse.to_dense(x["pos_token_type_ids"]), tf.int32),
                    tf.cast(tf.sparse.to_dense(x["pos_attention_mask"]), tf.int32),
                )
            if with_negative_sequence:
                neg_values = (
                    tf.cast(tf.sparse.to_dense(x["neg_input_ids"]), tf.int32),
                    tf.cast(tf.sparse.to_dense(x["neg_token_type_ids"]), tf.int32),
                    tf.cast(tf.sparse.to_dense(x["neg_attention_mask"]), tf.int32),
                )
            seq_values += pos_values
            seq_values += neg_values
            return seq_values

        dataset = utils.read_tfrecord_files(input_files, **kwargs)
        dataset = dataset.map(
            lambda x: tf.io.parse_example(x, _collect_features()),
            num_parallel_calls=num_parallel_calls,
        ).prefetch(buffer_size)
        dataset = dataset.map(_parse_fn, num_parallel_calls=num_parallel_calls).prefetch(buffer_size)
        d = cls(
            examples=None,
            with_positive_sequence=with_positive_sequence,
            with_negative_sequence=with_negative_sequence,
            **kwargs,
        )
        return d(dataset, **kwargs)

    def _parse_group(self, examples, prefix=""):
        input_ids, token_type_ids, attention_mask = [], [], []
        for e in examples:
            input_ids.append(getattr(e, prefix + "input_ids"))
            token_type_ids.append(getattr(e, prefix + "token_type_ids"))
            attention_mask.append(getattr(e, prefix + "attention_mask"))
        return input_ids, token_type_ids, attention_mask

    def parse_examples_to_dataset(self):
        input_ids, token_type_ids, attention_mask = self._parse_group(self.examples, prefix="")

        def _to_dataset(x, dtype=tf.int32):
            x = tf.ragged.constant(x, dtype=dtype)
            d = tf.data.Dataset.from_tensor_slices(x)
            d = d.map(lambda x: x)
            return d

        zipping_datasets = (
            _to_dataset(input_ids, dtype=tf.int32),
            _to_dataset(token_type_ids, dtype=tf.int32),
            _to_dataset(attention_mask, dtype=tf.int32),
        )
        if self.with_positive_sequence:
            pos_input_ids, pos_token_type_ids, pos_attention_mask = self._parse_group(self.examples, prefix="pos_")
            zipping_datasets += (
                _to_dataset(pos_input_ids, dtype=tf.int32),
                _to_dataset(pos_token_type_ids, dtype=tf.int32),
                _to_dataset(pos_attention_mask, dtype=tf.int32),
            )
        if self.with_negative_sequence:
            neg_input_ids, neg_token_type_ids, neg_attention_mask = self._parse_group(self.examples, prefix="neg_")
            zipping_datasets += (
                _to_dataset(neg_input_ids, dtype=tf.int32),
                _to_dataset(neg_token_type_ids, dtype=tf.int32),
                _to_dataset(neg_attention_mask, dtype=tf.int32),
            )
        dataset = tf.data.Dataset.zip(zipping_datasets)
        return dataset

    def _filter(self, dataset: tf.data.Dataset, do_filter=True, max_sequence_length=512, **kwargs) -> tf.data.Dataset:
        if not do_filter:
            return dataset
        if not self.with_positive_sequence and not self.with_negative_sequence:
            dataset = dataset.filter(lambda a, b, c: tf.size(a) <= max_sequence_length)
            return dataset
        if not self.with_negative_sequence:
            dataset = dataset.filter(
                lambda a, b, c, d, e, f: tf.size(a) <= max_sequence_length and tf.size(d) <= max_sequence_length
            )
            return dataset
        dataset = dataset.filter(
            lambda a, b, c, d, e, f, g, h, i: tf.size(a) <= max_sequence_length
            and tf.size(d) <= max_sequence_length
            and tf.size(g) <= max_sequence_length
        )
        return dataset

    def _to_tuple(self, dataset, num_parallel_calls=None, **kwargs):
        if not self.with_positive_sequence and not self.with_negative_sequence:
            dataset = dataset.map(
                lambda a, b, c: ((a, b, c), None),
                num_parallel_calls=num_parallel_calls,
            ).prefetch(utils.AUTOTUNE)
            return dataset
        if not self.with_negative_sequence:
            dataset = dataset.map(
                lambda a, b, c, d, e, f: ((a, b, c, d, e, f), None),
                num_parallel_calls=num_parallel_calls,
            ).prefetch(utils.AUTOTUNE)
            return dataset
        dataset = dataset.map(
            lambda a, b, c, d, e, f, g, h, i: ((a, b, c, d, e, f, g, h, i), None),
            num_parallel_calls=num_parallel_calls,
        ).prefetch(utils.AUTOTUNE)
        return dataset

    def _to_dict(self, dataset: tf.data.Dataset, to_dict=True, num_parallel_calls=None, **kwargs) -> tf.data.Dataset:
        if not to_dict:
            return self._to_tuple(dataset, num_parallel_calls=num_parallel_calls, **kwargs)
        if not self.with_positive_sequence and not self.with_negative_sequence:
            dataset = dataset.map(
                lambda a, b, c: (
                    {
                        "input_ids": a,
                        "token_type_ids": b,
                        "attention_mask": c,
                    },
                    None,
                ),
                num_parallel_calls=num_parallel_calls,
            ).prefetch(utils.AUTOTUNE)
            return dataset
        if not self.with_negative_sequence:
            dataset = dataset.map(
                lambda a, b, c, d, e, f: (
                    {
                        "input_ids": a,
                        "token_type_ids": b,
                        "attention_mask": c,
                        "pos_input_ids": d,
                        "pos_token_type_ids": e,
                        "pos_attention_mask": f,
                    },
                    None,
                ),
                num_parallel_calls=num_parallel_calls,
            ).prefetch(utils.AUTOTUNE)
            return dataset
        dataset = dataset.map(
            lambda a, b, c, d, e, f, g, h, i: (
                {
                    "input_ids": a,
                    "token_type_ids": b,
                    "attention_mask": c,
                    "pos_input_ids": d,
                    "pos_token_type_ids": e,
                    "pos_attention_mask": f,
                    "neg_input_ids": g,
                    "neg_token_type_ids": h,
                    "neg_attention_mask": i,
                },
                None,
            ),
            num_parallel_calls=num_parallel_calls,
        ).prefetch(utils.AUTOTUNE)
        return dataset

    def _fixed_padding(self, dataset: tf.data.Dataset, pad_id=0, max_sequence_length=512, **kwargs) -> tf.data.Dataset:
        if not self.with_positive_sequence and not self.with_negative_sequence:
            pad_id = tf.constant(pad_id, dtype=tf.int32)
            maxlen = tf.constant(max_sequence_length, dtype=tf.int32)
            # fmt: off
            padded_shapes = ([maxlen, ], [maxlen, ], [maxlen, ])
            padding_values = (pad_id, pad_id, pad_id)
            # fmt: on
            return utils.batching_and_padding(
                dataset, padded_shapes=padded_shapes, padding_values=padding_values, **kwargs
            )
        if not self.with_negative_sequence:
            pad_id = tf.constant(pad_id, dtype=tf.int32)
            maxlen = tf.constant(max_sequence_length, dtype=tf.int32)
            # fmt: off
            padded_shapes = ([maxlen, ], [maxlen, ], [maxlen, ], [maxlen, ], [maxlen, ], [maxlen, ])
            padding_values = (pad_id, pad_id, pad_id, pad_id, pad_id, pad_id)
            # fmt: on
            return utils.batching_and_padding(
                dataset, padded_shapes=padded_shapes, padding_values=padding_values, **kwargs
            )
        # only sequence
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        maxlen = tf.constant(max_sequence_length, dtype=tf.int32)
        # fmt: off
        padded_shapes = ([maxlen, ], [maxlen, ], [maxlen, ], [maxlen, ], [maxlen, ], [maxlen, ], [maxlen, ], [maxlen, ], [maxlen, ])
        padding_values = (pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id)
        # fmt: on
        return utils.batching_and_padding(dataset, padded_shapes=padded_shapes, padding_values=padding_values, **kwargs)

    def _batch_padding(self, dataset: tf.data.Dataset, pad_id=0, **kwargs) -> tf.data.Dataset:
        if not self.with_positive_sequence and not self.with_negative_sequence:
            pad_id = tf.constant(pad_id, dtype=tf.int32)
            # fmt: off
            padded_shapes = ([None, ], [None, ], [None, ])
            padding_values = (pad_id, pad_id, pad_id)
            # fmt: on
            return utils.batching_and_padding(
                dataset, padded_shapes=padded_shapes, padding_values=padding_values, **kwargs
            )
        if not self.with_negative_sequence:
            pad_id = tf.constant(pad_id, dtype=tf.int32)
            # fmt: off
            padded_shapes = ([None, ], [None, ], [None, ], [None, ], [None, ], [None, ])
            padding_values = (pad_id, pad_id, pad_id, pad_id, pad_id, pad_id)
            # fmt: on
            return utils.batching_and_padding(
                dataset, padded_shapes=padded_shapes, padding_values=padding_values, **kwargs
            )
        # triple inputs
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        padded_shapes = ([None, ], [None, ], [None, ], [None, ], [None, ], [None, ], [None, ], [None, ], [None, ])
        padding_values = (pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id)
        # fmt: on
        return utils.batching_and_padding(dataset, padded_shapes=padded_shapes, padding_values=padding_values, **kwargs)

    def _bucket_padding(self, dataset: tf.data.Dataset, pad_id=0, **kwargs) -> tf.data.Dataset:
        if not self.with_positive_sequence and not self.with_negative_sequence:
            pad_id = tf.constant(pad_id, dtype=tf.int32)
            # fmt: off
            padded_shapes = ([None, ], [None, ], [None, ])
            padding_values = (pad_id, pad_id, pad_id)
            # fmt: on
            return utils.bucketing_and_padding(
                dataset,
                bucket_fn=lambda a, b, c: tf.size(a),
                padded_shapes=padded_shapes,
                padding_values=padding_values,
                **kwargs,
            )
        if not self.with_negative_sequence:
            pad_id = tf.constant(pad_id, dtype=tf.int32)
            # fmt: off
            padded_shapes = ([None, ], [None, ], [None, ], [None, ], [None, ], [None, ])
            padding_values = (pad_id, pad_id, pad_id, pad_id, pad_id, pad_id)
            # fmt: on
            return utils.bucketing_and_padding(
                dataset,
                bucket_fn=lambda a, b, c, d, e, f: tf.size(a),
                padded_shapes=padded_shapes,
                padding_values=padding_values,
                **kwargs,
            )
        # triple inputs
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        padded_shapes = ([None, ], [None, ], [None, ], [None, ], [None, ], [None, ], [None, ], [None, ], [None, ])
        padding_values = (pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id)
        # fmt: on
        return utils.bucketing_and_padding(
            dataset,
            bucket_fn=lambda a, b, c, d, e, f, g, h, i: tf.size(a),
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            **kwargs,
        )
