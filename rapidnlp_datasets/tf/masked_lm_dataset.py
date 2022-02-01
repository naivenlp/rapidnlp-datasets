import tensorflow as tf

from . import utils
from .dataset import TFDataset


class TFDatasetForMaksedLanguageModel(TFDataset):
    """Dataset for masked lm in TensorFlow"""

    def __init__(
        self,
        examples,
        max_predictions=20,
        input_ids="input_ids",
        token_type_ids="token_type_ids",
        attention_mask="attention_mask",
        masked_positions="masked_pos",
        masked_ids="masked_ids",
        **kwargs
    ) -> None:
        super().__init__(examples, **kwargs)
        self.max_predictions = max_predictions
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.masked_positions = masked_positions
        self.masked_ids = masked_ids

    def parse_examples_to_dataset(self):
        input_ids, token_type_ids, attention_mask = [], [], []
        masked_pos, masked_ids = [], []
        for x in self.examples:
            input_ids.append(x.input_ids)
            token_type_ids.append(x.token_type_ids)
            attention_mask.append(x.attention_mask)
            masked_pos.append(x.masked_pos)
            masked_ids.append(x.masked_ids)

        def _to_dataset(x, dtype=tf.int32):
            x = tf.ragged.constant(x, dtype=dtype)
            d = tf.data.Dataset.from_tensor_slices(x)
            d = d.map(lambda x: x)
            return d

        # conver examples to dataset
        dataset = tf.data.Dataset.zip(
            (
                _to_dataset(input_ids, dtype=tf.int32),
                _to_dataset(token_type_ids, dtype=tf.int32),
                _to_dataset(attention_mask, dtype=tf.int32),
                _to_dataset(masked_ids, dtype=tf.int32),
                _to_dataset(masked_pos, dtype=tf.int32),
            )
        )
        return dataset

    @classmethod
    def from_tfrecord_files(cls, input_files, max_predictions=20, **kwargs) -> tf.data.Dataset:
        d = cls(
            examples=None,
            max_predictions=max_predictions,
            input_ids=kwargs.pop("input_ids", "input_ids"),
            token_type_ids=kwargs.pop("token_type_ids", "token_type_ids"),
            attention_mask=kwargs.pop("attention_mask", "attention_mask"),
            masked_positions=kwargs.pop("masked_positions", "masked_positions"),
            masked_ids=kwargs.pop("masked_ids", "masked_ids"),
            **kwargs,
        )

        dataset = utils.read_tfrecord_files(input_files, **kwargs)
        # parse example
        num_parallel_calls = kwargs.get("num_parallel_calls", utils.AUTOTUNE)
        buffer_size = kwargs.get("buffer_size", utils.AUTOTUNE)
        features = {
            d.input_ids: tf.io.VarLenFeature(tf.int64),
            d.token_type_ids: tf.io.VarLenFeature(tf.int64),
            d.attention_mask: tf.io.VarLenFeature(tf.int64),
            d.masked_ids: tf.io.VarLenFeature(tf.int64),
            d.masked_positions: tf.io.VarLenFeature(tf.int64),
        }
        dataset = dataset.map(
            lambda x: tf.io.parse_example(x, features),
            num_parallel_calls=num_parallel_calls,
        ).prefetch(buffer_size)
        dataset = dataset.map(
            lambda x: (
                tf.cast(tf.sparse.to_dense(x[d.input_ids]), tf.int32),
                tf.cast(tf.sparse.to_dense(x[d.token_type_ids]), tf.int32),
                tf.cast(tf.sparse.to_dense(x[d.attention_mask]), tf.int32),
                tf.cast(tf.sparse.to_dense(x[d.masked_ids]), tf.int32),
                tf.cast(tf.sparse.to_dense(x[d.masked_positions]), tf.int32),
            ),
            num_parallel_calls=num_parallel_calls,
        ).prefetch(buffer_size)
        return d(dataset, **kwargs)

    def _filter(self, dataset: tf.data.Dataset, do_filter=True, max_sequence_length=512, **kwargs) -> tf.data.Dataset:
        if not do_filter:
            return dataset
        dataset = dataset.filter(lambda a, b, c, x, y: tf.size(a) <= max_sequence_length)
        return dataset

    def _to_dict(self, dataset: tf.data.Dataset, to_dict=True, **kwargs) -> tf.data.Dataset:
        num_parallel_calls = kwargs.get("num_parallel_calls", utils.AUTOTUNE)
        buffer_size = kwargs.get("buffer_size", utils.AUTOTUNE)
        if not to_dict:
            dataset = dataset.map(
                lambda a, b, c, x, y: ((a, b, c), (x, y)),
                num_parallel_calls=num_parallel_calls,
            ).prefetch(buffer_size)
            return dataset
        dataset = dataset.map(
            lambda a, b, c, x, y: (
                {self.input_ids: a, self.token_type_ids: b, self.attention_mask: c},
                {self.masked_ids: x, self.masked_positions: y},
            ),
            num_parallel_calls=num_parallel_calls,
        ).prefetch(buffer_size)
        return dataset

    def _fixed_padding(self, dataset: tf.data.Dataset, pad_id=0, max_sequence_length=512, **kwargs) -> tf.data.Dataset:
        maxlen = tf.constant(max_sequence_length, dtype=tf.int32)
        maxpred = tf.constant(self.max_predictions, dtype=tf.int32)
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        padded_shapes = kwargs.get("padded_shapes", ([maxlen, ], [maxlen, ], [maxlen, ], [maxpred, ], [maxpred]))
        padding_values = kwargs.get("padding_values", (pad_id, pad_id, pad_id, pad_id, pad_id))
        # fmt: on
        dataset = utils.batching_and_padding(dataset, padded_shapes, padding_values, **kwargs)
        return dataset

    def _batch_padding(self, dataset: tf.data.Dataset, pad_id=0, **kwargs) -> tf.data.Dataset:
        maxpred = tf.constant(self.max_predictions, dtype=tf.int32)
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        padded_shapes = kwargs.get("padded_shapes", ([None, ], [None, ], [None, ], [maxpred, ], [maxpred, ]))
        padding_values = kwargs.get("padding_values", (pad_id, pad_id, pad_id, pad_id, pad_id))
        # fmt: on
        dataset = utils.batching_and_padding(dataset, padded_shapes, padding_values, **kwargs)
        return dataset

    def _bucket_padding(self, dataset: tf.data.Dataset, pad_id=0, **kwargs) -> tf.data.Dataset:
        maxpred = tf.constant(self.max_predictions, dtype=tf.int32)
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        padded_shapes = kwargs.get("padded_shapes", ([None, ], [None, ], [None, ], [maxpred, ], [maxpred, ]))
        padding_values = kwargs.get("padding_values", (pad_id, pad_id, pad_id, pad_id, pad_id))
        # fmt: on
        dataset = utils.bucketing_and_padding(
            dataset,
            bucket_fn=lambda a, b, c, x, y: tf.size(a),
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            **kwargs,
        )
        return dataset
