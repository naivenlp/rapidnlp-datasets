import logging
from typing import List

import tensorflow as tf
from rapidnlp_datasets.simcse import (
    CsvFileReaderForSimCSE,
    ExampleForSimCSE,
    ExampleParserForSimCSE,
    JsonlFileReaderForSimCSE,
)

from . import utils
from .dataset import TFDataset


class TFDatasetForUnsupSimCSE(TFDataset):
    """Datapipe for unsupervised SimCSE"""

    def save_tfrecord(self, output_files, **kwargs):
        """Save examples to tf records"""

        def _encoding(example: ExampleForSimCSE):
            feature = {
                "input_ids": utils.int64_feature([int(x) for x in example.input_ids]),
                "segment_ids": utils.int64_feature([int(x) for x in example.segment_ids]),
                "attention_mask": utils.int64_feature([int(x) for x in example.attention_mask]),
            }
            return feature

        utils.save_tfrecord(self.examples, _encoding, output_files, **kwargs)

    @classmethod
    def from_tfrecord_files(cls, input_files, **kwargs) -> tf.data.Dataset:
        dataset = utils.read_tfrecord_files(input_files, **kwargs)
        num_parallel_calls = kwargs.get("num_parallel_calls", utils.AUTOTUNE)
        buffer_size = kwargs.get("buffer_size", utils.AUTOTUNE)
        # parse example
        features = {
            "input_ids": tf.io.VarLenFeature(tf.int64),
            "segment_ids": tf.io.VarLenFeature(tf.int64),
            "attention_mask": tf.io.VarLenFeature(tf.int64),
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
            ),
            num_parallel_calls=num_parallel_calls,
        ).prefetch(buffer_size)
        # do transformation
        d = cls(**kwargs)
        return d(dataset, **kwargs)

    @classmethod
    def from_jsonl_files(cls, input_files, vocab_file, **kwargs) -> tf.data.Dataset:
        reader = JsonlFileReaderForSimCSE(with_pos_sequence=False, with_neg_sequence=False, **kwargs)
        instances = reader.read_files(input_files, **kwargs)
        return cls.from_instances(instances, vocab_file, **kwargs)

    @classmethod
    def from_csv_files(cls, input_files, vocab_file, **kwargs):
        reader = CsvFileReaderForSimCSE(with_pos_sequence=False, with_neg_sequence=False, **kwargs)
        instances = reader.read_files(input_files, **kwargs)
        return cls.from_instances(instances, vocab_file, **kwargs)

    @classmethod
    def from_instances(cls, instances, vocab_file, add_special_tokens=True, **kwargs) -> tf.data.Dataset:
        parser = ExampleParserForSimCSE(vocab_file, with_pos_sequence=False, with_neg_sequence=False, **kwargs)
        examples = []
        for e in parser.parse_instances(instances, add_special_tokens=add_special_tokens, **kwargs):
            if not e:
                continue
            examples.append(e)
        return cls.from_examples(examples, **kwargs)

    @classmethod
    def from_examples(
        cls, examples: List[ExampleForSimCSE], return_self=False, verbose=True, **kwargs
    ) -> tf.data.Dataset:
        if not examples:
            logging.warning("examples is empty or null, skipped to build dataset.")
            return None
        if verbose:
            n = min(5, len(examples))
            for i in range(n):
                logging.info("Showing NO.%d example: %s", i, examples[i])

        def _to_dataset(x, dtype=tf.int32):
            x = tf.ragged.constant(x, dtype=dtype)
            d = tf.data.Dataset.from_tensor_slices(x)
            d = d.map(lambda x: x)
            return d

        dataset = tf.data.Dataset.zip(
            (
                _to_dataset([e.input_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.segment_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.attention_mask for e in examples], dtype=tf.int32),
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
        dataset = dataset.filter(lambda a, b, c: tf.size(a) <= max_sequence_length)
        return dataset

    def _to_dict(self, dataset: tf.data.Dataset, to_dict=True, **kwargs) -> tf.data.Dataset:
        num_parallel_calls = kwargs.get("num_parallel_calls", utils.AUTOTUNE)
        buffer_size = kwargs.get("buffer_size", utils.AUTOTUNE)
        if not to_dict:
            dataset = dataset.map(
                lambda a, b, c: ((a, b, c), None),
                num_parallel_calls=num_parallel_calls,
            ).prefetch(buffer_size)
            return dataset
        dataset = dataset.map(
            lambda x, y, z: ({"input_ids": x, "segment_ids": y, "attention_mask": z}, None),
            num_parallel_calls=num_parallel_calls,
        ).prefetch(buffer_size)
        return dataset

    def _fixed_padding(self, dataset: tf.data.Dataset, pad_id=0, max_sequence_length=512, **kwargs) -> tf.data.Dataset:
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        maxlen = tf.constant(max_sequence_length, dtype=tf.int32)
        # fmt: off
        padded_shapes = ([maxlen, ], [maxlen, ], [maxlen, ])
        padding_values = (pad_id, pad_id, pad_id)
        # fmt: on
        return utils.batching_and_padding(dataset, padded_shapes=padded_shapes, padding_values=padding_values, **kwargs)

    def _batch_padding(self, dataset: tf.data.Dataset, pad_id=0, **kwargs) -> tf.data.Dataset:
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        padded_shapes = ([None, ], [None, ], [None, ])
        padding_values = (pad_id, pad_id, pad_id)
        # fmt: on
        return utils.batching_and_padding(dataset, padded_shapes=padded_shapes, padding_values=padding_values, **kwargs)

    def _bucket_padding(self, dataset: tf.data.Dataset, pad_id=0, **kwargs) -> tf.data.Dataset:
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


class TFDatasetForSupervisedSimCSE(TFDataset):
    """Datapipe for unsupervised SimCSE"""

    def save_tfrecord(self, output_files, **kwargs):
        """Save examples in tfrecord format"""

        def _encoding(example: ExampleForSimCSE):
            feature = {
                "input_ids": utils.int64_feature([int(x) for x in example.input_ids]),
                "segment_ids": utils.int64_feature([int(x) for x in example.segment_ids]),
                "attention_mask": utils.int64_feature([int(x) for x in example.attention_mask]),
                "pos_input_ids": utils.int64_feature([int(x) for x in example.pos_input_ids]),
                "pos_segment_ids": utils.int64_feature([int(x) for x in example.pos_segment_ids]),
                "pos_attention_mask": utils.int64_feature([int(x) for x in example.pos_attention_mask]),
            }
            return feature

        utils.save_tfrecord(self.examples, _encoding, output_files, **kwargs)

    @classmethod
    def from_tfrecord_files(cls, input_files, **kwargs) -> tf.data.Dataset:
        dataset = utils.read_tfrecord_files(input_files, **kwargs)
        num_parallel_calls = kwargs.get("num_parallel_calls", utils.AUTOTUNE)
        buffer_size = kwargs.get("buffer_size", utils.AUTOTUNE)
        # parse example
        features = {
            "input_ids": tf.io.VarLenFeature(tf.int64),
            "segment_ids": tf.io.VarLenFeature(tf.int64),
            "attention_mask": tf.io.VarLenFeature(tf.int64),
            "pos_input_ids": tf.io.VarLenFeature(tf.int64),
            "pos_segment_ids": tf.io.VarLenFeature(tf.int64),
            "pos_attention_mask": tf.io.VarLenFeature(tf.int64),
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
                tf.cast(tf.sparse.to_dense(x["pos_input_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["pos_segment_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["pos_attention_mask"]), tf.int32),
            ),
            num_parallel_calls=num_parallel_calls,
        ).prefetch(buffer_size)
        # do transformation
        d = cls(**kwargs)
        return d(dataset, **kwargs)

    @classmethod
    def from_jsonl_files(cls, input_files, vocab_file, **kwargs) -> tf.data.Dataset:
        reader = JsonlFileReaderForSimCSE(with_pos_sequence=True, with_neg_sequence=False, **kwargs)
        instances = reader.read_files(input_files, **kwargs)
        return cls.from_instances(instances, vocab_file, **kwargs)

    @classmethod
    def from_csv_files(cls, input_files, vocab_file, **kwargs):
        reader = CsvFileReaderForSimCSE(with_pos_sequence=True, with_neg_sequence=False, **kwargs)
        instances = reader.read_files(input_files, **kwargs)
        return cls.from_instances(instances, vocab_file, **kwargs)

    @classmethod
    def from_instances(cls, instances, vocab_file, add_special_tokens=True, **kwargs) -> tf.data.Dataset:
        parser = ExampleParserForSimCSE(vocab_file, with_pos_sequence=True, with_neg_sequence=False, **kwargs)
        examples = []
        for e in parser.parse_instances(instances, add_special_tokens=add_special_tokens, **kwargs):
            if not e:
                continue
            examples.append(e)
        return cls.from_examples(examples, **kwargs)

    @classmethod
    def from_examples(
        cls, examples: List[ExampleForSimCSE], return_self=False, verbose=True, **kwargs
    ) -> tf.data.Dataset:
        if not examples:
            logging.warning("examples is empty or null, skipped to build dataset.")
            return None
        if verbose:
            n = min(5, len(examples))
            for i in range(n):
                logging.info("Showing NO.%d example: %s", i, examples[i])

        def _to_dataset(x, dtype=tf.int32):
            x = tf.ragged.constant(x, dtype=dtype)
            d = tf.data.Dataset.from_tensor_slices(x)
            d = d.map(lambda x: x)
            return d

        dataset = tf.data.Dataset.zip(
            (
                _to_dataset([e.input_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.segment_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.attention_mask for e in examples], dtype=tf.int32),
                _to_dataset([e.pos_input_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.pos_segment_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.pos_attention_mask for e in examples], dtype=tf.int32),
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
        dataset = dataset.filter(lambda a, b, c, d, e, f: tf.size(a) <= max_sequence_length)
        return dataset

    def _to_dict(self, dataset: tf.data.Dataset, to_dict=True, **kwargs) -> tf.data.Dataset:
        num_parallel_calls = kwargs.get("num_parallel_calls", utils.AUTOTUNE)
        buffer_size = kwargs.get("buffer_size", utils.AUTOTUNE)
        if not to_dict:
            dataset = dataset.map(
                lambda a, b, c, d, e, f: ((a, b, c, d, e, f), None),
                num_parallel_calls=num_parallel_calls,
            ).prefetch(buffer_size)
            return dataset
        dataset = dataset.map(
            lambda a, b, c, d, e, f: (
                {
                    "input_ids": a,
                    "segment_ids": b,
                    "attention_mask": c,
                    "pos_input_ids": d,
                    "pos_segment_ids": e,
                    "pos_attention_mask": f,
                },
                None,
            ),
            num_parallel_calls=num_parallel_calls,
        ).prefetch(buffer_size)
        return dataset

    def _fixed_padding(self, dataset: tf.data.Dataset, pad_id=0, max_sequence_length=512, **kwargs) -> tf.data.Dataset:
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        maxlen = tf.constant(max_sequence_length, dtype=tf.int32)
        # fmt: off
        padded_shapes = ([maxlen, ], [maxlen, ], [maxlen, ], [maxlen, ], [maxlen, ], [maxlen, ])
        padding_values = (pad_id, pad_id, pad_id, pad_id, pad_id, pad_id)
        # fmt: on
        return utils.batching_and_padding(dataset, padded_shapes=padded_shapes, padding_values=padding_values, **kwargs)

    def _batch_padding(self, dataset: tf.data.Dataset, pad_id=0, **kwargs) -> tf.data.Dataset:
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        padded_shapes = ([None, ], [None, ], [None, ], [None, ], [None, ], [None, ])
        padding_values = (pad_id, pad_id, pad_id, pad_id, pad_id, pad_id)
        # fmt: on
        return utils.batching_and_padding(dataset, padded_shapes=padded_shapes, padding_values=padding_values, **kwargs)

    def _bucket_padding(self, dataset: tf.data.Dataset, pad_id=0, **kwargs) -> tf.data.Dataset:
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


class TFDatasetForHardNegativeSimCSE(TFDataset):
    """Datapipe for hard negative SimCSE"""

    def save_tfrecord(self, output_files, **kwargs):
        """Save examples in tfrecord format"""

        def _encoding(example: ExampleForSimCSE):
            feature = {
                "input_ids": utils.int64_feature([int(x) for x in example.input_ids]),
                "segment_ids": utils.int64_feature([int(x) for x in example.segment_ids]),
                "attention_mask": utils.int64_feature([int(x) for x in example.attention_mask]),
                "pos_input_ids": utils.int64_feature([int(x) for x in example.pos_input_ids]),
                "pos_segment_ids": utils.int64_feature([int(x) for x in example.pos_segment_ids]),
                "pos_attention_mask": utils.int64_feature([int(x) for x in example.pos_attention_mask]),
                "neg_input_ids": utils.int64_feature([int(x) for x in example.neg_input_ids]),
                "neg_segment_ids": utils.int64_feature([int(x) for x in example.neg_segment_ids]),
                "neg_attention_mask": utils.int64_feature([int(x) for x in example.neg_attention_mask]),
            }
            return feature

        utils.save_tfrecord(self.examples, _encoding, output_files, **kwargs)

    @classmethod
    def from_tfrecord_files(cls, input_files, **kwargs) -> tf.data.Dataset:
        dataset = utils.read_tfrecord_files(input_files, **kwargs)
        num_parallel_calls = kwargs.get("num_parallel_calls", utils.AUTOTUNE)
        buffer_size = kwargs.get("buffer_size", utils.AUTOTUNE)
        # parse example
        features = {
            "input_ids": tf.io.VarLenFeature(tf.int64),
            "segment_ids": tf.io.VarLenFeature(tf.int64),
            "attention_mask": tf.io.VarLenFeature(tf.int64),
            "pos_input_ids": tf.io.VarLenFeature(tf.int64),
            "pos_segment_ids": tf.io.VarLenFeature(tf.int64),
            "pos_attention_mask": tf.io.VarLenFeature(tf.int64),
            "neg_input_ids": tf.io.VarLenFeature(tf.int64),
            "neg_segment_ids": tf.io.VarLenFeature(tf.int64),
            "neg_attention_mask": tf.io.VarLenFeature(tf.int64),
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
                tf.cast(tf.sparse.to_dense(x["pos_input_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["pos_segment_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["pos_attention_mask"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["neg_input_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["neg_segment_ids"]), tf.int32),
                tf.cast(tf.sparse.to_dense(x["neg_attention_mask"]), tf.int32),
            ),
            num_parallel_calls=num_parallel_calls,
        ).prefetch(buffer_size)
        # do transformation
        d = cls(**kwargs)
        return d(dataset, **kwargs)

    @classmethod
    def from_jsonl_files(cls, input_files, vocab_file, **kwargs) -> tf.data.Dataset:
        reader = JsonlFileReaderForSimCSE(with_pos_sequence=True, with_neg_sequence=True, **kwargs)
        instances = reader.read_files(input_files, **kwargs)
        return cls.from_instances(instances, vocab_file, **kwargs)

    @classmethod
    def from_csv_files(cls, input_files, vocab_file, **kwargs):
        reader = CsvFileReaderForSimCSE(with_pos_sequence=True, with_neg_sequence=True, **kwargs)
        instances = reader.read_files(input_files, **kwargs)
        return cls.from_instances(instances, vocab_file, **kwargs)

    @classmethod
    def from_instances(cls, instances, vocab_file, add_special_tokens=True, **kwargs) -> tf.data.Dataset:
        parser = ExampleParserForSimCSE(vocab_file, with_pos_sequence=True, with_neg_sequence=True, **kwargs)
        examples = []
        for e in parser.parse_instances(instances, add_special_tokens=add_special_tokens, **kwargs):
            if not e:
                continue
            examples.append(e)
        return cls.from_examples(examples, **kwargs)

    @classmethod
    def from_examples(
        cls, examples: List[ExampleForSimCSE], return_self=False, verbose=True, **kwargs
    ) -> tf.data.Dataset:
        if not examples:
            logging.warning("examples is empty or null, skipped to build dataset.")
            return None
        if verbose:
            n = min(5, len(examples))
            for i in range(n):
                logging.info("Showing NO.%d example: %s", i, examples[i])

        def _to_dataset(x, dtype=tf.int32):
            x = tf.ragged.constant(x, dtype=dtype)
            d = tf.data.Dataset.from_tensor_slices(x)
            d = d.map(lambda x: x)
            return d

        dataset = tf.data.Dataset.zip(
            (
                _to_dataset([e.input_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.segment_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.attention_mask for e in examples], dtype=tf.int32),
                _to_dataset([e.pos_input_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.pos_segment_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.pos_attention_mask for e in examples], dtype=tf.int32),
                _to_dataset([e.neg_input_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.neg_segment_ids for e in examples], dtype=tf.int32),
                _to_dataset([e.neg_attention_mask for e in examples], dtype=tf.int32),
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
        dataset = dataset.filter(lambda a, b, c, d, e, f, g, h, i: tf.size(a) <= max_sequence_length)
        return dataset

    def _to_dict(self, dataset: tf.data.Dataset, to_dict=True, **kwargs) -> tf.data.Dataset:
        num_parallel_calls = kwargs.get("num_parallel_calls", utils.AUTOTUNE)
        buffer_size = kwargs.get("buffer_size", utils.AUTOTUNE)
        if not to_dict:
            dataset = dataset.map(
                lambda a, b, c, d, e, f, g, h, i: ((a, b, c, d, e, f, g, h, i), None),
                num_parallel_calls=num_parallel_calls,
            ).prefetch(buffer_size)
            return dataset
        dataset = dataset.map(
            lambda a, b, c, d, e, f, g, h, i: (
                {
                    "input_ids": a,
                    "segment_ids": b,
                    "attention_mask": c,
                    "pos_input_ids": d,
                    "pos_segment_ids": e,
                    "pos_attention_mask": f,
                    "neg_input_ids": g,
                    "neg_segment_ids": h,
                    "neg_attention_mask": i,
                },
                None,
            ),
            num_parallel_calls=num_parallel_calls,
        ).prefetch(buffer_size)
        return dataset

    def _fixed_padding(self, dataset: tf.data.Dataset, pad_id=0, max_sequence_length=512, **kwargs) -> tf.data.Dataset:
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        maxlen = tf.constant(max_sequence_length, dtype=tf.int32)
        # fmt: off
        padded_shapes = ([maxlen, ], [maxlen, ], [maxlen, ], [maxlen, ], [maxlen, ], [maxlen, ], [maxlen, ], [maxlen, ], [maxlen, ])
        padding_values = (pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id)
        # fmt: on
        return utils.batching_and_padding(dataset, padded_shapes=padded_shapes, padding_values=padding_values, **kwargs)

    def _batch_padding(self, dataset: tf.data.Dataset, pad_id=0, **kwargs) -> tf.data.Dataset:
        pad_id = tf.constant(pad_id, dtype=tf.int32)
        # fmt: off
        padded_shapes = ([None, ], [None, ], [None, ], [None, ], [None, ], [None, ], [None, ], [None, ], [None, ])
        padding_values = (pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id)
        # fmt: on
        return utils.batching_and_padding(dataset, padded_shapes=padded_shapes, padding_values=padding_values, **kwargs)

    def _bucket_padding(self, dataset: tf.data.Dataset, pad_id=0, **kwargs) -> tf.data.Dataset:
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


class TFDatasetForSimCSE(TFDataset):
    """Dataset for SimCSE in TensorFlow"""

    def save_tfrecord(self, output_files, **kwargs):
        """save examples to tf records"""
        self.dataset.save_tfrecord(output_files, **kwargs)

    @classmethod
    def from_tfrecord_files(cls, input_files, with_pos_sequence=False, with_neg_sequence=False, **kwargs):
        if with_pos_sequence and with_neg_sequence:
            return TFDatasetForHardNegativeSimCSE.from_tfrecord_files(input_files, **kwargs)
        if with_pos_sequence:
            return TFDatasetForSupervisedSimCSE.from_tfrecord_files(input_files, **kwargs)
        return TFDatasetForUnsupSimCSE.from_tfrecord_files(input_files, **kwargs)

    @classmethod
    def from_jsonl_files(cls, input_files, vocab_file, with_pos_sequence=False, with_neg_sequence=False, **kwargs):
        if with_pos_sequence and with_neg_sequence:
            return TFDatasetForHardNegativeSimCSE.from_jsonl_files(input_files, vocab_file, **kwargs)
        if with_pos_sequence:
            return TFDatasetForSupervisedSimCSE.from_jsonl_files(input_files, vocab_file, **kwargs)
        return TFDatasetForUnsupSimCSE.from_jsonl_files(input_files, vocab_file, **kwargs)

    @classmethod
    def from_csv_files(
        cls, input_files, vocab_file, with_pos_sequence=False, with_neg_sequence=False, sep=",", **kwargs
    ):
        if with_pos_sequence and with_neg_sequence:
            return TFDatasetForHardNegativeSimCSE.from_csv_files(input_files, vocab_file, sep=sep, **kwargs)
        if with_pos_sequence:
            return TFDatasetForSupervisedSimCSE.from_csv_files(input_files, vocab_file, sep=sep, **kwargs)
        return TFDatasetForUnsupSimCSE.from_csv_files(input_files, vocab_file, sep=sep, **kwargs)

    @classmethod
    def from_instances(cls, instances, vocab_file, with_pos_sequence=False, with_neg_sequence=False, **kwargs):
        if with_pos_sequence and with_neg_sequence:
            return TFDatasetForHardNegativeSimCSE.from_instances(instances, vocab_file, **kwargs)
        if with_pos_sequence:
            return TFDatasetForSupervisedSimCSE.from_instances(instances, vocab_file, **kwargs)
        return TFDatasetForUnsupSimCSE.from_instances(instances, vocab_file, **kwargs)

    @classmethod
    def from_examples(
        cls,
        examples: List[ExampleForSimCSE],
        with_pos_sequence=False,
        with_neg_sequence=False,
        return_self=False,
        **kwargs
    ):
        if with_pos_sequence and with_neg_sequence:
            return TFDatasetForHardNegativeSimCSE.from_examples(examples, return_self=return_self, **kwargs)
        if with_pos_sequence:
            return TFDatasetForSupervisedSimCSE.from_examples(examples, return_self=return_self, **kwargs)
        return TFDatasetForUnsupSimCSE.from_examples(examples, return_self=return_self, **kwargs)

    def __init__(self, examples=None, with_pos_sequence=False, with_neg_sequence=False, **kwargs) -> None:
        super().__init__(examples=examples, **kwargs)
        self.with_pos_sequence = with_pos_sequence
        self.with_neg_sequence = with_neg_sequence
        if self.with_pos_sequence and self.with_neg_sequence:
            self.dataset = TFDatasetForHardNegativeSimCSE(examples=self.examples, **kwargs)
        elif self.with_pos_sequence:
            self.dataset = TFDatasetForSupervisedSimCSE(examples=self.examples, **kwargs)
        else:
            self.dataset = TFDatasetForUnsupSimCSE(examples=self.examples, **kwargs)

    def _filter(self, dataset: tf.data.Dataset, do_filter=True, max_sequence_length=512, **kwargs) -> tf.data.Dataset:
        return self.dataset._filter(dataset, do_filter=do_filter, max_sequence_length=max_sequence_length, **kwargs)

    def _to_dict(self, dataset: tf.data.Dataset, to_dict=True, **kwargs) -> tf.data.Dataset:
        return self.dataset._to_dict(dataset, to_dict=to_dict, **kwargs)

    def _fixed_padding(self, dataset: tf.data.Dataset, pad_id=0, max_sequence_length=512, **kwargs) -> tf.data.Dataset:
        return self.dataset._fixed_padding(dataset, pad_id=pad_id, max_sequence_length=max_sequence_length, **kwargs)

    def _batch_padding(self, dataset: tf.data.Dataset, pad_id=0, **kwargs) -> tf.data.Dataset:
        return self.dataset._batch_padding(dataset, pad_id=pad_id, **kwargs)

    def _bucket_padding(self, dataset: tf.data.Dataset, pad_id=0, **kwargs) -> tf.data.Dataset:
        return self.dataset._bucket_padding(dataset, pad_id=pad_id, **kwargs)
