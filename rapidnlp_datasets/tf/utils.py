import json
import logging
import os

import tensorflow as tf

try:
    AUTOTUNE = tf.data.AUTOTUNE
except Exception as e:
    AUTOTUNE = tf.data.experimental.AUTOTUNE


def read_tfrecord_files(input_files, num_parallel_calls=None, **kwargs):
    num_parallel_calls = num_parallel_calls or AUTOTUNE
    if isinstance(input_files, str):
        input_files = [input_files]
    if len(input_files) == 1:
        dataset = tf.data.TFRecordDataset(input_files)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(input_files)
        dataset = dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x),
            cycle_length=len(input_files),
            num_parallel_calls=num_parallel_calls,
        )
    return dataset


def batching_and_padding(dataset: tf.data.Dataset, padded_shapes, padding_values, **kwargs) -> tf.data.Dataset:
    dataset = dataset.padded_batch(
        batch_size=kwargs.get("batch_size", 32),
        padded_shapes=padded_shapes,
        padding_values=padding_values,
        drop_remainder=kwargs.get("drop_remainder", False),
    )
    return dataset


def bucketing_and_padding(
    dataset: tf.data.Dataset, bucket_fn, padded_shapes, padding_values, **kwargs
) -> tf.data.Dataset:
    batch_size = kwargs.get("batch_size", 32)
    bucket_boundaries = kwargs.get("bucket_boundaries", None)
    if not bucket_boundaries:
        bucket_boundaries = []
        maxlen = kwargs.get("max_sequence_length", 512)
        numbuk = kwargs.get("num_buckets", 8)
        step = maxlen // numbuk
        for i in range(1, numbuk + 1):
            v = i * step
            if v >= maxlen:
                break
            bucket_boundaries.append(v)

    bucket_batch_sizes = kwargs.get("bucket_batch_sizes", None)
    if not bucket_batch_sizes:
        bucket_batch_sizes = [batch_size] * (1 + len(bucket_boundaries))

    assert (
        len(bucket_batch_sizes) == len(bucket_boundaries) + 1
    ), "len(bucket_batch_sizes) != len(bucket_boundaries) + 1"

    try:
        fn = tf.data.bucket_by_sequence_length
    except Exception as e:
        fn = tf.data.experimental.bucket_by_sequence_length

    dataset = dataset.apply(
        fn(
            element_length_func=bucket_fn,
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            drop_remainder=kwargs.get("drop_remainder", False),
            pad_to_bucket_boundary=kwargs.get("pad_to_bucket_boundary", False),
            no_padding=kwargs.get("no_padding", False),
        )
    )
    return dataset


def read_jsonl_files(input_files, select_keys=None, ignore_keys=None, **kwargs):
    if isinstance(input_files, str):
        input_files = [input_files]
    for f in input_files:
        with open(f, mode="rt", encoding="utf-8") as fin:
            for line in fin:
                d = json.loads(line)
                if select_keys:
                    for k in list(d.keys()):
                        if k not in select_keys:
                            d.pop(k, None)
                if ignore_keys:
                    for k in list(d.keys()):
                        if k in ignore_keys:
                            d.pop(k, None)
                yield d


def int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def save_tfrecord(examples, fn, output_files, **kwargs):
    if not examples:
        logging.warning("Examples is empty or None, skipped.")
        return
    if isinstance(output_files, str):
        output_files = [output_files]
    writers = [tf.io.TFRecordWriter(f) for f in output_files]
    idx, c = 0, 0
    for example in examples:
        if not example:
            continue
        try:
            feature = fn(example, **kwargs)
            record = tf.train.Example(features=tf.train.Features(feature=feature))
            writers[idx].write(record.SerializeToString())
        except Exception as e:
            logging.warning("Encode feature exception: ", e)
            continue
        c += 1
        idx += 1
        idx = idx % len(writers)
    for w in writers:
        w.close()
    logging.info("Finished to write %d examples to tfrecords.", c)
