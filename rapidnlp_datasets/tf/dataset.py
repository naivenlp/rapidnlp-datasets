import abc

import tensorflow as tf


class TFDataset(abc.ABC):
    """Abstract dataset"""

    def __init__(self, examples=None, **kwargs) -> None:
        super().__init__()
        self.examples = examples

    def __call__(
        self,
        dataset: tf.data.Dataset,
        batch_size=32,
        pad_id=0,
        max_sequence_length=512,
        padding="bucket",
        num_buckets=8,
        bucket_boundaries=[64, 128, 192, 256, 320, 384, 448],
        bucket_batch_sizes=None,
        drop_remainder=False,
        do_filter=True,
        do_repeat=False,
        repeat_count=None,
        do_shuffle=True,
        shuffle_buffer_size=1000000,
        shuffle_seed=None,
        reshuffle_each_iteration=True,
        to_dict=True,
        auto_shard_policy=None,
        **kwargs,
    ) -> tf.data.Dataset:
        dataset = self._filter(dataset, do_filter=do_filter, max_sequence_length=max_sequence_length, **kwargs)
        dataset = self._repeat(dataset, do_repeat=do_repeat, repeat_count=repeat_count, **kwargs)
        dataset = self._shuffle(
            dataset,
            do_shuffle=do_shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            shuffle_seed=shuffle_seed,
            reshuffle_each_iteration=reshuffle_each_iteration,
            **kwargs,
        )
        dataset = self._padding(
            dataset,
            batch_size=batch_size,
            pad_id=pad_id,
            padding=padding,
            max_sequence_length=max_sequence_length,
            num_buckets=num_buckets,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            drop_remainder=drop_remainder,
            **kwargs,
        )
        dataset = self._to_dict(dataset, to_dict=to_dict, **kwargs)
        dataset = self._auto_shard(dataset, auto_shard_policy=auto_shard_policy, **kwargs)
        return dataset

    @abc.abstractmethod
    def _filter(self, dataset: tf.data.Dataset, do_filter=True, max_sequence_length=512, **kwargs) -> tf.data.Dataset:
        raise NotImplementedError()

    def _repeat(self, dataset: tf.data.Dataset, do_repeat=False, repeat_count=None, **kwargs) -> tf.data.Dataset:
        if not do_repeat:
            return dataset
        if repeat_count is None:
            return dataset
        return dataset.repeat(repeat_count)

    def _shuffle(
        self,
        dataset: tf.data.Dataset,
        do_shuffle=True,
        shuffle_buffer_size=100000,
        shuffle_seed=None,
        reshuffle_each_iteration=True,
        **kwargs,
    ) -> tf.data.Dataset:
        if not do_shuffle:
            return dataset
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer_size, seed=shuffle_seed, reshuffle_each_iteration=reshuffle_each_iteration
        )
        return dataset

    def _padding(
        self,
        dataset: tf.data.Dataset,
        batch_size=32,
        pad_id=0,
        max_sequence_length=512,
        padding="bucket",
        bucket_boundaries=[64, 128, 192, 256, 320, 384, 448],
        bucket_batch_sizes=None,
        num_buckets=8,
        drop_remainder=False,
        **kwargs,
    ) -> tf.data.Dataset:
        if padding == "fixed":
            return self._fixed_padding(
                dataset,
                batch_size=batch_size,
                pad_id=pad_id,
                max_sequence_length=max_sequence_length,
                drop_remainder=drop_remainder,
                **kwargs,
            )
        if padding == "batch":
            return self._batch_padding(
                dataset,
                batch_size=batch_size,
                pad_id=pad_id,
                drop_remainder=drop_remainder,
                **kwargs,
            )
        return self._bucket_padding(
            dataset,
            batch_size=batch_size,
            pad_id=pad_id,
            max_sequence_length=max_sequence_length,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            num_buckets=num_buckets,
            drop_remainder=drop_remainder,
            **kwargs,
        )

    @abc.abstractmethod
    def _fixed_padding(
        self, dataset: tf.data.Dataset, batch_size=32, pad_id=0, max_sequence_length=512, drop_remainder=False, **kwargs
    ) -> tf.data.Dataset:
        raise NotImplementedError("Fixed padding not supported yet!")

    @abc.abstractmethod
    def _batch_padding(
        self, dataset: tf.data.Dataset, batch_size=32, pad_id=0, drop_remainder=False, **kwargs
    ) -> tf.data.Dataset:
        raise NotImplementedError("Batch padding is not supported yet!")

    @abc.abstractmethod
    def _bucket_padding(
        self,
        dataset: tf.data.Dataset,
        batch_size=32,
        pad_id=0,
        max_sequence_length=512,
        bucket_boundaries=[64, 128, 192, 256, 320, 384, 448],
        bucket_batch_sizes=None,
        num_buckets=8,
        drop_remainder=False,
        **kwargs,
    ) -> tf.data.Dataset:
        raise NotImplementedError("Bucket padding is not supported yet!")

    @abc.abstractmethod
    def _to_dict(self, dataset: tf.data.Dataset, to_dict=True, **kwargs) -> tf.data.Dataset:
        raise NotImplementedError()

    def _auto_shard(self, dataset: tf.data.Dataset, auto_shard_policy=None, **kwargs) -> tf.data.Dataset:
        if auto_shard_policy is not None:
            options = tf.data.Options()
            # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            options.experimental_distribute.auto_shard_policy = auto_shard_policy
            dataset = dataset.with_options(options)
        return dataset
