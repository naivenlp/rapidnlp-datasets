import abc

import tensorflow as tf

try:
    AUTOTUNE = tf.data.AUTOTUNE
except:
    AUTOTUNE = tf.data.experimental.AUTOTUNE


class AbcDataset(abc.ABC):
    """Abstract dataset"""

    def __call__(self, dataset: tf.data.Dataset, **kwargs) -> tf.data.Dataset:
        dataset = self._filter(dataset, **kwargs)
        dataset = self._repeat(dataset, **kwargs)
        dataset = self._shuffle(dataset, **kwargs)
        dataset = self._padding(dataset, **kwargs)
        dataset = self._to_dict(dataset, **kwargs)
        dataset = self._auto_shard(dataset, **kwargs)
        return dataset

    @abc.abstractmethod
    def _filter(self, dataset: tf.data.Dataset, **kwargs) -> tf.data.Dataset:
        raise NotImplementedError()

    def _repeat(self, dataset: tf.data.Dataset, repeat=None, **kwargs) -> tf.data.Dataset:
        if repeat is None:
            return dataset
        return dataset.repeat(repeat)

    def _shuffle(
        self,
        dataset: tf.data.Dataset,
        shuffle=True,
        buffer_size=100000,
        seed=None,
        reshuffle_each_iteration=True,
        **kwargs,
    ) -> tf.data.Dataset:
        if not shuffle:
            return dataset
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)
        return dataset

    def _padding(self, dataset: tf.data.Dataset, padding_strategy="bucket", **kwargs) -> tf.data.Dataset:
        if padding_strategy == "fixed":
            return self._fixed_padding(dataset, **kwargs)
        if padding_strategy == "batch":
            return self._batch_padding(dataset, **kwargs)
        return self._bucket_padding(dataset, **kwargs)

    @abc.abstractmethod
    def _fixed_padding(self, dataset: tf.data.Dataset, **kwargs) -> tf.data.Dataset:
        raise NotImplementedError("Fixed padding not supported yet!")

    @abc.abstractmethod
    def _batch_padding(self, dataset: tf.data.Dataset, **kwargs) -> tf.data.Dataset:
        raise NotImplementedError("Batch padding is not supported yet!")

    @abc.abstractmethod
    def _bucket_padding(self, dataset: tf.data.Dataset, **kwargs) -> tf.data.Dataset:
        raise NotImplementedError("Bucket padding is not supported yet!")

    @abc.abstractmethod
    def _to_dict(self, dataset: tf.data.Dataset, **kwargs) -> tf.data.Dataset:
        raise NotImplementedError()

    def _auto_shard(self, dataset: tf.data.Dataset, auto_shard_policy=None, **kwargs) -> tf.data.Dataset:
        if auto_shard_policy is not None:
            options = tf.data.Options()
            # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            options.experimental_distribute.auto_shard_policy = auto_shard_policy
            dataset = dataset.with_options(options)
        return dataset
