import abc
import multiprocessing
from collections import namedtuple
from typing import List, Union

from black import logging
from tokenizers import BertWordPieceTokenizer

from rapidnlp_datasets import readers

from .charlevel_tokenizer import BertCharLevelTokenizer

ExampleForTokenClassification = namedtuple(
    "ExampleForTokenClassification", ["input_ids", "token_type_ids", "attention_mask", "label_ids"]
)


class AbcDatasetForTokenClassification(abc.ABC):
    """Abstract dataset for token classification"""

    @abc.abstractmethod
    def to_tf_dataset(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def to_pt_dataset(self, **kwargs):
        raise NotImplementedError()


class DatasetForTokenClassification(AbcDatasetForTokenClassification):
    """Dataset for token classification"""

    def __init__(
        self,
        tokenizer: Union[BertWordPieceTokenizer, BertCharLevelTokenizer],
        examples: List[ExampleForTokenClassification] = None,
        max_sequence_length=512,
        **kwargs
    ) -> None:
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.examples = examples or []

    def to_pt_dataset(self, **kwargs):
        from rapidnlp_datasets.pt import PTDatasetForTokenClassification

        dataset = PTDatasetForTokenClassification(
            self.examples, pad_id=0, max_sequence_length=self.max_sequence_length, **kwargs
        )
        return dataset

    def save_tfrecord(self, output_files, **kwargs):
        """Save examples in tfrecord format"""
        input_ids = kwargs.pop("input_ids", "input_ids")
        token_type_ids = kwargs.pop("token_type_ids", "token_type_ids")
        attention_mask = kwargs.pop("attention_mask", "attention_mask")
        labels = kwargs.pop("labels", "labels")

        from rapidnlp_datasets import utils_tf as utils

        def _encode(example):
            feature = {
                input_ids: utils.int64_feature([int(x) for x in example.input_ids]),
                token_type_ids: utils.int64_feature([int(x) for x in example.token_type_ids]),
                attention_mask: utils.int64_feature([int(x) for x in example.attention_mask]),
                labels: utils.int64_feature([int(x) for x in example.label_ids]),
            }
            return feature

        utils.save_tfrecord(self.examples, _encode, output_files, **kwargs)

    def to_tf_dataset(
        self,
        batch_size=32,
        pad_id=0,
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
        auto_shard_policy=None,
        to_dict=True,
        **kwargs
    ):
        from rapidnlp_datasets.tf import TFDatasetForTokenClassification

        d = TFDatasetForTokenClassification(self.examples, **kwargs)
        dataset = d.parse_examples_to_dataset()
        dataset = d(
            dataset,
            batch_size=batch_size,
            pad_id=pad_id,
            max_sequence_length=self.max_sequence_length,
            padding=padding,
            num_buckets=num_buckets,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            drop_remainder=drop_remainder,
            do_filter=do_filter,
            do_repeat=do_repeat,
            repeat_count=repeat_count,
            do_shuffle=do_shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            shuffle_seed=shuffle_seed,
            reshuffle_each_iteration=reshuffle_each_iteration,
            to_dict=to_dict,
            auto_shard_policy=auto_shard_policy,
            **kwargs
        )
        return dataset

    def add_jsonl_files(self, input_files, label2id=None, feature_column="feature", label_column="label", **kwargs):
        instances = []
        for data in readers.read_jsonl_files(input_files, **kwargs):
            if not data:
                continue
            feature, label = data[feature_column], data[label_column]
            if len(feature) != len(label):
                continue
            instances.append({"feature": feature, "label": label})
        return self.add_instances(instances, label2id=label2id, **kwargs)

    def add_conll_files(self, input_files, label2id=None, feature_column=0, label_column=1, sep="\t", **kwargs):
        instances = []
        for instance in readers.read_conll_files(
            input_files, sep=sep, feature_column=feature_column, label_column=label_column, **kwargs
        ):
            if not instance:
                continue
            instances.append(instance)
        return self.add_instances(instances, label2id=label2id, **kwargs)

    def add_instances(self, instances, label2id=None, num_parallels=4, chunksize=1000, **kwargs):
        if num_parallels is None or num_parallels < 1:
            examples = self._parse_examples_sequential(instances, label2id=label2id, **kwargs)
            return self.add_examples(examples, **kwargs)
        # parallel mode
        pool = multiprocessing.Pool(num_parallels)
        futures = []
        for idx in range(0, len(instances), chunksize):
            chunk_instances = instances[idx : idx + chunksize]
            f = pool.apply_async(self._parse_examples_sequential, (chunk_instances, label2id), **kwargs)
            futures.append(f)
        logging.info("Added %d tasks to process pool.", len(futures))
        pool.close()
        logging.info("Process pool closed, waiting for tasks to complet.")
        pool.join()
        examples = []
        for f in futures:
            examples.extend(f.get())
        return self.add_examples(examples, **kwargs)

    def add_examples(self, examples: List[ExampleForTokenClassification], **kwargs):
        valid_examples = []
        for e in examples:
            if not e:
                continue
            if len(e.input_ids) > self.max_sequence_length:
                continue
            valid_examples.append(e)
        self.examples.extend(valid_examples)
        logging.info("Added %d examples.", len(valid_examples))
        return self

    def _parse_examples_sequential(self, instances, label2id=None, **kwargs):
        if label2id is None:
            label2id = lambda x: int(x)
        examples = []
        for instance in instances:
            feature_ids = [101] + [self.tokenizer.token_to_id(x) for x in instance["feature"]] + [102]
            label_ids = [101] + [label2id(x) for x in instance["label"]] + [102]
            assert len(feature_ids) == len(label_ids)
            e = ExampleForTokenClassification(
                input_ids=feature_ids,
                token_type_ids=[0] * len(feature_ids),
                attention_mask=[1] * len(feature_ids),
                label_ids=label_ids,
            )
            examples.append(e)
        return examples
