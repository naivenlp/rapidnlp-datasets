import abc
import logging
import multiprocessing
from collections import namedtuple
from typing import List

from tokenizers import BertWordPieceTokenizer

from rapidnlp_datasets import readers

ExampleForSequenceClassification = namedtuple(
    "ExampleForSequenceClassification", ["input_ids", "token_type_ids", "attention_mask", "label"]
)


class AbcDatasetForSequenceClassificarion(abc.ABC):
    """Abstract dataset for sequence classification"""

    @abc.abstractmethod
    def to_pt_dataset(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def to_tf_dataset(self, **kwargs):
        raise NotImplementedError()


class DatasetForSequenceClassification(AbcDatasetForSequenceClassificarion):
    """Dataset for sequence classification tasks"""

    def __init__(
        self,
        tokenizer: BertWordPieceTokenizer,
        examples: List[ExampleForSequenceClassification] = None,
        max_sequence_length=512,
        **kwargs
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.examples = examples or []

        self.max_sequence_length = max_sequence_length

    def to_pt_dataset(self, **kwargs):
        from rapidnlp_datasets.pt.sequence_classification_dataset import PTDatasetForSequenceClassification

        dataset = PTDatasetForSequenceClassification(
            self.examples,
            max_sequence_length=self.max_sequence_length,
            input_ids=kwargs.pop("input_ids", "input_ids"),
            token_type_ids=kwargs.pop("token_type_ids", "token_type_ids"),
            attention_mask=kwargs.pop("attention_mask", "attention_mask"),
            label=kwargs.pop("label", "label"),
            **kwargs
        )
        return dataset

    def save_tfrecord(self, output_files, **kwargs):
        """Save examples to tfrecord"""
        input_ids = kwargs.pop("input_ids", "input_ids")
        token_type_ids = kwargs.pop("token_type_ids", "token_type_ids")
        attention_mask = kwargs.pop("attention_mask", "attention_mask")
        label = kwargs.pop("label", "label")

        from rapidnlp_datasets import utils_tf as utils

        def _encode(example):
            feature = {
                input_ids: utils.int64_feature([int(x) for x in example.input_ids]),
                token_type_ids: utils.int64_feature([int(x) for x in example.token_type_ids]),
                attention_mask: utils.int64_feature([int(x) for x in example.attention_mask]),
                label: utils.int64_feature([int(example.label)]),
            }
            return feature

        if not self.examples:
            logging.warning("self.examples is empty or None, skipped.")
            return
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
        from rapidnlp_datasets.tf.sequence_classification_dataset import TFDatasetForSequenceClassifiation

        d = TFDatasetForSequenceClassifiation(
            self.examples,
            input_ids=kwargs.pop("input_ids", "input_ids"),
            token_type_ids=kwargs.pop("token_type_ids", "token_type_ids"),
            attention_mask=kwargs.pop("attention_mask", "attention_mask"),
            label=kwargs.pop("label", "label"),
            **kwargs
        )
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

    def add_jsonl_files(
        self,
        input_files,
        sequence_column="sequence",
        pair_column=None,
        label_column="label",
        label2id_fn=None,
        **kwargs
    ):
        instances = []
        if label2id_fn is None:
            label2id_fn = lambda x: int(x)
            logging.info("label2id_fn is None, using default function.")
        for data in readers.read_jsonl_files(input_files, **kwargs):
            if not data:
                continue
            sequence = data[sequence_column].strip()
            label = str(data[label_column]).strip()
            if not sequence or not label:
                continue
            label_id = label2id_fn(label)
            instance = {"sequence": sequence, "label": label_id}
            if pair_column is not None:
                pair = data[pair_column].strip()
                if not pair:
                    continue
                instance["pair"] = pair
            instances.append(instance)
        return self.add_instances(instances, **kwargs)

    def add_csv_files(
        self, input_files, sequence_column=0, pair_column=None, label_column=-1, label2id_fn=None, sep=",", **kwargs
    ):
        return self.add_tsv_files(
            input_files,
            sequence_column=sequence_column,
            pair_column=pair_column,
            label_column=label_column,
            label2id_fn=label2id_fn,
            sep=sep,
            **kwargs
        )

    def add_tsv_files(
        self, input_files, sequence_column=0, pair_column=None, label_column=-1, label2id_fn=None, sep="\t", **kwargs
    ):
        instances = []
        if label2id_fn is None:
            label2id_fn = lambda x: int(x)
            logging.info("label2id_fn is None, using default function.")
        for parts in readers.read_tsv_files(input_files, sep=sep, **kwargs):
            if not parts:
                continue
            sequence = parts[sequence_column].strip()
            label = parts[label_column].strip()
            if not sequence or not label:
                continue
            instance = {"sequence": sequence, "label": label2id_fn(label)}
            if pair_column is not None:
                pair = parts[pair_column].strip()
                if not pair:
                    continue
                instance["pair"] = pair
            instances.append(instance)
        return self.add_instances(instances, **kwargs)

    def add_instances(self, instances, num_parallels=4, chunksize=1000, **kwargs):
        if num_parallels is None or num_parallels < 1:
            examples = self._parse_examples_sequential(instances, **kwargs)
            return self.add_examples(examples, **kwargs)
        # parallel mode
        pool = multiprocessing.Pool(num_parallels)
        futures = []
        for idx in range(0, len(instances), chunksize):
            chunk_instances = instances[idx : idx + chunksize]
            f = pool.apply_async(self._parse_examples_sequential, (chunk_instances,), **kwargs)
            futures.append(f)
        logging.info("Added %d tasks to process pool.", len(futures))
        pool.close()
        logging.info("Process pool closed, waiting for tasks to complete.")
        pool.join()
        examples = []
        for f in futures:
            examples.extend(f.get())
        return self.add_examples(examples, **kwargs)

    def add_examples(self, examples, **kwargs):
        valid_examples = []
        for e in examples:
            if len(e.input_ids) > self.max_sequence_length:
                continue
            valid_examples.append(e)
        self.examples.extend(valid_examples)
        logging.info("Added %d examples.", len(valid_examples))
        return self

    def _parse_examples_sequential(self, instances, **kwargs):
        examples = []
        for instance in instances:
            sequence, pair = instance["sequence"], instance.get("pair", None)
            encoding = self.tokenizer.encode(sequence, pair=pair, add_special_tokens=True)
            e = ExampleForSequenceClassification(
                input_ids=encoding.ids,
                token_type_ids=encoding.type_ids,
                attention_mask=encoding.attention_mask,
                label=instance["label"],
            )
            examples.append(e)
        return examples
