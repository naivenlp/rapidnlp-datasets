import abc
import logging
import multiprocessing
from copy import deepcopy
from typing import List

from tokenizers import BertWordPieceTokenizer

from rapidnlp_datasets import readers


class ExampleForSimCSE:
    """Example for SimCSE model"""

    def __init__(self, **kwargs) -> None:
        self.tokens = kwargs.get("tokens", None)
        self.input_ids = kwargs.get("input_ids", None)
        self.token_type_ids = kwargs.get("token_type_ids", None)
        self.attention_mask = kwargs.get("attention_mask", None)
        self.pos_tokens = kwargs.get("pos_tokens", None)
        self.pos_input_ids = kwargs.get("pos_input_ids", None)
        self.pos_token_type_ids = kwargs.get("pos_token_type_ids", None)
        self.pos_attention_mask = kwargs.get("pos_attention_mask", None)
        self.neg_tokens = kwargs.get("neg_tokens", None)
        self.neg_input_ids = kwargs.get("neg_input_ids", None)
        self.neg_token_type_ids = kwargs.get("neg_token_type_ids", None)
        self.neg_attention_mask = kwargs.get("neg_attention_mask", None)

    def _asdict(self):
        return deepcopy(self.__dict__)

    def __hash__(self) -> int:
        values = [getattr(self, k, None) for k in self.__dict__.keys()]
        return hash(values)

    def __eq__(self, __o: object) -> bool:
        if not __o:
            return False
        if not isinstance(__o, ExampleForSimCSE):
            return False
        for k, v in self.__dict__.items():
            if v != getattr(__o, k):
                return False
        return True

    def __repr__(self) -> str:
        builder = []
        for k, v in self.__dict__.items():
            builder.append("{}={}".format(k, v))
        return "ExampleForSimCSE[" + ",".join(builder) + "]"

    def __str__(self) -> str:
        return self.__repr__()


class AbcDatasetForSimCSE(abc.ABC):
    """Abstract dataset for SimCSE"""

    @abc.abstractmethod
    def to_pt_dataset(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def to_tf_dataset(self, **kwargs):
        raise NotImplementedError()


class DatasetForSimCSE(AbcDatasetForSimCSE):
    """Dataset for SimCSE"""

    def __init__(
        self,
        tokenizer: BertWordPieceTokenizer,
        examples: List[ExampleForSimCSE] = None,
        max_sequence_length=512,
        with_positive_sequence=False,
        with_negative_sequence=False,
        **kwargs
    ) -> None:
        super().__init__()
        self.examples = examples or []
        self.max_sequence_length = max_sequence_length
        self.with_positive_sequence = with_positive_sequence
        self.with_negative_sequence = with_negative_sequence

        self.tokenizer = tokenizer

    def to_pt_dataset(self, **kwargs):
        from rapidnlp_datasets.pt import PTDatasetForSimCSE

        dataset = PTDatasetForSimCSE(
            self.examples,
            max_sequence_length=self.max_sequence_length,
            with_positive_sequence=self.with_positive_sequence,
            with_negative_sequence=self.with_negative_sequence,
            **kwargs
        )
        return dataset

    def save_tfrecord(self, output_files, **kwargs):
        input_ids = kwargs.pop("input_ids", "input_ids")
        token_type_ids = kwargs.pop("token_type_ids", "token_type_ids")
        attention_mask = kwargs.pop("attention_mask", "attention_mask")
        pos_input_ids = kwargs.pop("pos_input_ids", "pos_input_ids")
        pos_token_type_ids = kwargs.pop("pos_token_type_ids", "pos_token_type_ids")
        pos_attention_mask = kwargs.pop("pos_attention_mask", "pos_attention_mask")
        neg_input_ids = kwargs.pop("neg_input_ids", "neg_input_ids")
        neg_token_type_ids = kwargs.pop("neg_token_type_ids", "neg_token_type_ids")
        neg_attention_mask = kwargs.pop("neg_attention_mask", "neg_attention_mask")

        from rapidnlp_datasets import utils_tf as utils

        def _encoding(example: ExampleForSimCSE):
            seq_feature = {
                input_ids: utils.int64_feature([int(x) for x in example.input_ids]),
                token_type_ids: utils.int64_feature([int(x) for x in example.token_type_ids]),
                attention_mask: utils.int64_feature([int(x) for x in example.attention_mask]),
            }
            pos_feature, neg_feature = {}, {}
            if self.with_positive_sequence:
                pos_feature = {
                    pos_input_ids: utils.int64_feature([int(x) for x in example.pos_input_ids]),
                    pos_token_type_ids: utils.int64_feature([int(x) for x in example.pos_token_type_ids]),
                    pos_attention_mask: utils.int64_feature([int(x) for x in example.pos_attention_mask]),
                }
            if self.with_negative_sequence:
                neg_feature = {
                    neg_input_ids: utils.int64_feature([int(x) for x in example.neg_input_ids]),
                    neg_token_type_ids: utils.int64_feature([int(x) for x in example.neg_token_type_ids]),
                    neg_attention_mask: utils.int64_feature([int(x) for x in example.neg_attention_mask]),
                }
            seq_feature.update(pos_feature)
            seq_feature.update(neg_feature)
            return seq_feature

        utils.save_tfrecord(self.examples, _encoding, output_files, **kwargs)

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
        from rapidnlp_datasets.tf import TFDatasetForSimCSE

        d = TFDatasetForSimCSE(
            self.examples,
            with_positive_sequence=self.with_positive_sequence,
            with_negative_sequence=self.with_negative_sequence,
            **kwargs
        )
        dataset = d.parse_examples_to_dataset()
        return d(
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

    def add_jsonl_files(
        self,
        input_files,
        sequence_column="sequence",
        positive_sequence_column=None,
        negative_sequence_column=None,
        **kwargs
    ):
        instances = []
        if self.with_positive_sequence:
            assert positive_sequence_column, "positive_sequence_column must be provided!"
        if self.with_negative_sequence:
            assert negative_sequence_column, "negative_sequence_column must be provided!"
        for data in readers.read_jsonl_files(input_files, **kwargs):
            if not data:
                continue
            sequence = data[sequence_column].strip()
            if not sequence:
                continue
            instance = {"sequence": sequence}
            if self.with_positive_sequence:
                pos_sequence = data[positive_sequence_column].strip()
                if not pos_sequence:
                    continue
                instance["pos_sequence"] = pos_sequence
            if self.with_negative_sequence:
                neg_sequence = data[negative_sequence_column].strip()
                if not neg_sequence:
                    continue
                instance["neg_sequence"] = neg_sequence
            instances.append(instance)
        return self.add_instances(instances, **kwargs)

    def add_csv_files(
        self,
        input_files,
        sequence_column=0,
        positive_sequence_column=None,
        negative_sequence_column=None,
        sep=",",
        skip_rows=0,
        **kwargs
    ):
        return self.add_tsv_files(
            input_files,
            sequence_column=sequence_column,
            positive_sequence_column=positive_sequence_column,
            negative_sequence_column=negative_sequence_column,
            sep=sep,
            skip_rows=skip_rows,
            **kwargs
        )

    def add_tsv_files(
        self,
        input_files,
        sequence_column=0,
        positive_sequence_column=None,
        negative_sequence_column=None,
        sep="\t",
        skip_rows=0,
        **kwargs
    ):
        instances = []
        if self.with_positive_sequence:
            assert positive_sequence_column, "positive_sequence_column must be provided!"
        if self.with_negative_sequence:
            assert negative_sequence_column, "negative_sequence_column must be provided!"
        for parts in readers.read_tsv_files(input_files, sep=sep, skip_rows=skip_rows, **kwargs):
            if not parts:
                continue
            sequence = parts[sequence_column].strip()
            if not sequence:
                continue
            instance = {"sequence": sequence}
            if self.with_positive_sequence:
                pos_sequence = parts[positive_sequence_column].strip()
                if not pos_sequence:
                    continue
                instance["pos_sequence"] = pos_sequence
            if self.with_negative_sequence:
                neg_sequence = parts[negative_sequence_column].strip()
                if not neg_sequence:
                    continue
                instance["neg_sequence"] = neg_sequence
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
            if not e:
                continue
            if len(e.input_ids) > self.max_sequence_length:
                continue
            if self.with_positive_sequence and len(e.pos_input_ids) > self.max_sequence_length:
                continue
            if self.with_negative_sequence and len(e.neg_input_ids) > self.max_sequence_length:
                continue
            valid_examples.append(e)
        self.examples.extend(valid_examples)
        logging.info("Added %d examples.", len(valid_examples))
        return self

    def _parse_examples_sequential(self, instances, **kwargs):
        examples = []
        for instance in instances:
            seq_encoding = self.tokenizer.encode(instance["sequence"], add_special_tokens=True)
            e = ExampleForSimCSE(
                tokens=seq_encoding.tokens,
                input_ids=seq_encoding.ids,
                token_type_ids=seq_encoding.type_ids,
                attention_mask=seq_encoding.attention_mask,
            )
            if self.with_positive_sequence:
                pos_encoding = self.tokenizer.encode(instance["pos_sequence"], add_special_tokens=True)
                e.pos_tokens = pos_encoding.tokens
                e.pos_input_ids = pos_encoding.ids
                e.pos_token_type_ids = pos_encoding.type_ids
                e.pos_attention_mask = pos_encoding.attention_mask
            if self.with_negative_sequence:
                neg_encoding = self.tokenizer.encode(instance["neg_sequence"], add_special_tokens=True)
                e.neg_tokens = neg_encoding.tokens
                e.neg_input_ids = neg_encoding.ids
                e.neg_token_type_ids = neg_encoding.type_ids
                e.neg_attention_mask = neg_encoding.attention_mask
            examples.append(e)
        return examples
