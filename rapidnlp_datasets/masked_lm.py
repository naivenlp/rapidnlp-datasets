import abc
import logging
import multiprocessing
import random
from collections import namedtuple
from typing import List

from tokenizers import BertWordPieceTokenizer

from rapidnlp_datasets import readers

ExampleForMaskedLanguageModel = namedtuple(
    "ExampleForMaskedLanguageModel", ["input_ids", "token_type_ids", "attention_mask", "masked_ids", "masked_pos"]
)


class AbcMaskingForLanguageModel(abc.ABC):
    """Abstract masking strategy"""

    @abc.abstractmethod
    def __call__(self, tokens, **kwargs):
        raise NotImplementedError()


class WholeWordMaskingForLanguageModel(AbcMaskingForLanguageModel):
    """Default masking strategy from BERT."""

    def __init__(
        self, vocabs, change_prob=0.15, mask_prob=0.8, rand_prob=0.1, keep_prob=0.1, max_predictions=20, **kwargs
    ):
        self.vocabs = vocabs
        self.change_prob = change_prob
        self.mask_prob = mask_prob / (mask_prob + rand_prob + keep_prob)
        self.rand_prob = rand_prob / (mask_prob + rand_prob + keep_prob)
        self.keep_prob = keep_prob / (mask_prob + rand_prob + keep_prob)
        self.max_predictions = max_predictions
        # special tokens
        self.special_tokens = set(["[PAD]", "[CLS]", "[SEP]", "[MASK]"])

    def __call__(self, tokens, **kwargs):
        if not tokens:
            return None
        num_to_predict = min(self.max_predictions, max(1, round(self.change_prob * len(tokens))))
        cand_indexes = self._collect_candidates(tokens)
        # copy original tokens
        masked_tokens = [x for x in tokens]
        masked_indexes = [0] * len(tokens)
        for piece_indexes in cand_indexes:
            if sum(masked_indexes) >= num_to_predict:
                break
            if sum(masked_indexes) + len(piece_indexes) > num_to_predict:
                continue
            if any(masked_indexes[idx] == 1 for idx in piece_indexes):
                continue
            for index in piece_indexes:
                masked_indexes[index] = 1
                masked_tokens[index] = self._masking_tokens(index, tokens, self.vocabs)

        # add special tokens
        assert len(tokens) == len(masked_tokens) == len(masked_indexes)
        return {"tokens": tokens, "masked_tokens": masked_tokens, "masked_indexes": masked_indexes}

    def _masking_tokens(self, index, tokens, vocabs, **kwargs):
        # 80% of the time, replace with [MASK]
        if random.random() < self.mask_prob:
            return "[MASK]"
        # 10% of the time, keep original
        p = self.rand_prob / (self.rand_prob + self.keep_prob)
        if random.random() < p:
            return tokens[index]
        # 10% of the time, replace with random word
        masked_token = vocabs[random.randint(0, len(vocabs) - 1)]
        return masked_token

    def _collect_candidates(self, tokens):
        cand_indexes = [[]]
        for idx, token in enumerate(tokens):
            if token in self.special_tokens:
                continue
            if cand_indexes and token.startswith("##"):
                cand_indexes[-1].append(idx)
                continue
            cand_indexes.append([idx])
        random.shuffle(cand_indexes)
        return cand_indexes


class AbcDatasetForMaskedLanguageModel(abc.ABC):
    """Abstract dataset for MLM"""

    @abc.abstractmethod
    def to_pt_dataset(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def to_tf_dataset(self):
        raise NotImplementedError()


class DatasetForMaskedLanguageModel(AbcDatasetForMaskedLanguageModel):
    """Dataset for MLM"""

    def __init__(
        self,
        tokenizer: BertWordPieceTokenizer,
        examples: List[ExampleForMaskedLanguageModel] = None,
        max_sequence_length=512,
        change_prob=0.15,
        mask_prob=0.8,
        rand_prob=0.1,
        keep_prob=0.1,
        max_predictions=20,
        **kwargs
    ) -> None:
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.masking = WholeWordMaskingForLanguageModel(
            vocabs=list(self.tokenizer.get_vocab().keys()),
            change_prob=change_prob,
            mask_prob=mask_prob,
            rand_prob=rand_prob,
            keep_prob=keep_prob,
            max_predictions=max_predictions,
            **kwargs
        )

        self.examples = examples or []

    def to_pt_dataset(self, **kwargs):
        from rapidnlp_datasets.pt.masked_lm_dataset import PTDatasetForMaskedLanguageModel

        dataset = PTDatasetForMaskedLanguageModel(
            self.examples,
            max_sequence_length=self.max_sequence_length,
            input_ids=kwargs.pop("input_ids", "input_ids"),
            token_type_ids=kwargs.pop("token_type_ids", "token_type_ids"),
            attention_mask=kwargs.pop("attention_mask", "attention_mask"),
            labels=kwargs.pop("labels", "labels"),
            **kwargs
        )
        return dataset

    def save_tfrecord(self, output_files, **kwargs):
        """Save examples in tfrecord format"""
        input_ids = kwargs.pop("input_ids", "input_ids")
        token_type_ids = kwargs.pop("token_type_ids", "token_type_ids")
        attention_mask = kwargs.pop("attention_mask", "attention_mask")
        masked_positions = kwargs.pop("masked_positions", "masked_positions")
        masked_ids = kwargs.pop("masked_ids", "masked_ids")

        from rapidnlp_datasets import utils_tf as utils

        def _encode(example: ExampleForMaskedLanguageModel):
            feature = {
                input_ids: utils.int64_feature([int(x) for x in example.input_ids]),
                token_type_ids: utils.int64_feature([int(x) for x in example.token_type_ids]),
                attention_mask: utils.int64_feature([int(x) for x in example.attention_mask]),
                masked_ids: utils.int64_feature([int(x) for x in example.masked_ids]),
                masked_positions: utils.int64_feature([int(x) for x in example.masked_pos]),
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
        to_dict=True,
        auto_shard_policy=None,
        **kwargs
    ):
        from rapidnlp_datasets.tf.masked_lm_dataset import TFDatasetForMaksedLanguageModel

        d = TFDatasetForMaksedLanguageModel(
            self.examples,
            max_predictions=self.masking.max_predictions,
            input_ids=kwargs.pop("input_ids", "input_ids"),
            token_type_ids=kwargs.pop("token_type_ids", "token_type_ids"),
            attention_mask=kwargs.pop("attention_mask", "attention_mask"),
            masked_positions=kwargs.pop("masked_positions", "masked_pos"),
            masked_ids=kwargs.pop("masked_ids", "masked_ids"),
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

    def add_jsonl_files(self, input_files, sequence_column="sequence", **kwargs):
        instances = []
        for data in readers.read_jsonl_files(input_files, **kwargs):
            if not data:
                continue
            sequence = data[sequence_column].strip()
            if not sequence:
                continue
            instances.append({"sequence": sequence})
        return self.add_instances(instances, **kwargs)

    def add_text_files(self, input_files, **kwargs):
        instances = []
        for line in readers.read_text_files(input_files, **kwargs):
            if not line:
                continue
            instances.append({"sequence": line})
        return self.add_instances(instances, **kwargs)

    def add_csv_files(self, input_files, sequence_column=0, sep=",", **kwargs):
        instances = []
        for parts in readers.read_csv_files(input_files, sep=sep, **kwargs):
            if not parts:
                continue
            sequence = parts[sequence_column].strip()
            if not sequence:
                continue
            instances.append({"sequence": sequence})
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
            encoding = self.tokenizer.encode(instance["sequence"], add_special_tokens=True)
            outputs = self.masking(encoding.tokens, **kwargs)
            input_ids = [self.tokenizer.token_to_id(x) for x in outputs["masked_tokens"]]
            token_type_ids, attention_mask = [0] * len(input_ids), [1] * len(input_ids)
            masked_ids, masked_pos = [], []
            for idx, masked in enumerate(outputs["masked_indexes"]):
                if masked:
                    masked_pos.append(idx)
                    masked_ids.append(outputs["tokens"][idx])
            masked_ids = [self.tokenizer.token_to_id(x) for x in masked_ids]
            e = ExampleForMaskedLanguageModel(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                masked_ids=masked_ids,
                masked_pos=masked_pos,
            )
            examples.append(e)
        return examples
