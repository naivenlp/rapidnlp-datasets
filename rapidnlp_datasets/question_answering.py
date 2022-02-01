import abc
import logging
import re
from collections import namedtuple
from typing import List, Union

from tokenizers import BertWordPieceTokenizer

from rapidnlp_datasets import readers
from rapidnlp_datasets.charlevel_tokenizer import BertCharLevelTokenizer

from . import dureader

ExampleForQuestionAnswering = namedtuple(
    "ExampleForQuestionAnswering", ["input_ids", "token_type_ids", "attention_mask", "start_positions", "end_positions"]
)


class AbcDatasetForQuestionAnswering(abc.ABC):
    """Abstract dataset for QA"""

    @abc.abstractmethod
    def to_tf_dataset(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def to_pt_dataset(self, **kwargs):
        raise NotImplementedError()


class DatasetForQuestionAnswering(AbcDatasetForQuestionAnswering):
    """Dataset for Question Answering task"""

    def __init__(
        self,
        tokenizer: Union[BertWordPieceTokenizer, BertCharLevelTokenizer],
        examples: List[ExampleForQuestionAnswering] = None,
        max_sequence_length=512,
        sep_token="[SEP]",
        **kwargs
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.sep_token = sep_token

        self.examples = examples or []

    def to_pt_dataset(self, **kwargs):
        from rapidnlp_datasets.pt.question_answering_dataset import PTDatasetForQuestionAnswering

        dataset = PTDatasetForQuestionAnswering(
            self.examples,
            max_sequence_length=self.max_sequence_length,
            input_ids=kwargs.pop("input_ids", "input_ids"),
            token_type_ids=kwargs.pop("token_type_ids", "token_type_ids"),
            attention_mask=kwargs.pop("attention_mask", "attention_mask"),
            start_positions=kwargs.pop("start_positions", "start_positions"),
            end_positions=kwargs.pop("end_positions", "end_positions"),
        )
        return dataset

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
        from rapidnlp_datasets.tf.question_answering_dataset import TFDatasetForQuestionAnswering

        d = TFDatasetForQuestionAnswering(
            self.examples,
            input_ids=kwargs.pop("input_ids", "input_ids"),
            token_type_ids=kwargs.pop("token_type_ids", "token_type_ids"),
            attention_mask=kwargs.pop("attention_mask", "attention_mask"),
            start_positions=kwargs.pop("start_positions", "start_positions"),
            end_positions=kwargs.pop("end_positions", "end_positions"),
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

    def save_tfrecord(self, output_files, **kwargs):
        """Save examples to tfrecord"""

        input_ids = kwargs.pop("input_ids", "input_ids")
        token_type_ids = kwargs.pop("token_type_ids", "token_type_ids")
        attention_mask = kwargs.pop("attention_mask", "attention_mask")
        start_positions = kwargs.pop("start_positions", "start_positions")
        end_positions = kwargs.pop("end_positions", "end_positions")

        from rapidnlp_datasets import utils_tf as utils

        def _encoding(example):
            feature = {
                input_ids: utils.int64_feature([int(x) for x in example.input_ids]),
                token_type_ids: utils.int64_feature([int(x) for x in example.token_type_ids]),
                attention_mask: utils.int64_feature([int(x) for x in example.attention_mask]),
                start_positions: utils.int64_feature([int(example.start_positions)]),
                end_positions: utils.int64_feature([int(example.end_positions)]),
            }
            return feature

        if not self.examples:
            logging.warning("self.examples is empty or None, skippped")
            return
        utils.save_tfrecord(self.examples, _encoding, output_files, **kwargs)

    def add_dureader_robust(self, input_files, **kwargs):
        instances = []
        for instance in dureader.read_dureader_robust(input_files, **kwargs):
            if not instance:
                continue
            instances.append(instance)
        return self.add_instances(instance, **kwargs)

    def add_dureader_checklist(self, input_files, **kwargs):
        instances = []
        for instance in dureader.read_dureader_checklist(input_files, **kwargs):
            if not instance:
                continue
            instances.append(instance)
        return self.add_instances(instance, **kwargs)

    def add_csv_files(
        self,
        input_files,
        context_column=0,
        question_column=1,
        answer_column=2,
        sep=",",
        answer_sep=None,
        answer_start_column=None,
        anserr_end_column=None,
        **kwargs
    ):
        instances = []
        for parts in readers.read_tsv_files(input_files, sep=sep, **kwargs):
            # columns validation
            columns = [context_column, question_column, answer_column]
            if answer_start_column is not None:
                columns.append(answer_start_column)
            if len(parts) < max(columns):
                continue
            # process context, question and answer
            context, question, answer = parts[context_column], parts[question_column], parts[answer_column]
            context, question, answer = context.strip(), question.strip(), answer.strip()
            if not context or not question or not answer:
                continue
            # in case multi answers, take the first one
            if answer_sep is not None:
                answer = answer.split(answer_sep)[0]
            instance = {"context": context, "question": question, "answer": answer}
            # in case the answer start position is provided
            if answer_start_column is not None:
                answer_start = int(parts[answer_start_column])
                answer_end = answer_start + len(answer)
                if anserr_end_column is not None:
                    answer_end = int(anserr_end_column)
                instance.update({"answer_start": answer_start, "answer_end": answer_end})
            instances.append(instance)
        return self.add_instances(instances, **kwargs)

    def add_jsonl_files(
        self,
        input_files,
        context_column="context",
        question_column="question",
        answer_column="answer",
        answer_start_column=None,
        answer_end_column=None,
        **kwargs
    ):
        instances = []
        for data in readers.read_jsonl_files(input_files, **kwargs):
            if any(x not in data for x in [context_column, question_column, answer_column]):
                continue
            context, question, answer = data[context_column], data[question_column], data[answer_column]
            context, question = context.strip(), question.strip()
            # in case multi answers, take the first one
            if isinstance(answer, list):
                answer = answer[0]
            answer = answer.strip()
            if not context or not question or not answer:
                continue
            instance = {"context": context, "question": question, "answer": answer}
            if answer_start_column is not None:
                answer_start = int(data[answer_start_column])
                answer_end = answer_start + len(answer)
                if answer_end_column is not None:
                    answer_end = int(data[answer_end_column])
                instance.update({"answer_start": answer_start, "answer_end": answer_end})
            instances.append(instance)
        return self.add_instances(instances, **kwargs)

    def add_instances(self, instances, num_parallels=4, chunksize=1000, **kwargs):
        """Parse json instances to examples"""
        if num_parallels is None or num_parallels <= 1:
            examples = self._parse_examples_sequential(instances, **kwargs)
            return self.add_examples(examples, **kwargs)
        else:
            from multiprocessing import Pool

            pool = Pool(num_parallels)
            futures = []
            for idx in range(0, len(instances), chunksize):
                chunk_instances = instances[idx : idx + chunksize]
                f = pool.apply_async(self._parse_examples_sequential, (chunk_instances,), **kwargs)
                futures.append(f)
            logging.info("Added %d tasks to process pool.", len(futures))
            pool.close()
            logging.info("Process pool closed, waiting tasks to compelet.")
            pool.join()
            # collect examples
            examples = []
            for f in futures:
                examples.extend(f.get())
            return self.add_examples(examples, **kwargs)

    def add_examples(self, examples: List[ExampleForQuestionAnswering], **kwargs):
        padded_examples = []
        for e in examples:
            # filter
            if len(e.input_ids) > self.max_sequence_length:
                continue
            # padding
            # while e.input_ids < self.max_sequence_length:
            #     e.input_ids.append(0)
            #     e.token_type_ids.append(0)
            #     e.attention_mask.append(0)
            padded_examples.append(e)
        self.examples.extend(padded_examples)
        logging.info("Added %d examples.", len(padded_examples))
        return self

    def _parse_examples_sequential(self, instances, **kwargs):
        examples = []
        for instance in instances:
            answer_start, answer_end = instance.get("answer_start", None), instance.get("answer_end", None)
            if answer_start is None or answer_end is None:
                answer_start, answer_end = 0, 0
                for m in re.finditer(re.escape(instance["answer"]), instance["context"], re.IGNORECASE):
                    answer_start, answer_end = m.span()
                    break
            d = self._parse_example(instance, answer_start, answer_end, **kwargs)
            if not d:
                continue
            examples.append(ExampleForQuestionAnswering(**d))
        return examples

    def _parse_example(self, instance, answer_start, answer_end, **kwargs):
        context, question, answer = instance["context"], instance["question"], instance["answer"]
        encoding = self.tokenizer.encode(context, question, add_special_tokens=True)
        is_char_in_ans = [0] * len(context)
        for idx in range(answer_start, answer_end):
            is_char_in_ans[idx] = 1
        ans_token_index = []
        for idx, (start_char_index, end_char_index) in enumerate(encoding.offsets):
            if encoding.tokens[idx] == self.sep_token:
                break
            if sum(is_char_in_ans[start_char_index:end_char_index]) > 0:
                ans_token_index.append(idx)
        if not ans_token_index:
            logging.debug("No answering found. Skipped.")
            return None

        ans_token_start, ans_token_end = ans_token_index[0], ans_token_index[-1]

        def _valid_example(answer, tokens, start, end):
            sub_tokens = tokens[start:end]
            ans_tokens = self.tokenizer.encode(answer, add_special_tokens=False).tokens
            if ans_tokens != sub_tokens:
                logging.debug("Invalid example. ans_tokens: %s\t sub_tokens: %s", ans_tokens, sub_tokens)
                return False
            return True

        valid = _valid_example(answer, encoding.tokens, ans_token_start, ans_token_end + 1)
        if not valid:
            return None
        return {
            "input_ids": encoding.ids,
            "token_type_ids": encoding.type_ids,
            "attention_mask": encoding.attention_mask,
            "start_positions": ans_token_start,
            "end_positions": ans_token_end,
        }
