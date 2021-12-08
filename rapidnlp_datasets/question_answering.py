import json
import logging
import re
from collections import namedtuple
from typing import Iterator, List

from tokenizers import BertWordPieceTokenizer

from rapidnlp_datasets.readers import JsonlFileReader

from .charlevel_tokenizer import BertCharLevelTokenizer

ExampleForQuestionAnswering = namedtuple(
    "ExampleForQuestionAnswering",
    ["tokens", "input_ids", "segment_ids", "attention_mask", "start_positions", "end_positions"],
)


class JsonlFileReaderForQuestionAnswering(JsonlFileReader):
    """Jsonl file reader for question answering"""

    def _parse_instance(
        self, data, context_column="context", question_column="question", answer_column="answer", **kwargs
    ):
        instance = {
            "context": data[context_column],
            "question": data[question_column],
        }
        answer = data[answer_column]
        # if has multi answers, use the first one
        if isinstance(answer, (List, list)):
            answer = answer[0]
        instance["answer"] = answer
        return instance


class ExampleParserForQuestionAnswering:
    """Parse json instance to ExampleForQuestionAnswering"""

    def __init__(self, vocab_file, tokenization="bert-wordpiece", do_lower_case=True, **kwargs) -> None:
        assert tokenization in ["bert-wordpiece", "bert-charlevel"], "Unsupported tokenization: {}".format(tokenization)
        if tokenization == "bert-wordpiece":
            self.tokenizer = BertWordPieceTokenizer.from_file(vocab_file, lowercase=do_lower_case)
        if tokenization == "bert-charlevel":
            self.tokenizer = BertCharLevelTokenizer.from_file(vocab_file, lowercase=do_lower_case)

    @property
    def sep_token(self):
        if isinstance(self.tokenizer, BertCharLevelTokenizer):
            return self.tokenizer.sep_token
        if isinstance(self.tokenizer, BertWordPieceTokenizer):
            return self.tokenizer._parameters["sep_token"]
        raise ValueError("Invalid type of self.tokenizer.")

    def _find_answer_span(self, context, answer):
        for m in re.finditer(re.escape(answer), context, re.IGNORECASE):
            start, end = m.span()
            return start, end
        return 0, 0

    def _parse(self, context, question, answer, answer_start, answer_end, **kwargs):
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
        valid = self._valid_example(answer, encoding.tokens, ans_token_start, ans_token_end + 1)
        if not valid:
            return None
        return ExampleForQuestionAnswering(
            tokens=encoding.tokens,
            input_ids=encoding.ids,
            segment_ids=encoding.type_ids,
            attention_mask=encoding.attention_mask,
            start_positions=ans_token_start,
            end_positions=ans_token_end,
        )

    def _valid_example(self, answer, tokens, start, end):
        sub_tokens = tokens[start:end]
        ans_tokens = self.tokenizer.encode(answer, add_special_tokens=False).tokens
        if ans_tokens != sub_tokens:
            logging.debug("Invalid example. ans_tokens: %s\t sub_tokens: %s", ans_tokens, sub_tokens)
            return False
        return True

    def parse_instance(
        self,
        instance,
        context_column="context",
        question_column="question",
        answer_column="answer",
        answer_start_column=None,
        **kwargs
    ) -> ExampleForQuestionAnswering:
        context, question, answer = instance[context_column], instance[question_column], instance[answer_column]
        if answer_start_column is not None and answer_start_column in instance:
            answer_start = int(instance[answer_start_column])
            answer_end = answer_start + len(answer)
        else:
            answer_start, answer_end = self._find_answer_span(context, answer)

        if answer_end <= answer_start:
            logging.warning("Invalid answer span, skipped.")
            return None
        return self._parse(context, question, answer, answer_start, answer_end, **kwargs)

    def parse_instances(
        self,
        instances,
        context_column="context",
        question_column="question",
        answer_column="answer",
        answer_start_column=None,
        **kwargs
    ) -> Iterator[ExampleForQuestionAnswering]:
        for instance in instances:
            example = self.parse_instance(
                instance,
                context_column=context_column,
                question_column=question_column,
                answer_column=answer_column,
                answer_start_column=answer_start_column,
                **kwargs
            )
            if not example:
                continue
            yield example


def read_dureader_rubost(input_files, **kwargs):
    if isinstance(input_files, str):
        input_files = [input_files]
    for input_file in input_files:
        with open(input_file, mode="rt", encoding="utf-8") as fin:
            info = json.load(fin)
        for d in info["data"]:
            paragraphs = d["paragraphs"]
            for p in paragraphs:
                context = p["context"]
                for qa in p["qas"]:
                    question = qa["question"]
                    _id = qa["id"]
                    if not qa["answers"]:
                        continue
                    answer = qa["answers"][0]["text"]
                    instance = {"context": context, "question": question, "answer": answer, "id": _id}
                    yield instance


def read_dureader_checklist(input_files, **kwargs):
    if isinstance(input_files, str):
        input_files = [input_files]
    for input_file in input_files:
        with open(input_file, mode="rt", encoding="utf-8") as fin:
            info = json.load(fin)
        for data in info["data"]:
            for p in data["paragraphs"]:
                # title = p["title"]
                context = p["context"]
                for qa in p["qas"]:
                    if qa["is_impossible"]:
                        continue
                    question = qa["question"]
                    answer = qa["answers"][0]["text"]
                    instance = {"context": context, "question": question, "answer": answer, "id": qa["id"]}
                    yield instance
