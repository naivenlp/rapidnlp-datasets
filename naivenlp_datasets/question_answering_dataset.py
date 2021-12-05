import json
import re
from collections import namedtuple
from typing import Iterator, List

from tokenizers import BertWordPieceTokenizer

from naivenlp_datasets.readers import JsonlFileReader

from .charlevel_tokenizer import BertCharLevelTokenizer

ExampleForQuestionAnswering = namedtuple(
    "ExampleForQuestionAnswering",
    ["tokens", "input_ids", "segment_ids", "attention_mask", "start_positions", "end_positions"],
)


class JsonlFileReaderForQuestionAnswering(JsonlFileReader):
    """Jsonl file reader for question answering"""

    def _parse_instance(
        self, data, context_column="context", question_column="question", answers_column="answer", **kwargs
    ):
        instance = {
            "context": data[context_column],
            "question": data[question_column],
        }
        answer = data[answers_column]
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

    def parse_instance(self, instance, **kwargs) -> ExampleForQuestionAnswering:
        pass

    def parse_instances(self, instances, **kwargs) -> Iterator[ExampleForQuestionAnswering]:
        pass


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
