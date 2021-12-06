import abc
import logging
import os
import random
import re
from collections import namedtuple

from tokenizers import BertWordPieceTokenizer

from .parsers import AbstractExampleParser
from .readers import AbstractFileReader, CsvFileReader, JsonlFileReader

ExampleForMaskedLanguageModel = namedtuple(
    "ExampleForMaskedLanguageModel",
    ["tokens", "input_ids", "segment_ids", "attention_mask", "masked_ids", "masked_pos"],
)


ResultForMasking = namedtuple("ResultForMasking", ["origin_tokens", "masked_tokens", "masked_indexes"])


class AbstractMaskingStrategy(abc.ABC):
    """Abstract masking strategy"""

    @abc.abstractmethod
    def __call__(self, tokens, **kwargs) -> ResultForMasking:
        raise NotImplementedError()


class WholeWordMask(AbstractMaskingStrategy):
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

    def __call__(self, tokens, max_sequence_length=512, **kwargs) -> ResultForMasking:
        tokens = self._truncate_sequence(tokens, max_sequence_length - 2)
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
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        masked_tokens = ["[CLS]"] + masked_tokens + ["[SEP]"]
        masked_indexes = [0] + masked_indexes + [0]
        assert len(tokens) == len(masked_tokens) == len(masked_indexes)
        return ResultForMasking(origin_tokens=tokens, masked_tokens=masked_tokens, masked_indexes=masked_indexes)

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
            if cand_indexes and token.startswith("##"):
                cand_indexes[-1].append(idx)
                continue
            cand_indexes.append([idx])
        random.shuffle(cand_indexes)
        return cand_indexes

    def _truncate_sequence(self, tokens, max_tokens=512, **kwargs):
        while len(tokens) > max_tokens:
            if len(tokens) > max_tokens:
                tokens.pop(0)
                # truncate whole world
                while tokens and tokens[0].startswith("##"):
                    tokens.pop(0)
            if len(tokens) > max_tokens:
                while tokens and tokens[-1].startswith("##"):
                    tokens.pop()
                if tokens:
                    tokens.pop()
        return tokens


class JsonlFileReaderForMaskedLanguageModel(JsonlFileReader):
    """Jsonl file reader for mlm"""

    def _parse_instance(self, data, sequence_column="sequence", **kwargs):
        return {"sequence": data[sequence_column]}


class CsvFileReaderForMaskedLanguageModel(CsvFileReader):
    """Csv file reader for mlm"""

    def _parse_instance(self, line, sequence_column=0, sep=",", **kwargs):
        parts = re.split(sep, line)
        return {"sequence": parts[sequence_column]}


class TextFileReaderForMaskedLanguageModel(AbstractFileReader):
    """Text file reader for mlm"""

    def read_files(self, input_files, **kwargs):
        if isinstance(input_files, str):
            input_files = [input_files]
        for f in input_files:
            if not os.path.exists(f) or not os.path.isfile(f):
                logging.warning("File %d does not exist, skipped.", f)
                continue
            with open(f, mode="rt", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    instance = {"sequence": line}
                    yield instance


class ExampleParserForMaskedLanguageModel(AbstractExampleParser):
    """Parse instance to ExampleForMaskedLanguageModel"""

    def __init__(self, vocab_file, masking="whole-word-mask", do_lower_case=True, **kwargs) -> None:
        super().__init__()
        self.tokenizer = BertWordPieceTokenizer.from_file(vocab_file, lowercase=do_lower_case)
        vocabs = list(self.tokenizer.get_vocab().keys())
        assert masking in ["whole-word-mask"]
        if masking == "whole-word-mask":
            self.masking = WholeWordMask(vocabs=vocabs, **kwargs)
        else:
            self.masking = None

    def parse_instance(self, instance, max_sequence_length=512, **kwargs):
        sequence = instance["sequence"]
        encoding = self.tokenizer.encode(sequence, add_special_tokens=False)
        masking_results = self.masking(tokens=encoding.tokens, max_sequence_length=max_sequence_length, **kwargs)
        origin_tokens, masked_tokens = masking_results.origin_tokens, masking_results.masked_tokens
        example = ExampleForMaskedLanguageModel(
            tokens=masked_tokens,
            input_ids=[self.tokenizer.token_to_id(x) for x in masked_tokens],
            segment_ids=[0] * len(masked_tokens),
            attention_mask=[1] * len(masked_tokens),
            masked_ids=[self.tokenizer.token_to_id(x) for x in origin_tokens],
            masked_pos=masking_results.masked_indexes,
        )
        return example

    def parse_instances(self, instances, max_sequence_length=512, **kwargs):
        for instance in instances:
            e = self.parse_instance(instance, max_sequence_length=max_sequence_length, **kwargs)
            if not e:
                continue
            yield e
