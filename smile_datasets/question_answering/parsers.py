import abc
import json
import logging
import re
from typing import Dict

from tokenizers import BertWordPieceTokenizer

from .example import ExampleForQuestionAnswering


class AbstractExampleParser(abc.ABC):
    """Abc example parser"""

    @abc.abstractmethod
    def parse(self, instance: Dict, **kwargs):
        raise NotImplementedError()


class ParserForQuestionAnswering(AbstractExampleParser):
    """Parse example for qa task"""

    def __init__(self, tokenizer=None, vocab_file=None, do_lower_case=True, **kwargs) -> None:
        assert tokenizer or vocab_file, "`tokenizer` or `vocab_file` must be provided!"
        self.tokenizer = tokenizer or BertWordPieceTokenizer.from_file(vocab_file, lowercase=do_lower_case)

    def from_vocab(cls, vocab_file, **kwargs):
        tokenizer = BertWordPieceTokenizer.from_file(vocab_file, lowercase=kwargs.get("do_lower_case", True))
        return cls.from_tokenizer(tokenizer, **kwargs)

    def from_tokenizer(cls, tokenizer: BertWordPieceTokenizer, **kwargs):
        return cls(tokenizer=tokenizer, vocab_file=None, **kwargs)

    def _find_answer_span(self, context, answer):
        for m in re.finditer(re.escape(answer), context, re.IGNORECASE):
            start, end = m.span()
            return start, end
        return 0, 0

    def parse(self, instance: Dict, **kwargs) -> ExampleForQuestionAnswering:
        context, question, answer = instance["context"], instance["question"], instance["answer"]
        start_char_idx, end_char_idx = self._find_answer_span(context, answer)
        if end_char_idx <= start_char_idx:
            return None
        # Mark the character indexes in context that are in answer
        is_char_in_ans = [0] * len(context)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1
        context_encoding = self.tokenizer.encode(context, add_special_tokens=True)
        # Find tokens that were created from answer characters
        ans_token_idx = []
        for idx, (start_char_idx, end_char_idx) in enumerate(context_encoding.offsets):
            if sum(is_char_in_ans[start_char_idx:end_char_idx]) > 0:
                ans_token_idx.append(idx)
        if not ans_token_idx:
            logging.warning("No answer instance: %s", json.dumps(instance, ensure_ascii=False))
            logging.warning("Skipped.")
            return None
        start_token_idx, end_token_idx = ans_token_idx[0], ans_token_idx[-1]

        question_encoding = self.tokenizer.encode(question, add_special_tokens=True)
        input_ids = context_encoding.ids + question_encoding.ids[1:]
        segment_ids = [0] * len(context_encoding.type_ids) + [1] * len(question_encoding.type_ids[1:])
        attention_mask = [1] * len(context_encoding.attention_mask + question_encoding.attention_mask[1:])
        assert len(input_ids) == len(segment_ids), "input_ids length:{} VS segment_ids length: {}".format(
            len(input_ids), len(segment_ids)
        )
        assert len(input_ids) == len(attention_mask), "input_ids length:{} VS attention_mask length: {}".format(
            len(input_ids), len(attention_mask)
        )
        tokens = context_encoding.tokens + question_encoding.tokens[1:]
        valid = self.validate_answer(tokens, start_token_idx, end_token_idx, answer)
        if not valid:
            logging.warning("Invalid instance: %s", json.dumps(instance, ensure_ascii=False))
            logging.warning("Skipped.")
            return None
        example = ExampleForQuestionAnswering(
            tokens=tokens,
            input_ids=input_ids,
            segment_ids=segment_ids,
            attention_mask=attention_mask,
            start=start_token_idx,
            end=end_token_idx,
        )
        return example

    def validate_answer(self, tokens, start_index, end_index, answer):
        words = []
        for idx in range(start_index, end_index + 1):
            w = tokens[idx].lstrip("##")
            if w == "[UNK]":
                logging.warning("answer tokens constains [UNK]: start_index=%d, end_index=%d", start_index, end_index)
                return False
            words.append(w)
        words = "".join(words).lower()

        def _normalize(x):
            x = x.lower()
            x = "".join(x.split())
            return x

        words = _normalize(words)
        answer = _normalize(answer)
        if words != answer:
            logging.warning("tokens: {} != answer: {}".format(words, answer))
            return False
        return True
