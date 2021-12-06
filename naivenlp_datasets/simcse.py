import re

from tokenizers import BertWordPieceTokenizer

from .parsers import AbstractExampleParser
from .readers import CsvFileReader, JsonlFileReader


class ExampleForSimCSE:
    """Example for SimCSE model"""

    def __init__(self, **kwargs) -> None:
        self.tokens = kwargs.get("tokens", None)
        self.input_ids = kwargs.get("input_ids", None)
        self.segment_ids = kwargs.get("segment_ids", None)
        self.attention_mask = kwargs.get("attention_mask", None)
        self.pos_tokens = kwargs.get("pos_tokens", None)
        self.pos_input_ids = kwargs.get("pos_input_ids", None)
        self.pos_segment_ids = kwargs.get("pos_segment_ids", None)
        self.pos_attention_mask = kwargs.get("pos_attention_mask", None)
        self.neg_tokens = kwargs.get("neg_tokens", None)
        self.neg_input_ids = kwargs.get("neg_input_ids", None)
        self.neg_segment_ids = kwargs.get("neg_segment_ids", None)
        self.neg_attention_mask = kwargs.get("neg_attention_mask", None)

    def __hash__(self) -> int:
        values = [getattr(self, k, None) for k in self.__dict__.keys()]
        return hash(values)

    def __eq__(self, __o: object) -> bool:
        if not __o:
            return False
        if not isinstance(__o, ExampleForSimCSE):
            return False
        for k, v in self.__dict__.items():
            if v != getattr(__o, k, None):
                return False
        return True

    def __repr__(self) -> str:
        builder = []
        for k, v in self.__dict__.items():
            builder.append("{}={}".format(k, v))
        return "ExampleForSimCSE[" + ",".join(builder) + "]"

    def __str__(self) -> str:
        return self.__repr__()


class JsonlFileReaderForSimCSE(JsonlFileReader):
    """Jsonl file reader for simcse"""

    def __init__(
        self,
        sequence_column="sequence",
        with_pos_sequence=False,
        pos_sequence_column="pos_sequence",
        with_neg_sequence=False,
        neg_sequence_column="neg_sequence",
        **kwargs
    ) -> None:
        super().__init__()
        self.sequence_column = sequence_column
        self.with_pos_sequence = with_pos_sequence
        self.pos_sequence_column = pos_sequence_column
        self.with_neg_sequence = with_neg_sequence
        self.neg_sequence_column = neg_sequence_column

    def _parse_instance(self, data, **kwargs):
        instance = {
            "sequence": data[self.sequence_column],
        }
        if self.with_pos_sequence and self.pos_sequence_column in data:
            instance["pos_sequence"] = data[self.pos_sequence_column]
        if self.with_neg_sequence and self.neg_sequence_column in data:
            instance["neg_sequence"] = data[self.neg_sequence_column]
        return instance


class CsvFileReaderForSimCSE(CsvFileReader):
    """Csv file reader for simcse"""

    def __init__(
        self,
        sequence_column=0,
        with_pos_sequence=False,
        pos_sequence_column=1,
        with_neg_sequence=False,
        neg_sequence_column=2,
        **kwargs
    ) -> None:
        super().__init__()
        self.sequence_column = sequence_column
        self.with_pos_sequence = with_pos_sequence
        self.pos_sequence_column = pos_sequence_column
        self.with_neg_sequence = with_neg_sequence
        self.neg_sequence_column = neg_sequence_column

    def _parse_instance(self, line, sep=",", **kwargs):
        data = re.split(sep, line)
        instance = {
            "sequence": data[self.sequence_column],
        }
        if self.with_pos_sequence and self.pos_sequence_column in data:
            instance["pos_sequence"] = data[self.pos_sequence_column]
        if self.with_neg_sequence and self.neg_sequence_column in data:
            instance["neg_sequence"] = data[self.neg_sequence_column]
        return instance


class ExampleParserForSimCSE(AbstractExampleParser):
    """Parse json instance to ExampleForSimCSE"""

    def __init__(
        self, vocab_file, with_pos_sequence=False, with_neg_sequence=False, do_lower_case=True, **kwargs
    ) -> None:
        super().__init__()
        self.tokenizer = BertWordPieceTokenizer.from_file(vocab_file, lowercase=do_lower_case)
        self.with_pos_sequence = with_pos_sequence
        self.with_neg_sequence = with_neg_sequence

    def parse_instance(self, instance, add_special_tokens=True, **kwargs):
        seq_encoding = self.tokenizer.encode(instance["sequence"], add_special_tokens=add_special_tokens)
        e = ExampleForSimCSE(
            tokens=seq_encoding.tokens,
            input_ids=seq_encoding.ids,
            segment_ids=seq_encoding.type_ids,
            attention_mask=seq_encoding.attention_mask,
        )
        if self.with_pos_sequence:
            pos_encoding = self.tokenizer.encode(instance["pos_sequence"], add_special_tokens=add_special_tokens)
            e.pos_tokens = pos_encoding.tokens
            e.pos_input_ids = pos_encoding.ids
            e.pos_segment_ids = pos_encoding.type_ids
            e.pos_attention_mask = pos_encoding.attention_mask
        if self.with_neg_sequence:
            neg_encoding = self.tokenizer.encode(instance["neg_sequence"], add_special_tokens=add_special_tokens)
            e.neg_tokens = neg_encoding.tokens
            e.neg_input_ids = neg_encoding.ids
            e.neg_segment_ids = neg_encoding.type_ids
            e.neg_attention_mask = neg_encoding.attention_mask
        return e

    def parse_instances(self, instances, add_special_tokens=True, **kwargs):
        for instance in instances:
            e = self.parse_instance(instance, add_special_tokens=add_special_tokens, **kwargs)
            yield e
