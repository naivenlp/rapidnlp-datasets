from tokenizers import BertWordPieceTokenizer

from .example import ExampleForHardNegativeSimCSE, ExampleForSupervisedSimCSE, ExampleForUnsupervisedSimCSE


class ParserForUnsupervisedSimCSE:
    """Parser for simcse"""

    @classmethod
    def from_tokenizer(cls, tokenizer: BertWordPieceTokenizer, **kwargs):
        return cls(tokenizer=tokenizer, vocab_file=None, **kwargs)

    @classmethod
    def from_vocab_file(cls, vocab_file, **kwargs):
        return cls(tokenizer=None, vocab_file=vocab_file, **kwargs)

    def __init__(
        self, tokenizer: BertWordPieceTokenizer = None, vocab_file=None, do_lower_case=True, add_special_tokens=True, **kwargs
    ) -> None:
        assert tokenizer or vocab_file, "`tokenizer` or `vocab_file` must be provided!"
        self.tokenizer = tokenizer or BertWordPieceTokenizer.from_file(vocab_file, lowercase=do_lower_case)
        self.add_special_tokens = add_special_tokens

    def parse(self, instance, **kwargs) -> ExampleForUnsupervisedSimCSE:
        sequence = instance["sequence"]
        encoding = self.tokenizer.encode(sequence, add_special_tokens=self.add_special_tokens)
        example = ExampleForUnsupervisedSimCSE(
            tokens=encoding.tokens,
            input_ids=encoding.ids,
            segment_ids=encoding.type_ids,
            attention_mask=encoding.attention_mask,
        )
        return example


class ParserForSupervisedSimCSE(ParserForUnsupervisedSimCSE):
    """Parser for supervised simcse."""

    def parse(self, instance, **kwargs) -> ExampleForSupervisedSimCSE:
        sequence, pos_sequence = instance["sequence"], instance["pos_sequence"]
        seq_encoding = self.tokenizer.encode(sequence, add_special_tokens=self.add_special_tokens)
        pos_encoding = self.tokenizer.encode(pos_sequence, add_special_tokens=self.add_special_tokens)
        example = ExampleForSupervisedSimCSE(
            tokens=seq_encoding.tokens,
            input_ids=seq_encoding.ids,
            segment_ids=seq_encoding.type_ids,
            attention_mask=seq_encoding.attention_mask,
            pos_tokens=pos_encoding.tokens,
            pos_input_ids=pos_encoding.ids,
            pos_segment_ids=pos_encoding.type_ids,
            pos_attention_mask=pos_encoding.attention_mask,
        )
        return example


class ParserForHardNegSimCSE(ParserForUnsupervisedSimCSE):
    """Parser for hard negative simcse."""

    def parse(self, instance, **kwargs) -> ExampleForHardNegativeSimCSE:
        sequence, pos_sequence, neg_sequence = instance["sequence"], instance["pos_sequence"], instance["neg_sequence"]
        seq_encoding = self.tokenizer.encode(sequence, add_special_tokens=self.add_special_tokens)
        pos_encoding = self.tokenizer.encode(pos_sequence, add_special_tokens=self.add_special_tokens)
        neg_encoding = self.tokenizer.encode(neg_sequence, add_special_tokens=self.add_special_tokens)
        example = ExampleForHardNegativeSimCSE(
            tokens=seq_encoding.tokens,
            input_ids=seq_encoding.ids,
            segment_ids=seq_encoding.type_ids,
            attention_mask=seq_encoding.attention_mask,
            pos_tokens=pos_encoding.tokens,
            pos_input_ids=pos_encoding.ids,
            pos_segment_ids=pos_encoding.type_ids,
            pos_attention_mask=pos_encoding.attention_mask,
            neg_tokens=neg_encoding.tokens,
            neg_input_ids=neg_encoding.ids,
            neg_segment_ids=neg_encoding.type_ids,
            neg_attention_mask=neg_encoding.attention_mask,
        )
        return example
