from tokenizers import BertWordPieceTokenizer

from .example import ExampleForSequenceClassification


class ParserForSequenceClassification:
    """ """

    @classmethod
    def from_tokenizer(cls, tokenizer: BertWordPieceTokenizer, **kwargs):
        return cls(tokenizer=tokenizer, vocab_file=None, **kwargs)

    @classmethod
    def from_vocab_file(cls, vocab_file, **kwargs):
        return cls(tokenizers=None, vocab_file=vocab_file, **kwargs)

    def __init__(self, tokenizer: BertWordPieceTokenizer = None, vocab_file=None, do_lower_case=True, **kwargs) -> None:
        assert tokenizer or vocab_file, "`tokenizer` or `vocab_file` must be provided!"
        self.tokenizer = tokenizer or BertWordPieceTokenizer.from_file(vocab_file, lowercase=do_lower_case)

    def parse(self, instance, add_special_tokens=True, **kwargs) -> ExampleForSequenceClassification:
        sequence = instance["sequence"]
        pair = instance.get("pair", None)
        label = int(instance["label"])
        encoding = self.tokenizer.encode(sequence, pair=pair, add_special_tokens=add_special_tokens)
        example = ExampleForSequenceClassification(
            tokens=encoding.tokens,
            input_ids=encoding.ids,
            segment_ids=encoding.type_ids,
            attention_mask=encoding.attention_mask,
            label=label,
        )
        return example
