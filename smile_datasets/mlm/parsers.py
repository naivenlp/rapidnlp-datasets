from tokenizers import BertWordPieceTokenizer

from .example import ExampleForMaskedLanguageModel
from .masking_strategy import WholeWordMask


class ParserForMaskedLanguageModel:
    """ """

    @classmethod
    def from_vocab_file(cls, vocab_file, **kwargs):
        return cls(tokenizer=None, vocab_file=vocab_file, **kwargs)

    @classmethod
    def from_tokenizer(cls, tokenizer: BertWordPieceTokenizer, **kwargs):
        return cls(tokenizer=tokenizer, vocab_file=None, **kwargs)

    def __init__(self, tokenizer: BertWordPieceTokenizer, vocab_file=None, do_lower_case=True, **kwargs) -> None:
        assert tokenizer or vocab_file, "`tokenizer` or `vocab_file` must be provided!"
        self.tokenizer = tokenizer or BertWordPieceTokenizer.from_file(vocab_file, lowercase=do_lower_case)
        self.vocabs = list(self.tokenizer.get_vocab().keys())
        self.masking = WholeWordMask(vocabs=self.vocabs, **kwargs)

    def parse(self, instance, max_sequence_length=512, **kwargs) -> ExampleForMaskedLanguageModel:
        sequence = instance["sequence"]
        # set add_special_tokens=False here, masking strategy will add these special tokens
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
