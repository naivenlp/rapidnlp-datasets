from .example import ExampleForTokenClassification
from .tokenizers import BertCharLevelTokenizer, LabelTokenizerForTokenClassification


class ParserForTokenClassification:
    """Example parser for token classification"""

    def __init__(
        self,
        feature_tokenizer: BertCharLevelTokenizer = None,
        feature_vocab_file=None,
        label_tokenizer: LabelTokenizerForTokenClassification = None,
        label_vocab_file=None,
        do_lower_case=True,
        o_token="O",
        **kwargs
    ) -> None:
        assert feature_tokenizer or feature_vocab_file, "`feature_tokenizer` or `feature_vocab_file` must be provided!"
        assert label_tokenizer or label_vocab_file, "`label_tokenizer` or `label_vocab_file` must be provided!"

        self.feature_tokenizer = feature_tokenizer or BertCharLevelTokenizer.from_file(
            feature_vocab_file,
            do_lower_case=do_lower_case,
        )
        self.label_tokenizer = label_tokenizer or LabelTokenizerForTokenClassification.from_file(
            label_vocab_file,
            o_token=o_token,
        )

    @classmethod
    def from_vocab_file(cls, feature_vocab_file, label_vocab_file, **kwargs):
        return cls(
            feature_tokenizer=None,
            feature_vocab_file=feature_vocab_file,
            label_tokenizer=None,
            label_vocab_file=label_vocab_file,
            **kwargs
        )

    @classmethod
    def from_tokenizer(
        cls, feature_tokenizer: BertCharLevelTokenizer, label_tokenizer: LabelTokenizerForTokenClassification = None, **kwargs
    ):
        return cls(
            feature_tokenizer=feature_tokenizer,
            feature_vocab_file=None,
            label_tokenizer=label_tokenizer,
            label_vocab_file=None,
            **kwargs
        )

    def parse(self, instance, **kwargs) -> ExampleForTokenClassification:
        feature, label = instance["feature"], instance["label"]
        token_ids = [self.feature_tokenizer.token_to_id(x) for x in feature]
        label_ids = self.label_tokenizer.labels_to_ids(label, add_cls=False, add_sep=False)
        example = ExampleForTokenClassification(
            tokens=feature,
            labels=label,
            input_ids=token_ids,
            segment_ids=[0] * len(token_ids),
            attention_mask=[1] * len(token_ids),
            label_ids=label_ids,
        )
        return example
