import abc
from collections import namedtuple

from tokenizers import BertWordPieceTokenizer

CharLevelEncoding = namedtuple("CharLevelEncoding", ["tokens", "ids", "type_ids", "attention_mask", "offsets"])


class CharLevelTokenizer(abc.ABC):
    """Abstract char-level tokenizer"""

    @abc.abstractmethod
    def encode(self, sequence, pair=None, is_pretokenized=False, add_special_tokens=True):
        raise NotImplementedError()

    @abc.abstractmethod
    def encode_batch(self, inputs, is_pretokenized=False, add_special_tokens=True):
        raise NotImplementedError()

    @abc.abstractmethod
    def decode(self, ids, skip_special_tokens=True):
        raise NotImplementedError()

    @abc.abstractmethod
    def decode_batch(self, sequences, skip_special_tokens=True):
        raise NotImplementedError()


class BertCharLevelTokenizer(CharLevelTokenizer):
    """Bert char-level tokenizer"""

    @classmethod
    def from_file(cls, vocab_file, lowercase=True, **kwargs):
        tokenizer = BertWordPieceTokenizer.from_file(vocab_file, lowercase=lowercase)
        return cls(tokenizer=tokenizer, **kwargs)

    def __init__(self, tokenizer: BertWordPieceTokenizer, **kwargs) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    @property
    def pad_token(self):
        return self.tokenizer._parameters["cls_token"]

    @property
    def unk_token(self):
        return self.tokenizer._parameters["unk_token"]

    @property
    def cls_token(self):
        return self.tokenizer._parameters["cls_token"]

    @property
    def sep_token(self):
        return self.tokenizer._parameters["sep_token"]

    @property
    def mask_token(self):
        return self.tokenizer._parameters["mask_token"]

    @property
    def special_tokens(self):
        return [self.unk_token, self.pad_token, self.cls_token, self.sep_token, self.mask_token]

    def token_to_id(self, token):
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, _id):
        return self.tokenizer.id_to_token(_id)

    def _to_char_level(self, encoding, **kwargs):
        tokens, ids, type_ids, attention_mask, offsets = [], [], [], [], []
        for token, _id, tid, mask, offset in zip(
            encoding.tokens, encoding.ids, encoding.type_ids, encoding.attention_mask, encoding.offsets
        ):
            if token in self.special_tokens:
                tokens.append(token)
                ids.append(_id)
                type_ids.append(tid)
                attention_mask.append(mask)
                offsets.append(offset)
                continue
            token = str(token).lstrip("##")
            for idx, c in enumerate(token):
                ids.append(len(tokens))
                tokens.append(c)
                type_ids.append(tid)
                attention_mask.append(mask)
                start = offset[0]
                offsets.append((start + idx, start + idx + 1))
        char_encoding = CharLevelEncoding(
            tokens=tokens,
            ids=ids,
            type_ids=type_ids,
            attention_mask=attention_mask,
            offsets=offsets,
        )
        return char_encoding

    def encode(self, sequence, pair=None, is_pretokenized=False, add_special_tokens=True):
        encoding = self.tokenizer.encode(
            sequence, pair=pair, is_pretokenized=is_pretokenized, add_special_tokens=add_special_tokens
        )
        return self._to_char_level(encoding)

    def encode_batch(self, inputs, is_pretokenized=False, add_special_tokens=True):
        encodings = self.tokenizer.encode_batch(
            inputs, is_pretokenized=is_pretokenized, add_special_tokens=add_special_tokens
        )
        return [self._to_char_level(encoding) for encoding in encodings]

    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, sequences, skip_special_tokens=True):
        return self.tokenizer.decode_batch(sequences, skip_special_tokens=skip_special_tokens)
