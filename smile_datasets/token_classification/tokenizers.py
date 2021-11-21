from collections import namedtuple

TokenizerEncoding = namedtuple("TokenizerEncoding", ["text", "tokens", "ids", "type_ids", "attention_mask", "offsets"])


class LabelTokenizerForTokenClassification:
    """Tokenizer for labels, used in token classification tasks."""

    def __init__(self, label2id, o_token="O", **kwargs) -> None:
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        self.o_token = o_token
        self.o_id = self.label2id[o_token]

    def label_to_id(self, token, **kwargs):
        return self.label2id.get(token, self.o_id)

    def id_to_label(self, id, **kwargs):
        return self.id2label.get(id, self.o_token)

    def labels_to_ids(self, tokens, add_cls=False, add_sep=False, **kwargs):
        ids = [self.label2id.get(token, self.o_id) for token in tokens]
        if add_cls:
            ids = [self.o_id] + ids
        if add_sep:
            ids = ids + [self.o_id]
        return ids

    def ids_to_labels(self, ids, del_cls=True, del_sep=True, **kwargs):
        tokens = [self.id2label.get(_id, self.o_token) for _id in ids]
        if tokens and del_cls and tokens[0] == self.o_token:
            tokens = tokens[1:]
        if tokens and del_sep and tokens[-1] == self.o_token:
            tokens = tokens[:-1]
        return tokens

    @classmethod
    def from_file(cls, vocab_file, o_token="O", **kwargs):
        label2id = cls._read_label_vocab(vocab_file)
        return cls(label2id=label2id, o_token=o_token, **kwargs)

    @classmethod
    def _read_label_vocab(cls, vocab_file):
        m = {}
        with open(vocab_file, mode="rt", encoding="utf-8") as fin:
            for line in fin:
                k = line.strip()
                m[k] = len(m)
        return m


class BertCharLevelTokenizer:
    """Bert char level tokenizer for token classification tasks."""

    def __init__(
        self,
        token2id,
        do_lower_case=True,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        **kwargs
    ):
        super().__init__()
        self.token2id = token2id
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token
        self.unk_id = self.token2id[self.unk_token]
        self.pad_id = self.token2id[self.pad_token]
        self.cls_id = self.token2id[self.cls_token]
        self.sep_id = self.token2id[self.sep_token]
        self.mask_id = self.token2id[self.mask_token]
        self.do_lower_case = do_lower_case

    @classmethod
    def from_file(cls, vocab_file, **kwargs):
        token2id = cls._load_vocab(vocab_file=vocab_file)
        tokenizer = cls(token2id, **kwargs)
        return tokenizer

    @classmethod
    def _load_vocab(cls, vocab_file):
        vocab = {}
        idx = 0
        with open(vocab_file, mode="rt", encoding="utf-8") as fin:
            for line in fin:
                word = line.rstrip("\n")
                vocab[word] = idx
                idx += 1
        return vocab

    def _encode_legacy(self, text, add_cls=True, add_sep=True, **kwargs):
        tokens, ids, type_ids, attention_mask, offsets = [], [], [], [], []
        if self.do_lower_case:
            text = str(text).lower()
        for idx, char in enumerate(text):
            tokens.append(char)
            ids.append(self.token2id.get(char, self.unk_id))
            type_ids.append(0)
            attention_mask.append(1)
            offsets.append((idx, idx + 1))

        if add_cls:
            tokens.insert(0, self.cls_token)
            ids.insert(0, self.cls_id)
            type_ids.insert(0, 0)
            attention_mask.insert(0, 1)
            offsets.insert(0, (0, 0))

        if add_sep:
            tokens.append(self.sep_token)
            ids.append(self.sep_id)
            type_ids.append(0)
            attention_mask.append(1)
            offsets.append((0, 0))

        encoding = TokenizerEncoding(
            text=text,
            tokens=tokens,
            ids=ids,
            type_ids=type_ids,
            attention_mask=attention_mask,
            offsets=offsets,
        )
        return encoding

    def encode(self, sequence, pair=None, add_special_tokens=False, **kwargs):
        """Encode text to encodings"""
        if self.do_lower_case:
            sequence = [x.lower() for x in sequence]
        if not pair:
            encoding = self._encode_sequence(sequence, add_special_tokens=add_special_tokens, **kwargs)
            return encoding
        encoding = self._encode_pair(sequence, pair, add_special_tokens=True, **kwargs)
        return encoding

    def _encode_sequence(self, sequence, add_special_tokens=False, **kwargs):
        encoding = self._encode_legacy(sequence, add_cls=add_special_tokens, add_sep=add_special_tokens, **kwargs)
        return encoding

    def _encode_pair(self, sequence, pair, add_special_tokens=True, **kwargs):
        seq_encoding = self._encode_legacy(sequence, add_cls=True, add_sep=True)
        pai_encoding = self._encode_legacy(pair, add_cls=False, add_sep=True)
        encoding = TokenizerEncoding(
            text=seq_encoding.text + pai_encoding.text,
            tokens=seq_encoding.tokens + pai_encoding.tokens,
            ids=seq_encoding.ids + pai_encoding.ids,
            type_ids=[0] * len(seq_encoding.type_ids) + [1] * len(pai_encoding.type_ids),
            attention_mask=[1] * (len(seq_encoding.attention_mask) + len(pai_encoding.attention_mask)),
            offsets=None,  # offsets set to None in CharLevel tokenizer
        )
        return encoding

    def decode(self, ids, **kwargs):
        tokens = [self.id2token.get(_id, self.unk_token) for _id in ids]
        return tokens

    def token_to_id(self, token, **kwargs):
        if self.do_lower_case:
            token = token.lower()
        return self.token2id.get(token, self.unk_id)

    def id_to_token(self, id, **kwargs):
        return self.id2token.get(id, self.unk_token)