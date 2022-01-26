import torch


class PTDatasetForSimCSE(torch.utils.data.Dataset):
    """Dataset for SimCSE in PyTorch"""

    def __init__(
        self, examples, max_sequence_length=512, with_positive_sequence=False, with_negative_sequence=False, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.examples = examples
        self.max_sequence_length = max_sequence_length
        self.with_positive_sequence = with_positive_sequence
        self.with_negative_sequence = with_negative_sequence

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        e = self.examples[index]
        features = {
            "input_ids": e.input_ids,
            "token_type_ids": e.token_type_ids,
            "attention_mask": e.attention_mask,
        }
        if self.with_positive_sequence:
            features.update(
                {
                    "pos_input_ids": e.pos_input_ids,
                    "pos_token_type_ids": e.pos_token_type_ids,
                    "pos_attention_mask": e.pos_attention_mask,
                }
            )
        if self.with_negative_sequence:
            features.update(
                {
                    "neg_input_ids": e.neg_input_ids,
                    "neg_token_type_ids": e.neg_token_type_ids,
                    "neg_attention_mask": e.neg_attention_mask,
                }
            )
        return features

    def _padding_group(self, batch, max_length, prefix=""):
        input_ids, token_type_ids, attention_mask = [], [], []
        for x in batch:
            delta = max_length - len(x[prefix + "input_ids"])
            input_ids.append(x[prefix + "input_ids"] + [0] * delta)
            token_type_ids.append(x[prefix + "token_type_ids"] + [0] * delta)
            attention_mask.append(x[prefix + "attention_mask"] + [0] * delta)
        return {
            prefix + "input_ids": torch.LongTensor(input_ids),
            prefix + "token_type_ids": torch.LongTensor(token_type_ids),
            prefix + "attention_mask": torch.LongTensor(attention_mask),
        }

    def _padding(self, batch, max_length):
        inputs = {}
        inputs.update(self._padding_group(batch, max_length, prefix=""))
        if self.with_positive_sequence:
            inputs.update(self._padding_group(batch, max_length, prefix="pos_"))
        if self.with_negative_sequence:
            inputs.update(self._padding_group(batch, max_length, prefix="neg_"))
        return inputs

    @property
    def batch_padding_collator(self):
        """Padding sequence to max length in batch"""

        def _collate_fn(batch):
            max_length = max([len(x["input_ids"]) for x in batch])
            return self._padding(batch, max_length)

        return _collate_fn

    @property
    def fixed_padding_collator(self):
        """Padding sequence to a fixed length"""

        def _collate_fn(batch):
            return self._padding(batch, self.max_sequence_length)

        return _collate_fn
