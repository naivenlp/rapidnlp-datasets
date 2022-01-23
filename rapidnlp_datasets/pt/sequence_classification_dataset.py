import torch


class PTDatasetForSequenceClassification(torch.utils.data.Dataset):
    """Dataset for sequence classification in PyTorch"""

    def __init__(
        self,
        examples,
        max_sequence_length=512,
        input_ids="input_ids",
        token_type_ids="token_type_ids",
        attention_mask="attention_mask",
        label="label",
        **kwargs
    ) -> None:
        super().__init__()
        self.examples = examples
        self.max_sequence_length = max_sequence_length
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.label = label

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        e = self.examples[index]
        return {
            self.input_ids: e.input_ids,
            self.token_type_ids: e.token_type_ids,
            self.attention_mask: e.attention_mask,
            self.label: e.label,
        }

    def _padding(self, batch, max_length, **kwargs):
        input_ids, token_type_ids, attention_mask, label = [], [], [], []
        for x in batch:
            delta = max_length - len(x[self.input_ids])
            input_ids.append(x[self.input_ids] + [0] * delta)
            token_type_ids.append(x[self.token_type_ids] + [0] * delta)
            attention_mask.append(x[self.attention_mask] + [0] * delta)
            label.append(x[self.label])
        return {
            self.input_ids: torch.LongTensor(input_ids),
            self.token_type_ids: torch.LongTensor(token_type_ids),
            self.attention_mask: torch.LongTensor(attention_mask),
            self.label: torch.LongTensor(label),
        }

    @property
    def batch_padding_collator(self):
        """Padding sequence to max length in a batch."""

        def _collate_fn(batch):
            max_length = max([len(x[self.input_ids]) for x in batch])
            return self._padding(batch, max_length=max_length)

        return _collate_fn

    @property
    def fixed_padding_collator(self):
        """Padding sequence to fixed length."""

        def _collate_fn(batch):
            return self._padding(batch, max_length=self.max_sequence_length)

        return _collate_fn
