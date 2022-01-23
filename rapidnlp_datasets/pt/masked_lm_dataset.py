import torch
from torch.utils.data import Dataset


class PTDatasetForMaskedLanguageModel(Dataset):
    """Dataset for mlm in PyTorch"""

    def __init__(
        self,
        examples,
        max_sequence_length=512,
        ignore_index=-100,
        input_ids="input_ids",
        token_type_ids="token_type_ids",
        attention_mask="attention_mask",
        labels="labels",
        **kwargs
    ):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.ignore_index = ignore_index
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.labels = labels
        # examples
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        e = self.examples[index]
        labels = [self.ignore_index] * len(e.input_ids)
        for pos, _id in zip(e.masked_pos, e.masked_ids):
            labels[pos] = _id
        return {
            self.input_ids: e.input_ids,
            self.token_type_ids: e.token_type_ids,
            self.attention_mask: e.attention_mask,
            self.labels: labels,
        }

    def _padding(self, batch, max_length, **kwargs):
        input_ids, token_type_ids, attention_mask, labels = [], [], [], []
        for x in batch:
            delta = max_length - len(x[self.input_ids])
            input_ids.append(x[self.input_ids] + [0] * delta)
            token_type_ids.append(x[self.token_type_ids] + [0] * delta)
            attention_mask.append(x[self.attention_mask] + [1] * delta)
            labels.append(x[self.labels] + [self.ignore_index] * delta)
        return {
            self.input_ids: torch.LongTensor(input_ids),
            self.token_type_ids: torch.LongTensor(token_type_ids),
            self.attention_mask: torch.LongTensor(attention_mask),
            self.labels: torch.LongTensor(labels),
        }

    @property
    def batch_padding_collator(self):
        """Padding sequence to max length in a batch"""

        def _collate_fn(batch):
            max_length = max([len(x[self.input_ids]) for x in batch])
            return self._padding(batch, max_length=max_length)

        return _collate_fn

    @property
    def fixed_padding_collator(self):
        """Padding sequence to a fixed length"""

        def _collate_fn(batch):
            return self._padding(batch, max_length=self.max_sequence_length)

        return _collate_fn
