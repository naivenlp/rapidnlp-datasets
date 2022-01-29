import torch

from .dataset import PTDataset


class PTDatasetForTokenClassification(PTDataset):
    """Dataset for token classification in PyTorch"""

    def __init__(self, examples=None, pad_id=0, max_sequence_length=512, **kwargs) -> None:
        super().__init__(pad_id=pad_id, max_sequence_length=max_sequence_length, **kwargs)
        self.input_ids = kwargs.pop("input_ids", "input_ids")
        self.token_type_ids = kwargs.pop("token_type_ids", "token_type_ids")
        self.attention_mask = kwargs.pop("attention_mask", "attention_mask")
        self.labels = kwargs.pop("labels", "labels")
        self.examples = examples or []

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        e = self.examples[index]
        inputs = {
            self.input_ids: e.input_ids,
            self.token_type_ids: e.token_type_ids,
            self.attention_mask: e.attention_mask,
            self.labels: e.label_ids,
        }
        return inputs

    def _padding(self, batch, max_sequence):
        input_ids, token_type_ids, attention_mask, labels = [], [], [], []
        for x in batch:
            delta = max_sequence - len(x[self.input_ids])
            input_ids.append(x[self.input_ids] + [self.pad_id] * delta)
            token_type_ids.append(x[self.token_type_ids] + [self.pad_id] * delta)
            attention_mask.append(x[self.attention_mask] + [self.pad_id] * delta)
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
            return self._padding(batch, max_length)

        return _collate_fn

    @property
    def fixed_padding_collator(self):
        """Padding sequence to a fixed max length"""

        def _collate_fn(batch):
            return self._padding(batch, self.max_sequence_length)

        return _collate_fn
