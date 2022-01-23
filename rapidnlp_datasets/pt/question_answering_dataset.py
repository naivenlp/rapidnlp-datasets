import torch
from torch.utils.data import Dataset


class PTDatasetForQuestionAnswering(Dataset):
    """Dataset for QA in PyTorch"""

    def __init__(
        self,
        examples,
        max_sequence_length=512,
        input_ids="input_ids",
        token_type_ids="token_type_ids",
        attention_mask="attention_mask",
        start_positions="start_positions",
        end_positions="end_positions",
        **kwargs
    ) -> None:
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.start_positions = start_positions
        self.end_positions = end_positions

        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        return {
            self.input_ids: example.input_ids,
            self.token_type_ids: example.token_type_ids,
            self.attention_mask: example.attention_mask,
            self.start_positions: example.start_positions,
            self.end_positions: example.end_positions,
        }

    def _padding(self, batch, max_length):
        input_ids, token_type_ids, attention_mask = [], [], []
        start_positions, end_positions = [], []
        for x in batch:
            delta = max_length - len(x[self.input_ids])
            input_ids.append(x[self.input_ids] + [0] * delta)
            token_type_ids.append(x[self.token_type_ids] + [0] * delta)
            attention_mask.append(x[self.attention_mask] + [0] * delta)
            start_positions.append(x[self.start_positions])
            end_positions.append(x[self.end_positions])
        padded_inputs = {
            self.input_ids: torch.LongTensor(input_ids),
            self.token_type_ids: torch.LongTensor(token_type_ids),
            self.attention_mask: torch.LongTensor(attention_mask),
            self.start_positions: torch.LongTensor(start_positions),
            self.end_positions: torch.LongTensor(end_positions),
        }
        return padded_inputs

    @property
    def fixed_padding_collator(self):
        """Padding sequence to fixed length"""

        def _collate_fn(batch):
            return self._padding(batch, max_length=self.max_sequence_length)

        return _collate_fn

    @property
    def batch_padding_collator(self):
        """Padding sequence to max length in a batch"""

        def _collate_fn(batch):
            max_length = max([len(x["input_ids"]) for x in batch])
            return self._padding(batch, max_length=max_length)

        return _collate_fn
