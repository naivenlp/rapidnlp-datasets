import abc

import torch


class PTDataset(torch.utils.data.Dataset):
    """Dataset in PyTorch"""

    def __init__(self, pad_id=0, max_sequence_length=512, **kwargs) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.max_sequence_length = max_sequence_length

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def batch_padding_collate(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def fixed_padding_collate(self):
        raise NotImplementedError()
