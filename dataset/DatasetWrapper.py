import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def __metadata_injector__(s, *a, **kw): return s


class DatasetWrapper(Dataset):

    @staticmethod
    def convert_key(key, inplace):
        return key

    def __init__(self, dataset, inplace=False, metadata=None, metadata_injector=None, injection=0, transform=None,
                 **kwargs):
        if metadata is not None:
            for k in metadata.keys():
                if type(metadata[k]) == np.ndarray:
                    metadata[k] = torch.tensor(metadata[k])
        if metadata_injector is None:
            metadata_injector = __metadata_injector__
        self.dataset = dataset
        self.inplace = inplace
        self.metadata = metadata
        self.metadata_injector = metadata_injector
        self.injection = injection
        self.injection_kwargs = kwargs
        self.operator_kwargs = {}

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        sample = self.dataset[item]
        if self.injection < 0:
            sample = self.metadata_injector(sample, self.metadata, **self.injection_kwargs)
        sample = self.operator(sample, **self.operator_kwargs)
        if self.injection > 0:
            sample = self.metadata_injector(sample, self.metadata, **self.injection_kwargs)

        if self.transform:
            sample = self.transform(sample)
        return sample

    def operator(self, sample, **kwargs):
        return sample
