from abc import ABC

import torch.utils.data as data


class DefaultDataset(data.Dataset, ABC):
    def __init__(self):
        super(DefaultDataset, self).__init__()

    def init_dataset(self):
        pass


class DefaultDataGenerator:
    def __init__(self, args):
        super(DefaultDataGenerator, self).__init__()
        self._args = args

    def download(self):
        raise NotImplementedError

    def preprocess(self):
        raise NotImplementedError

    def get_train_dataset(self):
        raise NotImplementedError

    def get_valid_dataset(self):
        raise NotImplementedError

    def get_valid_metric_dataset(self):
        raise NotImplementedError

    def get_train_loader(self, dataset, batch_size):
        raise NotImplementedError

    def get_valid_loader(self, dataset, batch_size):
        raise NotImplementedError

    def get_valid_metric_loader(self, dataset, batch_size):
        raise NotImplementedError
