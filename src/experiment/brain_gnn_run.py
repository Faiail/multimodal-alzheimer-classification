from .run import Run
from src.data.utils import BatchKeys
from src.data import BrainGraphLoader
from src.utils import ParameterKeys
import torch


class BrainGNNRun(Run):
    def _init_loaders(self):
        dataset_parameters = self.parameters.get(ParameterKeys.DATA)
        train_dataset, val_dataset, test_dataset = (
            self._init_dataset(dataset_parameters.get(ParameterKeys.TRAIN)),
            self._init_dataset(dataset_parameters.get(ParameterKeys.VAL)),
            self._init_dataset(
                dataset_parameters.get(ParameterKeys.TEST), is_test=True
            ),
        )
        loader_parameters = self.parameters.get(ParameterKeys.LOADER)
        self.train_loader = BrainGraphLoader(dataset=train_dataset, **loader_parameters)
        self.val_loader = BrainGraphLoader(dataset=val_dataset, **loader_parameters)
        self.test_loader = BrainGraphLoader(
            dataset=self.test_loader, **loader_parameters
        )

    def train_epoch(self, epoch):
        pass

    @torch.no_grad()
    def validate_epoch(self, epoch):
        pass

    @torch.no_grad()
    def test(self):
        pass