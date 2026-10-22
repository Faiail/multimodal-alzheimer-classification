import os
from copy import deepcopy

import torch
import torch.optim as optimizers
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import src.data as datasets
import src.early_stop as early_stops
import src.losses as losses
import src.models as models
import src.scheduler as schedulers
from src.utils import ParameterKeys
from src.utils.general import set_seed


class Run:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.trigger = False
        self.init()

    def init(self):
        print("Init general parameters")
        self._init_general()
        print("Init dataloaders")
        self._init_loaders()
        print("Init model")
        self._init_model()
        print("Init metrics")
        self._init_metrics()
        print("Init optimizer")
        self._init_optimizer()
        print("Init criterion")
        self._init_criterion()
        print("Init early stop")
        self._init_early_stop()
        print("Init scheduler")
        self._init_scheduler()

    def _init_general(self):
        general_parameters = self.parameters.get(ParameterKeys.GENERAL, {})
        self.out_dir = general_parameters.get(ParameterKeys.OUT_DIR, "models/default")
        os.makedirs(self.out_dir, exist_ok=True)
        device_str = general_parameters.get(ParameterKeys.DEVICE, "cpu")
        if str(device_str).startswith("cuda") and not torch.cuda.is_available():
            device_str = "cpu"
        self.device = device_str
        self.pbar = general_parameters.get(ParameterKeys.PBAR, False)
        self.num_epochs = general_parameters.get(ParameterKeys.NUM_EPOCHS, 1)
        self.seed = general_parameters.get(ParameterKeys.SEED, None)
        self.set_seed()

    def set_seed(self):
        if self.seed is not None:
            set_seed(self.seed)

    def _init_loaders(self):
        dataset_parameters = self.parameters.get(ParameterKeys.DATA)
        train_dataset, val_dataset, test_dataset = (
            self._init_dataset(dataset_parameters.get(ParameterKeys.TRAIN)),
            self._init_dataset(dataset_parameters.get(ParameterKeys.VAL)),
            self._init_dataset(dataset_parameters.get(ParameterKeys.TEST)),
        )
        loader_params = deepcopy(self.parameters.get(ParameterKeys.LOADER, {}))
        val_loader_params = deepcopy(loader_params)
        val_loader_params["shuffle"] = False
        val_loader_params["drop_last"] = False

        self.train_loader = DataLoader(
            dataset=train_dataset,
            collate_fn=(
                train_dataset.collate_fn
                if hasattr(train_dataset, "collate_fn")
                else None
            ),
            **loader_params,
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            collate_fn=(
                val_dataset.collate_fn if hasattr(val_dataset, "collate_fn") else None
            ),
            **val_loader_params,
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            collate_fn=(
                test_dataset.collate_fn if hasattr(test_dataset, "collate_fn") else None
            ),
            **val_loader_params,
        )

    def _init_dataset(self, dataset_parameters: dict) -> Dataset:
        dataset_name = dataset_parameters.get(ParameterKeys.NAME)
        dataset_cfg = dataset_parameters.get(ParameterKeys.CFG, {})
        return datasets.__dict__[dataset_name](**dataset_cfg)

    def _init_model(self):
        model_parameters = self.parameters.get(ParameterKeys.MODEL)
        model_name = model_parameters.get(ParameterKeys.NAME)
        model_config = deepcopy(model_parameters.get(ParameterKeys.CFG, {}))
        self.model = models.__dict__[model_name](**model_config)
        self.model = self.model.to(self.device)

    def _init_metrics(self):
        raise NotImplementedError()

    def print_stats(
        self,
        epoch: int,
        phase: str,
        metrics: dict = None,
        cumulated_loss: float = 0.0,
    ):
        print(f"Epoch {epoch}/{self.num_epochs}: {phase} loss: {cumulated_loss:.4f}")
        self.print_metrics(metrics or {})

    def print_metrics(self, metrics: dict = None):
        raise NotImplementedError()

    def _init_optimizer(self):
        optimizer_parameters = self.parameters.get(ParameterKeys.OPTIMIZER)
        optimizer_name = optimizer_parameters.get(ParameterKeys.NAME)
        optimizer_cfg = optimizer_parameters.get(ParameterKeys.CFG, {})
        self.optimizer = optimizers.__dict__[optimizer_name](
            params=self.model.parameters(), **optimizer_cfg
        )

    def _init_criterion(self):
        criterion_parameters = self.parameters.get(ParameterKeys.CRITERION, {})
        criterion_name = criterion_parameters.get(ParameterKeys.NAME)
        criterion_cfg = criterion_parameters.get(ParameterKeys.CFG, {})
        self.criterion = losses.__dict__[criterion_name](**criterion_cfg)

    def _init_early_stop(self):
        early_stop_parameters = self.parameters.get(ParameterKeys.EARLY_STOP, None)
        if early_stop_parameters is None:
            self.early_stop = None
            return
        early_stop_name = early_stop_parameters.get(ParameterKeys.NAME)
        early_stop_cfg = deepcopy(early_stop_parameters.get(ParameterKeys.CFG, {}))
        early_stop_cfg.setdefault("out_dir", self.out_dir)
        self.early_stop = early_stops.__dict__[early_stop_name](**early_stop_cfg)

    def _init_scheduler(self):
        scheduler_parameters = self.parameters.get(ParameterKeys.SCHEDULER, None)
        if scheduler_parameters is None:
            self.scheduler = None
            return
        scheduler_name = scheduler_parameters.get(ParameterKeys.NAME)
        scheduler_cfg = deepcopy(scheduler_parameters.get(ParameterKeys.CFG, {}))
        if scheduler_name == ParameterKeys.HF_COSINE_SCHEDULER:
            scheduler_cfg = self._init_cosine_scheduler_params(params=scheduler_cfg)
            self.cosine_sched = True
        else:
            self.cosine_sched = False
        self.scheduler = schedulers.__dict__[scheduler_name](
            optimizer=self.optimizer, **scheduler_cfg
        )

    def _init_cosine_scheduler_params(self, params: dict) -> dict:
        total_steps = max(1, self.num_epochs * len(self.train_loader))
        params.update({"num_training_steps": total_steps})
        return params

    def schedule(self, phase, **kwargs):
        if self.scheduler is None:
            return
        if phase == ParameterKeys.VAL and self.cosine_sched:
            return
        if phase == ParameterKeys.TRAIN and not self.cosine_sched:
            return
        self.scheduler.step(**kwargs)

    def early_stop_callback(self, monitored_value: float) -> bool:
        if self.early_stop is None:
            return False
        self.early_stop(monitored_value, self.model)
        return self.early_stop.early_stop

    def get_bar(self, loader: DataLoader, desc: str = "", **kwargs):
        return (
            loader
            if not self.pbar
            else tqdm(
                loader,
                total=len(loader),
                desc=desc,
                **kwargs,
            )
        )

    def update_bar(self, bar, loss, **kwargs):
        if not self.pbar:
            return
        bar.set_postfix({ParameterKeys.CRITERION: loss, **kwargs})

    def train_epoch(self, epoch: int):
        raise NotImplementedError()

    def validate_epoch(self, epoch: int):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()

    def launch(self):
        print("Start experiment!")
        self.model = self.model.to(self.device)
        for epoch in range(1, self.num_epochs + 1):
            self.train_epoch(epoch=epoch)
            self.validate_epoch(epoch=epoch)
            if self.trigger:
                print(f"Early stopping at epoch {epoch}/{self.num_epochs}")
                break
        return self.test()

