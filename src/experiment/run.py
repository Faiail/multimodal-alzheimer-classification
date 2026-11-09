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
from src.utils.general import set_seed, save_json


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
        self.fold_id = general_parameters.get(ParameterKeys.FOLD)
        os.makedirs(f"{self.out_dir}/{self.fold_id}", exist_ok=True)
        device_str = general_parameters.get(ParameterKeys.DEVICE, "cpu")
        if str(device_str).startswith("cuda") and not torch.cuda.is_available():
            device_str = "cpu"
        self.device = device_str
        self.pbar = general_parameters.get(ParameterKeys.PBAR, False)
        self.num_epochs = general_parameters.get(ParameterKeys.NUM_EPOCHS, 1)
        self.seed = general_parameters.get(ParameterKeys.SEED, None)
        self.set_seed()
        self.run_state_fname = f"{self.out_dir}/run_state.pt"
        self.run_state = (
            torch.load(
                self.run_state_fname, map_location=self.device, weights_only=False
            )
            if os.path.exists(self.run_state_fname)
            else {}
        )
        self.start_epoch = self.run_state.get(ParameterKeys.EPOCH, 0) + 1

    def set_seed(self):
        if self.seed is not None:
            set_seed(self.seed)

    def _init_loaders(self):
        dataset_parameters = self.parameters.get(ParameterKeys.DATA)
        train_dataset, val_dataset, test_dataset = (
            self._init_dataset(dataset_parameters.get(ParameterKeys.TRAIN)),
            self._init_dataset(dataset_parameters.get(ParameterKeys.VAL)),
            self._init_dataset(
                dataset_parameters.get(ParameterKeys.TEST), is_test=True
            ),
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

    def _init_dataset(self, dataset_parameters: dict, is_test: bool = False) -> Dataset:
        dataset_name = dataset_parameters.get(ParameterKeys.NAME)
        dataset_cfg = dataset_parameters.get(ParameterKeys.CFG, {})
        return datasets.__dict__[dataset_name](
            **dataset_cfg, fold_id=(self.fold_id if not is_test else None)
        )

    def _init_model(self):
        model_parameters = self.parameters.get(ParameterKeys.MODEL)
        model_name = model_parameters.get(ParameterKeys.NAME)
        model_config = deepcopy(model_parameters.get(ParameterKeys.CFG, {}))
        self.model = models.__dict__[model_name](**model_config)
        self.model = self.model.to(self.device)
        if self.run_state.get(ParameterKeys.MODEL) is not None:
            print("Restoring model weights from previous run")
            self.model.load_state_dict(self.run_state.get(ParameterKeys.MODEL))

    def _init_metrics(self):
        raise NotImplementedError()

    def print_stats(
        self,
        epoch: int,
        phase: str,
        metrics: dict = {},
        cumulated_loss: float = 0.0,
    ):
        print(f"Epoch {epoch}/{self.num_epochs}: {phase} loss: {cumulated_loss:.4f}")
        self.print_metrics(phase=phase, metrics=metrics)

    def print_metrics(self, phase: str, metrics: dict = None):
        raise NotImplementedError()

    def _init_optimizer(self):
        optimizer_parameters = self.parameters.get(ParameterKeys.OPTIMIZER)
        optimizer_name = optimizer_parameters.get(ParameterKeys.NAME)
        optimizer_cfg = optimizer_parameters.get(ParameterKeys.CFG, {})
        self.optimizer = optimizers.__dict__[optimizer_name](
            params=self.model.parameters(), **optimizer_cfg
        )
        if self.run_state.get(ParameterKeys.OPTIMIZER) is not None:
            print("Restoring optimizer state from previous run")
            self.optimizer.load_state_dict(self.run_state.get(ParameterKeys.OPTIMIZER))

    def _init_criterion(self):
        criterion_parameters = self.parameters.get(ParameterKeys.CRITERION, {})
        criterion_name = criterion_parameters.get(ParameterKeys.NAME)
        criterion_cfg = criterion_parameters.get(ParameterKeys.CFG, {})
        self.criterion = losses.__dict__[criterion_name](**criterion_cfg)
        if self.run_state.get(ParameterKeys.CRITERION) is not None:
            print("Restoring criterion state from previous run")
            self.criterion.load_state_dict(self.run_state.get(ParameterKeys.CRITERION))

    def _init_early_stop(self):
        early_stop_parameters = self.parameters.get(ParameterKeys.EARLY_STOP, None)
        if early_stop_parameters is None:
            self.early_stop = None
            return
        early_stop_name = early_stop_parameters.get(ParameterKeys.NAME)
        early_stop_cfg = deepcopy(early_stop_parameters.get(ParameterKeys.CFG, {}))
        early_stop_cfg.setdefault("out_dir", self.out_dir)
        self.early_stop = early_stops.__dict__[early_stop_name](**early_stop_cfg)
        if self.run_state.get(ParameterKeys.EARLY_STOP) is not None:
            print("Restoring early stop state from previous run")
            self.early_stop.__dict__.update(
                self.run_state.get(ParameterKeys.EARLY_STOP)
            )
            self.trigger = self.early_stop.early_stop

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
        if self.run_state.get(ParameterKeys.SCHEDULER) is not None:
            print("Restoring scheduler state from previous run")
            self.scheduler.load_state_dict(self.run_state.get(ParameterKeys.SCHEDULER))

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

    def save_run_state(self, epoch: int):
        early_stop_state = (
            {
                "counter": self.early_stop.counter,
                "best_score": self.early_stop.best_score,
                "early_stop": self.early_stop.early_stop,
                "val_loss_min": self.early_stop.val_loss_min,
            }
            if self.early_stop is not None
            else None
        )
        self.run_state[ParameterKeys.EPOCH] = epoch
        self.run_state[ParameterKeys.MODEL] = self.model.state_dict()
        self.run_state[ParameterKeys.OPTIMIZER] = self.optimizer.state_dict()
        self.run_state[ParameterKeys.CRITERION] = self.criterion.state_dict()
        self.run_state[ParameterKeys.EARLY_STOP] = early_stop_state
        if self.scheduler is not None:
            self.run_state[ParameterKeys.SCHEDULER] = self.scheduler.state_dict()
        torch.save(self.run_state, self.run_state_fname)

    def save_metrics(self, metrics: dict = {}, loss=None):
        metric_dict = deepcopy(metrics)
        metric_dict.update(dict(loss=f"{loss:.4f}") if loss is not None else dict())
        save_json(f"{self.out_dir}/test_metrics.json", metric_dict)

    def train_epoch(self, epoch: int):
        raise NotImplementedError()

    def validate_epoch(self, epoch: int):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()

    def launch(self):
        print("Start experiment!")
        self.model = self.model.to(self.device)
        if self.start_epoch > self.num_epochs or self.trigger:
            print("Training was already completed. Proceeding directly to test...")
            return self.test()
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self.train_epoch(epoch=epoch)
            self.validate_epoch(epoch=epoch)
            self.save_run_state(epoch=epoch)
            if self.trigger:
                print(f"Early stopping at epoch {epoch}/{self.num_epochs}")
                break
        return self.test()
