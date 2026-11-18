from .population_run import PopulationRun
import src.data as data_pkg
from src.utils import ParameterKeys
from src.data.utils import BatchKeys
import torch


class MMGNNRun(PopulationRun):
    def _init_loaders(self):
        dataset_parameters = self.parameters.get(ParameterKeys.DATA)
        dataset_name = dataset_parameters.get(ParameterKeys.NAME)
        dataset_config = dataset_parameters.get(ParameterKeys.CFG, {})
        self.dataset = data_pkg.__dict__[dataset_name](
            **dataset_config, fold_id=self.fold_id
        )
        dataset = self.dataset()
        self.mri_graph = dataset.get(BatchKeys.MRI)
        self.pet_graph = dataset.get(BatchKeys.PET)
        self.mri_mask = dataset.get(BatchKeys.HAS_MRI)
        self.pet_mask = dataset.get(BatchKeys.HAS_PET)
        self.labels = dataset.get(BatchKeys.GT)

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        train_mask = self.dataset.get_mask(split=ParameterKeys.TRAIN).to(self.device)
        preds = self.model(
            mri_graph=self.mri_graph.to(self.device),
            pet_graph=self.pet_graph.to(self.device),
            mri_mask=self.mri_mask.to(self.device),
            pet_mask=self.pet_mask.to(self.device),
        )
        preds = preds[train_mask]
        gts = self.labels[train_mask].to(self.device)
        loss = self.criterion(preds, gts)
        loss.backward()
        self.optimizer.step()
        self.metrics.update(self._format_preds_for_metrics(preds), gts.cpu())
        cumulated_loss = loss.detach().cpu().item()
        self.schedule(phase=ParameterKeys.TRAIN)
        metric_dict = self.metrics.compute()
        self.metrics.reset()
        self.print_stats(
            epoch=epoch,
            phase=ParameterKeys.TRAIN,
            cumulated_loss=cumulated_loss,
            metrics=metric_dict,
        )

    @torch.no_grad()
    def validate_epoch(self, epoch):
        self.model.eval()
        val_mask = self.dataset.get_mask(split=ParameterKeys.VAL).to(self.device)
        preds = self.model(
            mri_graph=self.mri_graph.to(self.device),
            pet_graph=self.pet_graph.to(self.device),
            mri_mask=self.mri_mask.to(self.device),
            pet_mask=self.pet_mask.to(self.device),
        )
        preds = preds[val_mask]
        gts = self.labels[val_mask].to(self.device)
        loss = self.criterion(preds, gts)
        self.metrics.update(self._format_preds_for_metrics(preds), gts.cpu())
        cumulated_loss = loss.detach().cpu().item()
        self.schedule(phase=ParameterKeys.VAL, loss=cumulated_loss)
        metric_dict = self.metrics.compute()
        self.metrics.reset()
        self.print_stats(
            epoch=epoch,
            phase=ParameterKeys.VAL,
            cumulated_loss=cumulated_loss,
            metrics=metric_dict,
        )

    @torch.no_grad()
    def test(self):
        self.model.load_state_dict(torch.load(f"{self.out_dir}/{self.early_stop.path}"))
        self.model = self.model.to(self.device)
        self.model.eval()
        test_mask = self.dataset.get_mask(split=ParameterKeys.TEST).to(self.device)
        preds = self.model(
            mri_graph=self.mri_graph.to(self.device),
            pet_graph=self.pet_graph.to(self.device),
            mri_mask=self.mri_mask.to(self.device),
            pet_mask=self.pet_mask.to(self.device),
        )
        preds = preds[test_mask]
        gts = self.labels[test_mask].to(self.device)
        loss = self.criterion(preds, gts)
        self.metrics.update(self._format_preds_for_metrics(preds), gts.cpu())
        cumulated_loss = loss.detach().cpu().item()
        metric_dict = self.metrics.compute()
        self.metrics.reset()   
        self.metrics.reset()
        self.print_metrics(phase=ParameterKeys.TEST, metrics=metric_dict)
        self.save_metrics(metrics=metric_dict, loss=cumulated_loss)             