from .run import Run
import src.data as data_pkg
from src.utils import ParameterKeys
import torch


class PopulationRun(Run):
    def _init_loaders(self):
        dataset_parameters = self.parameters.get(ParameterKeys.DATA)
        dataset_name = dataset_parameters.get(ParameterKeys.NAME)
        dataset_config = dataset_parameters.get(ParameterKeys.CFG, {})
        self.dataset = data_pkg.__dict__[dataset_name](
            **dataset_config, fold_id=self.fold_id
        )()

    def _init_cosine_scheduler_params(self, params):
        total_steps = max(1, self.num_epochs)
        num_warmup_epochs = params.pop(ParameterKeys.NUM_WARMUP_EPOCHS, 1)
        params.update({
            "num_training_steps": total_steps,
            "num_warmup_steps": num_warmup_epochs,
        })
        return params

    def train_epoch(self, epoch):
        self.dataset = self.dataset.to(self.device)
        mask = self.dataset.train_mask
        self.model.train()
        self.optimizer.zero_grad()
        preds = self.model(
            self.dataset.x, self.dataset.edge_index, self.dataset.edge_weight
        )
        preds = preds[mask]
        gts = self.dataset.y[mask]
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
        self.dataset = self.dataset.to(self.device)
        mask = self.dataset.val_mask
        preds = self.model(
            self.dataset.x, self.dataset.edge_index, self.dataset.edge_weight
        )
        preds = preds[mask]
        gts = self.dataset.y[mask]
        loss = self.criterion(preds, gts)
        self.metrics.update(self._format_preds_for_metrics(preds), gts.cpu())
        cumulated_loss = loss.detach().cpu().item()
        self.schedule(phase=ParameterKeys.VAL, loss=cumulated_loss)
        metric_dict = self.metrics.compute()
        self.metrics.reset()
        self.trigger = self.early_stop_callback(monitored_value=cumulated_loss)
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
        self.dataset = self.dataset.to(self.device)
        mask = self.dataset.test_mask
        preds = self.model(
            self.dataset.x, self.dataset.edge_index, self.dataset.edge_weight
        )
        preds = preds[mask]
        gts = self.dataset.y[mask]
        loss = self.criterion(preds, gts)
        self.metrics.update(self._format_preds_for_metrics(preds), gts.cpu())
        cumulated_loss = loss.detach().cpu().item()
        metric_dict = self.metrics.compute()
        self.metrics.reset()
        self.print_metrics(phase=ParameterKeys.TEST, metrics=metric_dict)
        self.save_metrics(metrics=metric_dict, loss=cumulated_loss)
