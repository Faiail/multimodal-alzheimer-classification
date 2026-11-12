import torch
from .run import Run
from src.utils import ParameterKeys
import src.metrics as metric_pkg
from src.data.utils import BatchKeys
from torchmetrics import MetricCollection


class VisualRun(Run):
    def print_metrics(self, phase: str, metrics=None):
        for k, v in metrics.items():
            print(f"{phase} {k}: {v:.4f}")

    def train_epoch(self, epoch: int):
        bar = self.get_bar(
            loader=self.train_loader,
            desc=f"{ParameterKeys.TRAIN} at epoch {epoch}/{self.num_epochs}",
        )
        cumulated_loss = 0.0
        self.model.train()
        for batch in bar:
            self.optimizer.zero_grad()
            gts = batch.get(BatchKeys.GT).to(self.device)
            imgs = batch.get(BatchKeys.IMAGE).to(self.device)
            preds = self.model(imgs)
            loss = self.criterion(preds, gts)
            loss.backward()
            self.optimizer.step()
            self.metrics.update(self._format_preds_for_metrics(preds), gts.cpu())
            batch_loss = loss.detach().cpu().item()
            cumulated_loss += batch_loss
            self.update_bar(bar, loss=batch_loss)
            self.schedule(phase=ParameterKeys.TRAIN)
        cumulated_loss /= len(self.train_loader)
        metric_dict = self.metrics.compute()
        self.metrics.reset()
        self.print_stats(
            epoch=epoch,
            phase=ParameterKeys.TRAIN,
            cumulated_loss=cumulated_loss,
            metrics=metric_dict,
        )

    @torch.no_grad()
    def validate_epoch(self, epoch: int):
        bar = self.get_bar(
            loader=self.val_loader,
            desc=f"{ParameterKeys.VAL} at epoch {epoch}/{self.num_epochs}",
        )
        cumulated_loss = 0.0
        self.model.eval()
        for batch in bar:
            gts = batch.get(BatchKeys.GT).to(self.device)
            imgs = batch.get(BatchKeys.IMAGE).to(self.device)
            preds = self.model(imgs)
            loss = self.criterion(preds, gts)
            self.metrics.update(self._format_preds_for_metrics(preds), gts.cpu())
            batch_loss = loss.detach().cpu().item()
            cumulated_loss += batch_loss
            self.update_bar(bar, loss=batch_loss)
        cumulated_loss /= len(self.val_loader)
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
        if not self.test_loader:
            return {}
        self.model.load_state_dict(torch.load(f"{self.out_dir}/{self.early_stop.path}"))
        self.model = self.model.to(self.device)
        bar = self.get_bar(
            loader=self.test_loader,
            desc=f"{ParameterKeys.TEST}",
        )
        cumulated_loss = 0.0
        self.model.eval()
        for batch in bar:
            gts = batch.get(BatchKeys.GT).to(self.device)
            imgs = batch.get(BatchKeys.IMAGE).to(self.device)
            preds = self.model(imgs)
            loss = self.criterion(preds, gts)
            self.metrics.update(self._format_preds_for_metrics(preds), gts.cpu())
            batch_loss = loss.detach().cpu().item()
            cumulated_loss += batch_loss
            self.update_bar(bar, loss=batch_loss)
        cumulated_loss /= len(self.test_loader)
        metric_dict = self.metrics.compute()
        self.metrics.reset()
        self.print_metrics(phase=ParameterKeys.TEST, metrics=metric_dict)
        self.save_metrics(metrics=metric_dict, loss=cumulated_loss)

