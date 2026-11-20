from .run import Run
from src.utils import ParameterKeys
from src.data.utils import BatchKeys
from safetensors.torch import save_file
import torch


class LassoNetRun(Run):
    def _init_metrics(self):
        pass

    def _init_criterion(self):
        pass

    def save_run_state(self, epoch):
        early_stop_state = (
            {
                "counter": self.early_stop.counter,
                "best_score": self.early_stop.best_score,
                "early_stop": self.early_stop.early_stop,
                "val_best": self.early_stop.val_best,
            }
            if self.early_stop is not None
            else None
        )
        self.run_state[ParameterKeys.EPOCH] = epoch
        self.run_state[ParameterKeys.MODEL] = self.model.state_dict()
        self.run_state[ParameterKeys.OPTIMIZER] = self.optimizer.state_dict()
        self.run_state[ParameterKeys.EARLY_STOP] = early_stop_state
        if self.scheduler is not None:
            self.run_state[ParameterKeys.SCHEDULER] = self.scheduler.state_dict()
        torch.save(self.run_state, self.run_state_fname)

    def train_epoch(self, epoch):
        bar = self.get_bar(
            loader=self.train_loader,
            desc=f"{ParameterKeys.TRAIN} at epoch {epoch}/{self.num_epochs}",
        )
        cumulated_loss = 0.0
        self.model.train()
        for batch in bar:
            self.optimizer.zero_grad()
            gts = batch.get(BatchKeys.GT).to(self.device)
            x = batch.get(BatchKeys.X).to(self.device)
            preds = self.model(x)
            loss, ce, reg = self.model.compute_loss(preds, gts)
            loss.backward()
            self.optimizer.step()
            self.model.project_hierarchical_constraint()
            batch_loss = loss.detach().cpu().item()
            cumulated_loss += batch_loss
            self.update_bar(bar, loss=batch_loss)
            self.schedule(phase=ParameterKeys.TRAIN)
        cumulated_loss /= len(self.train_loader)
        self.print_stats(
            epoch=epoch,
            phase=ParameterKeys.TRAIN,
            cumulated_loss=cumulated_loss,
        )

    @torch.no_grad()
    def validate_epoch(self, epoch):
        bar = self.get_bar(
            loader=self.val_loader,
            desc=f"{ParameterKeys.VAL} at epoch {epoch}/{self.num_epochs}",
        )
        cumulated_loss = 0.0
        self.model.eval()
        for batch in bar:
            gts = batch.get(BatchKeys.GT).to(self.device)
            x = batch.get(BatchKeys.X).to(self.device)
            preds = self.model(x)
            loss, ce, reg = self.model.compute_loss(preds, gts)
            batch_loss = loss.detach().cpu().item()
            cumulated_loss += batch_loss
            self.update_bar(bar, loss=batch_loss)
        self.schedule(phase=ParameterKeys.VAL, loss=cumulated_loss)
        cumulated_loss /= len(self.val_loader)
        self.trigger = self.early_stop_callback(monitored_value=cumulated_loss)
        self.print_stats(
            epoch=epoch,
            phase=ParameterKeys.VAL,
            cumulated_loss=cumulated_loss,
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
        self.model.eval()
        for batch in bar:
            gts = batch.get(BatchKeys.GT).to(self.device)
            x = batch.get(BatchKeys.X).to(self.device)
            preds = self.model(x)
            loss, ce, reg = self.model.compute_loss(preds, gts)
            batch_loss = loss.detach().cpu().item()
            self.update_bar(bar, loss=batch_loss)
        top_indices, _, _ = self.model.select_top_features()
        save_file({BatchKeys.ROI: top_indices}, f"{self.out_dir}/top_indices.safetensors")
