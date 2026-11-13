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
            dataset=test_dataset, **loader_parameters
        )

    def train_epoch(self, epoch):
        bar = self.get_bar(
            loader=self.train_loader,
            desc=f"{ParameterKeys.TRAIN} at epoch {epoch}/{self.num_epochs}",
        )
        cumulated_loss = 0.0
        self.model.train()
        for batch in bar:
            self.optimizer.zero_grad()
            gts = batch.pop(BatchKeys.GT).to(self.device)
            graphs = batch.pop(BatchKeys.GRAPH).to(self.device)
            logits, aux = self.model(
                x=graphs.x,
                pos_id=graphs.pos_id,
                edge_index=graphs.edge_index,
                edge_weight=graphs.edge_weight,
                batch=graphs.batch,
                return_aux=True,
            )
            loss = self.criterion(logits=logits, y=gts, aux=aux)["loss"]
            loss.backward()
            self.optimizer.step()
            self.metrics.update(self._format_preds_for_metrics(logits), gts.cpu())
            batch_loss = loss.detach().cpu().item()
            cumulated_loss += batch_loss
            self.update_bar(bar=bar, loss=batch_loss)
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
    def validate_epoch(self, epoch):
        bar = self.get_bar(
            loader=self.val_loader,
            desc=f"{ParameterKeys.VAL} at epoch {epoch}/{self.num_epochs}",
        )
        cumulated_loss = 0.0
        self.model.eval()
        for batch in bar:
            gts = batch.pop(BatchKeys.GT).to(self.device)
            graphs = batch.pop(BatchKeys.GRAPH).to(self.device)
            logits, aux = self.model(
                x=graphs.x,
                pos_id=graphs.pos_id,
                edge_index=graphs.edge_index,
                edge_weight=graphs.edge_weight,
                batch=graphs.batch,
                return_aux=True,
            )
            loss = self.criterion(logits=logits, y=gts, aux=aux)["loss"]
            self.metrics.update(self._format_preds_for_metrics(logits), gts.cpu())
            batch_loss = loss.detach().cpu().item()
            cumulated_loss += batch_loss
            self.update_bar(bar=bar, loss=batch_loss)
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
            gts = batch.pop(BatchKeys.GT).to(self.device)
            graphs = batch.pop(BatchKeys.GRAPH).to(self.device)
            logits, aux = self.model(
                x=graphs.x,
                pos_id=graphs.pos_id,
                edge_index=graphs.edge_index,
                edge_weight=graphs.edge_weight,
                batch=graphs.batch,
                return_aux=True,
            )
            loss = self.criterion(logits=logits, y=gts, aux=aux)
            self.metrics.update(self._format_preds_for_metrics(logits), gts.cpu())
            batch_loss = loss.detach().cpu().item()
            cumulated_loss += batch_loss
            self.update_bar(bar=bar, loss=batch_loss)
        cumulated_loss /= len(self.val_loader)
        metric_dict = self.metrics.compute()
        self.metrics.reset()
        self.print_metrics(phase=ParameterKeys.TEST, metrics=metric_dict)
        self.save_metrics(metrics=metric_dict, loss=cumulated_loss)
