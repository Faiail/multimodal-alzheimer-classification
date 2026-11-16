from .run import Run
import src.data as data_pkg
from src.data import BiGraphLoader
import torch
from src.utils import ParameterKeys
from src.data.utils import BatchKeys


class HiGNNRun(Run):
    def _init_loaders(self) -> None:
        dataset_parameters = self.parameters.get(ParameterKeys.DATA)
        dataset_name = dataset_parameters.get(ParameterKeys.NAME)
        dataset_config = dataset_parameters.get(ParameterKeys.CFG, dict())
        self.dataset = data_pkg.__dict__[dataset_name](**dataset_config)
        loader_parameters = self.parameters.get(ParameterKeys.LOADER)
        self.dataloader = BiGraphLoader(dataset=self.dataset, **loader_parameters)
        population_topology = self.dataset.get_base_topology()
        self.pop_edge_index = population_topology[BatchKeys.EDGE_INDEX].to(self.device)
        self.pop_edge_weight = population_topology[BatchKeys.EDGE_WEIGHT].to(
            self.device
        )

    def _init_cosine_scheduler_params(self, params):
        total_steps = max(1, self.num_epochs)
        num_warmup_epochs = params.pop(ParameterKeys.NUM_WARMUP_EPOCHS, 1)
        params.update(
            {
                "num_training_steps": total_steps,
                "num_warmup_steps": num_warmup_epochs,
            }
        )
        return params

    def get_brain_embeddings(self, loader) -> tuple[torch.Tensor, torch.Tensor]:
        brain_embeddings = None
        labels = None
        for batch in loader:
            brain_graphs = batch.get(BatchKeys.GRAPH).to(self.device)
            batch_labels = batch.get(BatchKeys.GT)
            batch_brain_embeddings = self.model.encode_brains(
                brain_x=brain_graphs.x,
                brain_edge_index=brain_graphs.edge_index,
                brain_edge_weight=brain_graphs.edge_weight,
                brain_batch=brain_graphs.batch,
            )
            brain_embeddings = (
                batch_brain_embeddings
                if brain_embeddings is None
                else torch.cat([brain_embeddings, batch_brain_embeddings], dim=0)
            )
            labels = (
                batch_labels
                if labels is None
                else torch.cat([labels, batch_labels], dim=-1)
            )
        return brain_embeddings, labels

    def train_epoch(self, epoch: int) -> None:
        bar = self.get_bar(
            loader=self.dataloader,
            desc=f"{ParameterKeys.TRAIN} at epoch {epoch}/{self.num_epochs}",
        )
        cumulated_loss = 0.0
        train_mask = self.dataset.get_mask(split=ParameterKeys.TRAIN)
        self.model.train()
        self.optimizer.zero_grad()
        brain_embeddings = None
        labels = None
        brain_embeddings, labels = self.get_brain_embeddings(bar)
        pred = self.model(
            brain_embeddings=brain_embeddings,
            pop_edge_index=self.pop_edge_index,
            pop_edge_weight=self.pop_edge_weight,
        )[train_mask]
        labels = labels[train_mask].to(self.device)

        loss = self.criterion(pred, labels)
        loss.backward()
        self.optimizer.step()
        self.metrics.update(self._format_preds_for_metrics(pred), labels.cpu())
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
    def validate_epoch(self, epoch: int) -> None:
        bar = self.get_bar(
            loader=self.dataloader,
            desc=f"{ParameterKeys.VAL} at epoch {epoch}/{self.num_epochs}",
        )
        cumulated_loss = 0.0
        val_mask = self.dataset.get_mask(split=ParameterKeys.VAL)
        self.model.eval()
        brain_embeddings = None
        labels = None
        brain_embeddings, labels = self.get_brain_embeddings(bar)
        pred = self.model(
            brain_embeddings=brain_embeddings,
            pop_edge_index=self.pop_edge_index,
            pop_edge_weight=self.pop_edge_weight,
        )[val_mask]
        labels = labels[val_mask].to(self.device)

        loss = self.criterion(pred, labels)
        self.metrics.update(self._format_preds_for_metrics(pred), labels.cpu())
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
        bar = self.get_bar(
            loader=self.dataloader,
            desc=f"{ParameterKeys.TEST}",
        )
        cumulated_loss = 0.0
        test_mask = self.dataset.get_mask(split=ParameterKeys.TEST)
        self.model.eval()
        brain_embeddings = None
        labels = None
        brain_embeddings, labels = self.get_brain_embeddings(bar)
        pred = self.model(
            brain_embeddings=brain_embeddings,
            pop_edge_index=self.pop_edge_index,
            pop_edge_weight=self.pop_edge_weight,
        )[test_mask]
        labels = labels[test_mask].to(self.device)

        loss = self.criterion(pred, labels)
        self.metrics.update(self._format_preds_for_metrics(pred), labels.cpu())
        cumulated_loss = loss.detach().cpu().item()
        self.schedule(phase=ParameterKeys.VAL, loss=cumulated_loss)
        metric_dict = self.metrics.compute()
        self.metrics.reset()
        self.print_metrics(phase=ParameterKeys.TEST, metrics=metric_dict)
        self.save_metrics(metrics=metric_dict, loss=cumulated_loss)        
