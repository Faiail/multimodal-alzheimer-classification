import os
import torch
from src.data.batch_utils import move_graph_batch_to_device
from src.experiment.run import Run
from src.metrics import ClassificationMetrics
from src.utils import ParameterKeys, save_json


class GNNRun(Run):
    """Standard single-split / single-fold GNN experiment runner."""

    def _init_metrics(self):
        model_cfg = self.parameters.get(ParameterKeys.MODEL, {}).get(ParameterKeys.CFG, {})
        self.num_classes = int(model_cfg.get("n_classes", model_cfg.get("out_classes", 2)))
        self.graph_type = str(model_cfg.get("graph_type", "static"))
        self.has_clinical = int(model_cfg.get("n_clinical_features", 0)) > 0
        self.metrics_tracker = ClassificationMetrics(num_classes=self.num_classes)

    def print_metrics(self, metrics: dict = None):
        if not metrics:
            return
        summary_keys = ["accuracy", "balanced_accuracy", "f1", "auc", "mcc"]
        parts = [f"{k}: {metrics[k]:.4f}" for k in summary_keys if k in metrics]
        print("  " + " | ".join(parts))

    def _forward_batch(self, batch_data):
        if self.has_clinical:
            x_graph, adj, x_clin, y = move_graph_batch_to_device(
                batch_data, self.device, graph_type=self.graph_type, has_clinical=True
            )
        else:
            x_graph, adj, y = move_graph_batch_to_device(
                batch_data, self.device, graph_type=self.graph_type, has_clinical=False
            )
            x_clin = None
        logits = self.model(x_graph, adj, x_clin)
        loss = self.criterion(logits, y)
        return logits, y, loss

    def train_epoch(self, epoch: int):
        self.model.train()
        self.metrics_tracker.reset()
        total_loss = 0.0
        bar = self.get_bar(self.train_loader, desc=f"Train Epoch {epoch}")

        for batch_data in bar:
            self.optimizer.zero_grad()
            logits, y, loss = self._forward_batch(batch_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.schedule(ParameterKeys.TRAIN)

            total_loss += float(loss.item())
            probs = torch.softmax(logits.detach(), dim=1).cpu().numpy()
            preds = torch.argmax(logits.detach(), dim=1).cpu().numpy()
            self.metrics_tracker.update(probs, preds, y.detach().cpu().numpy())
            self.update_bar(bar, float(loss.item()))

        avg_loss = total_loss / max(1, len(self.train_loader))
        train_metrics = self.metrics_tracker.compute()
        self.print_stats(epoch, "train", train_metrics, avg_loss)
        return avg_loss, train_metrics

    def validate_epoch(self, epoch: int):
        self.model.eval()
        self.metrics_tracker.reset()
        total_loss = 0.0
        bar = self.get_bar(self.val_loader, desc=f"Val Epoch {epoch}")

        with torch.no_grad():
            for batch_data in bar:
                logits, y, loss = self._forward_batch(batch_data)
                total_loss += float(loss.item())
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                self.metrics_tracker.update(probs, preds, y.cpu().numpy())
                self.update_bar(bar, float(loss.item()))

        avg_loss = total_loss / max(1, len(self.val_loader))
        val_metrics = self.metrics_tracker.compute()
        self.schedule(ParameterKeys.VAL)
        self.print_stats(epoch, "val", val_metrics, avg_loss)

        monitored = avg_loss
        if self.early_stop is not None and getattr(self.early_stop, "mode", "min") == "max":
            monitored = val_metrics.get("balanced_accuracy", 0.0)
        self.trigger = self.early_stop_callback(monitored)
        return avg_loss, val_metrics

    def test(self):
        ckpt_path = os.path.join(self.out_dir, "model.pt")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                self.model.load_state_dict(state["state_dict"])
            else:
                self.model.load_state_dict(state)

        self.model.eval()
        self.metrics_tracker.reset()
        total_loss = 0.0

        with torch.no_grad():
            for batch_data in self.test_loader:
                logits, y, loss = self._forward_batch(batch_data)
                total_loss += float(loss.item())
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                self.metrics_tracker.update(probs, preds, y.cpu().numpy())

        avg_loss = total_loss / max(1, len(self.test_loader))
        test_metrics = self.metrics_tracker.compute()
        test_metrics["test_loss"] = avg_loss
        self.print_stats(self.num_epochs, "test", test_metrics, avg_loss)
        save_json(os.path.join(self.out_dir, "test_metrics.json"), test_metrics)
        return test_metrics

