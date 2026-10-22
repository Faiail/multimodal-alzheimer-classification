import os
import numpy as np
import torch


class EarlyStopping:
    """Early stops training if monitored quantity doesn't improve after a given patience."""

    def __init__(
        self,
        patience: int = 15,
        verbose: bool = False,
        delta: float = 0.0,
        out_dir: str = "./",
        mode: str = "min",
        filename: str = "model.pt",
        trace_func=print,
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_best = np.inf if mode == "min" else -np.inf
        self.delta = delta
        self.out_dir = out_dir
        self.mode = mode
        self.filename = filename
        self.trace_func = trace_func

    def __call__(self, val_metric: float, model: torch.nn.Module, extra_payload: dict = None):
        score = -val_metric if self.mode == "min" else val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model, extra_payload)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model, extra_payload)
            self.counter = 0

    def save_checkpoint(self, val_metric: float, model: torch.nn.Module, extra_payload: dict = None):
        os.makedirs(self.out_dir, exist_ok=True)
        if self.verbose:
            self.trace_func(
                f"Monitored metric improved ({self.val_best:.6f} --> {val_metric:.6f}). Saving model ..."
            )
        save_path = os.path.join(self.out_dir, self.filename)
        if extra_payload is not None:
            payload = {"state_dict": model.state_dict(), **extra_payload}
            torch.save(payload, save_path)
        else:
            torch.save(model.state_dict(), save_path)
        self.val_best = val_metric

