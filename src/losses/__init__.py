from torch.nn import CrossEntropyLoss
from .focal_loss import FocalLoss
from .brain_gnn_loss import BrainGNNLoss


__all__ = ["CrossEntropyLoss", "FocalLoss", "BrainGNNLoss"]

