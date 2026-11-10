from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import get_cosine_schedule_with_warmup


class HuggingFaceCosineScheduler(LRScheduler):
    def __new__(
        cls,
        optimizer: Optimizer,
        num_warmup_steps: int = 0,
        num_training_steps: int = 1,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
        **kwargs,
    ):
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            last_epoch=last_epoch,
        )

