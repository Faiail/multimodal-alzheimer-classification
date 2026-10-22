from .grid_optimizer import GridOptimizer
from .optuna_optimizer import OptunaOptimizer


def optimize_fn(parameters: dict, cls: str):
    opt_cls = globals()[cls]
    return opt_cls(parameters).launch()


__all__ = ["GridOptimizer", "OptunaOptimizer", "optimize_fn"]

