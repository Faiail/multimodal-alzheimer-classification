from .run import Run
from .visual_run import VisualRun
from .population_run import PopulationRun
from .brain_gnn_run import BrainGNNRun
from .hi_gnn_run import HiGNNRun
from .mm_gnn_run import MMGNNRun


def exe_experiment(parameters: dict, cls: str):
    run_cls = globals()[cls]
    return run_cls(parameters).launch()


def exe_test(parameters: dict, cls: str):
    run_cls = globals()[cls]
    return run_cls(parameters).test()


__all__ = [
    "Run",
    "exe_experiment",
]
