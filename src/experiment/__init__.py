from .alzheimer_benchmark_run import AlzheimerBenchmarkRun
from .alzheimer_gnn_run import AlzheimerGNNRun
from .config_helper import VALID_GRAPH_MODES, build_model_and_data_configs
from .gnn_run import GNNRun
from .run import Run
from .smoke_test_run import SmokeTestRun


def exe_experiment(parameters: dict, cls: str):
    run_cls = globals()[cls]
    return run_cls(parameters).launch()


def exe_test(parameters: dict, cls: str):
    run_cls = globals()[cls]
    return run_cls(parameters).test()


__all__ = [
    "AlzheimerBenchmarkRun",
    "AlzheimerGNNRun",
    "GNNRun",
    "Run",
    "SmokeTestRun",
    "VALID_GRAPH_MODES",
    "build_model_and_data_configs",
    "exe_experiment",
    "exe_test",
]

