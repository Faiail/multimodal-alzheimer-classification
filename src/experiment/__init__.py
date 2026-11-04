from .run import Run


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

