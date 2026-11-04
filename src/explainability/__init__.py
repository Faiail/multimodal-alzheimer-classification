
def exe_explain(parameters: dict, cls: str):
    run_cls = globals()[cls]
    return run_cls(parameters).launch()


__all__ = [
    "exe_explain",
]

