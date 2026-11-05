from .split_processor import SplitProcessor
from .coregistration_processor import CoregistrationProcessor


def exe_preprocess_command(
    parameters: dict,
    cls: str,
):
    run_cls = globals()[cls]
    return run_cls(parameters).launch()


__all__ = ["exe_preprocess_command"]

