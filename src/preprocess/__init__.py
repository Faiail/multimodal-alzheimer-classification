from .split_processor import SplitProcessor
from .coregistration_processor import CoregistrationProcessor
from .mri_atlas_registration_processor import MRIAtlasRegistrationProcessor
from .pet_atlas_registration_processor import PETAtlasRegistrationProcessor


def exe_preprocess_command(
    parameters: dict,
    cls: str,
):
    run_cls = globals()[cls]
    return run_cls(parameters).launch()


__all__ = ["exe_preprocess_command"]

