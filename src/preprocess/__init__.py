from .split_processor import SplitProcessor
from .coregistration_processor import CoregistrationProcessor
from .mri_atlas_registration_processor import MRIAtlasRegistrationProcessor
from .pet_atlas_registration_processor import PETAtlasRegistrationProcessor
from .multimodal_tensor_processor import MultimodalTensorProcessor


def exe_preprocess_command(
    parameters: dict,
    cls: str,
):
    run_cls = globals()[cls]
    return run_cls(parameters).launch()


__all__ = [
    "SplitProcessor",
    "CoregistrationProcessor",
    "MRIAtlasRegistrationProcessor",
    "PETAtlasRegistrationProcessor",
    "MultimodalTensorProcessor",
    "exe_preprocess_command",
]

