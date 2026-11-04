def exe_preprocess_command(
    parameters: dict,
    discover: bool = False,
    dicom: bool = False,
    smri: bool = False,
    pet: bool = False,
    features: bool = False,
    coords: bool = False,
    clinical: bool = False,
    cn_ad: bool = False,
    multiclass: bool = False,
    zhang: bool = False,
    ontology: bool = False,
    visualize: bool = False,
):
    if discover:
        from .discover_data import DicomDiscoveryProcessor

        return DicomDiscoveryProcessor(parameters).launch()
    if dicom:
        from .convert_dicom_to_nifti import DicomToNiftiProcessor

        return DicomToNiftiProcessor(parameters).launch()
    if smri:
        from .batch_process_sMRI import SMRIProcessor

        return SMRIProcessor(parameters).launch()
    if pet:
        from .batch_process_pet import PETProcessor

        return PETProcessor(parameters).launch()
    if features:
        from .batch_extract_features import FeatureExtractorProcessor

        return FeatureExtractorProcessor(parameters).launch()
    if coords:
        from .extract_coords import AtlasCoordsExtractor

        return AtlasCoordsExtractor(parameters).launch()
    if clinical:
        from .create_clinical_dataset import ClinicalDatasetBuilder

        return ClinicalDatasetBuilder(parameters).launch()
    if cn_ad:
        from .create_cn_ad_dataset import CNADDatasetBuilder

        return CNADDatasetBuilder(parameters).launch()
    if multiclass:
        from .create_multiclass_cn_mci_ad_dataset import MulticlassDatasetBuilder

        return MulticlassDatasetBuilder(parameters).launch()
    if zhang:
        from .create_zhang_dataset import ZhangDatasetBuilder

        return ZhangDatasetBuilder(parameters).launch()
    if ontology:
        from .ont import (
            FMAKnowledgeGraphBuilder,
            FMAKnowledgeGraphValidator,
            FMAMappingBuilder,
        )

        ont_cfg = parameters.get("ontology", parameters)
        mode = ont_cfg.get("mode", "kg")
        if mode in {"map", "mapping", "lookup"}:
            return FMAMappingBuilder(parameters).launch()
        if mode in {"kg", "build", "graph"}:
            return FMAKnowledgeGraphBuilder(parameters).launch()
        if mode in {"validate", "check"}:
            return FMAKnowledgeGraphValidator(parameters).launch()
        raise ValueError(
            f"Unsupported ontology mode '{mode}'. Supported modes: 'mapping', 'kg', 'validate'."
        )
    if visualize:
        from src.visualization import run_visualization_task

        return run_visualization_task(parameters)

    raise ValueError("Invalid preprocess configuration: specify at least one flag.")


__all__ = ["exe_preprocess_command"]

