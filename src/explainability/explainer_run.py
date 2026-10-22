from pathlib import Path
from src.utils import ParameterKeys, resolve_data_path


class ExplainerRun:
    """CLI-compatible wrapper for single-sample checkpoint explanation."""

    def __init__(self, parameters: dict):
        self.parameters = parameters
        general = parameters.get(ParameterKeys.GENERAL, {})
        explainer_cfg = parameters.get(ParameterKeys.EXPLAINER, parameters)
        self.checkpoint = Path(resolve_data_path(explainer_cfg.get(ParameterKeys.CHECKPOINT)))
        self.sample_index = int(explainer_cfg.get("sample_index", 0))
        self.steps = int(explainer_cfg.get("ig_steps", explainer_cfg.get("steps", 64)))
        self.target = explainer_cfg.get("target_class", explainer_cfg.get("target"))
        self.device = str(general.get(ParameterKeys.DEVICE, explainer_cfg.get("device", "cpu")))
        self.output_dir = Path(
            general.get(ParameterKeys.OUT_DIR, explainer_cfg.get("output_dir", "reports/explanations"))
        )

    def launch(self):
        import sys
        from .explain_checkpoint import main as explain_single_main

        argv = [
            "explain_checkpoint",
            str(self.checkpoint),
            "--sample-index",
            str(self.sample_index),
            "--steps",
            str(self.steps),
            "--device",
            self.device,
            "--output-dir",
            str(self.output_dir),
        ]
        if self.target is not None:
            argv.extend(["--target", str(self.target)])
        orig_argv = sys.argv
        try:
            sys.argv = argv
            return explain_single_main()
        finally:
            sys.argv = orig_argv


class BatchExplainerRun:
    """CLI-compatible wrapper for batch cohort checkpoint explanation."""

    def __init__(self, parameters: dict):
        self.parameters = parameters
        general = parameters.get(ParameterKeys.GENERAL, {})
        explainer_cfg = parameters.get(ParameterKeys.EXPLAINER, parameters)
        self.checkpoints = [
            str(resolve_data_path(p))
            for p in explainer_cfg.get(ParameterKeys.CHECKPOINTS, [])
        ]
        self.per_class = int(explainer_cfg.get("per_class", explainer_cfg.get("max_samples", 10)))
        self.steps = int(explainer_cfg.get("ig_steps", explainer_cfg.get("steps", 32)))
        self.device = str(general.get(ParameterKeys.DEVICE, explainer_cfg.get("device", "cpu")))
        self.output_dir = Path(
            general.get(ParameterKeys.OUT_DIR, explainer_cfg.get("output_dir", "reports/explanations"))
        )

    def launch(self):
        import sys
        from .batch_explain_checkpoints import main as explain_batch_main

        argv = [
            "batch_explain_checkpoints",
            *self.checkpoints,
            "--per-class",
            str(self.per_class),
            "--steps",
            str(self.steps),
            "--device",
            self.device,
            "--output-dir",
            str(self.output_dir),
        ]
        orig_argv = sys.argv
        try:
            sys.argv = argv
            return explain_batch_main()
        finally:
            sys.argv = orig_argv

