from src.experiment.smoke_test_graph_modes import run_smoke_test
from src.utils import ParameterKeys, set_seed


class SmokeTestRun:
    """Synthetic end-to-end smoke test verifying all graph topologies, FMA KG integration, and models."""

    def __init__(self, parameters: dict = None):
        self.parameters = parameters or {}
        general = self.parameters.get(ParameterKeys.GENERAL, {})
        self.seed = int(general.get(ParameterKeys.SEED, 42))
        self.backend = str(general.get("fma_embedding_backend", "hash"))
        set_seed(self.seed)

    def launch(self):
        return run_smoke_test(backend=self.backend)

    def test(self):
        return self.launch()

