import itertools
import os
from copy import deepcopy

import pandas as pd

import src.experiment as run_pkg
from src.utils import ParameterKeys, save_json, save_ruamel


class GridOptimizer:
    """Deterministic grid-search optimizer across model/data/training configurations."""

    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters
        general = parameters.get(ParameterKeys.GENERAL, {})
        self.out_dir = general.get(ParameterKeys.OUT_DIR, "models/grid_search")
        os.makedirs(self.out_dir, exist_ok=True)
        run_cfg = parameters.get(ParameterKeys.RUN, {})
        self.run_name = run_cfg.get(ParameterKeys.NAME, "AlzheimerGNNRun")
        self.base_config = run_cfg.get(ParameterKeys.CFG, {})
        self.grid = parameters.get("grid", {})
        self.metric_to_watch = parameters.get(ParameterKeys.METRIC_TO_WATCH, "balanced_accuracy")

    def launch(self):
        keys = list(self.grid.keys())
        values = [self.grid[k] for k in keys]
        combinations = list(itertools.product(*values))
        print(f"Starting GridOptimizer with {len(combinations)} combinations...")

        results = []
        best_score = None
        best_combo = None

        for idx, combo in enumerate(combinations):
            trial_cfg = deepcopy(self.base_config)
            trial_dir = os.path.join(self.out_dir, f"trial_{idx:03d}")
            trial_cfg.setdefault(ParameterKeys.GENERAL, {})[ParameterKeys.OUT_DIR] = trial_dir

            combo_dict = dict(zip(keys, combo))
            for dotted_key, val in combo_dict.items():
                parts = dotted_key.split(".")
                curr = trial_cfg
                for p in parts[:-1]:
                    curr = curr.setdefault(p, {})
                curr[parts[-1]] = val

            save_ruamel(trial_cfg, os.path.join(trial_dir, "parameters.yaml"))
            runner = run_pkg.__dict__[self.run_name](trial_cfg)
            metrics = runner.launch()
            score = float(metrics.get(self.metric_to_watch, 0.0))

            row = {"trial": idx, **combo_dict, **metrics}
            results.append(row)

            if best_score is None or score > best_score:
                best_score = score
                best_combo = row

        df_res = pd.DataFrame(results)
        df_res.to_csv(os.path.join(self.out_dir, "grid_results.csv"), index=False)
        if best_combo is not None:
            save_json(os.path.join(self.out_dir, "best_config.json"), best_combo)
        return results

