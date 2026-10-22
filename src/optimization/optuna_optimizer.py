import os
from copy import deepcopy

import optuna

import src.experiment as run_pkg
from src.utils import ParameterKeys, load_ruamel, save_ruamel


class OptunaOptimizer:
    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters
        self.init()

    def init(self) -> None:
        print("Init general")
        self._init_general()
        print("Init hyper parameter space")
        self._init_space()
        print("Init run parameters")
        self._init_run()
        print("Init study")
        self._init_study()

    def _init_general(self) -> None:
        general_parameters = self.parameters.get(ParameterKeys.GENERAL, {})
        self.seed = general_parameters.get(ParameterKeys.SEED, 42)
        self.out_dir = general_parameters.get(ParameterKeys.OUT_DIR, "models/optuna")
        os.makedirs(self.out_dir, exist_ok=True)

    def _init_space(self) -> None:
        self.space = self.parameters.get(ParameterKeys.HSPACE, {})

    def _init_run(self) -> None:
        run_parameters = self.parameters.get(ParameterKeys.RUN, {})
        self.run_name = run_parameters.get(ParameterKeys.NAME, "AlzheimerGNNRun")
        self.run_config = run_parameters.get(ParameterKeys.CFG, {})

    def _init_study(self) -> None:
        study_parameters = self.parameters.get(ParameterKeys.STUDY, {})
        self.study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(
                multivariate=True,
                seed=self.seed,
            ),
            **study_parameters.get(ParameterKeys.STUDY, {"direction": "maximize"}),
        )
        self.optimization_parameters = deepcopy(study_parameters.get(ParameterKeys.OPTIMIZER, {"n_trials": 20}))
        self.mapping = deepcopy(study_parameters.get(ParameterKeys.MAPPING, {}))
        self.metric_to_watch = study_parameters.get(ParameterKeys.METRIC_TO_WATCH, "balanced_accuracy")

    def sample_space(self, trial: optuna.Trial) -> dict:
        return {
            name: getattr(trial, v[ParameterKeys.NAME])(**v[ParameterKeys.CFG])
            for name, v in self.space.items()
        }

    def _apply_params(self, trial: optuna.Trial) -> dict:
        trial_run_params = deepcopy(self.run_config)
        hopt_trial_params = self.sample_space(trial)
        mapping_params = deepcopy(self.mapping)

        trial_out_dir = f"{self.out_dir}/{trial.number}"
        trial_run_params.setdefault(ParameterKeys.GENERAL, {})[ParameterKeys.OUT_DIR] = trial_out_dir
        if ParameterKeys.EARLY_STOP in trial_run_params and ParameterKeys.CFG in trial_run_params[ParameterKeys.EARLY_STOP]:
            trial_run_params[ParameterKeys.EARLY_STOP][ParameterKeys.CFG][ParameterKeys.OUT_DIR] = trial_out_dir

        plain_params = mapping_params.pop("plain", {})
        for k, v in plain_params.items():
            trial_run_params.setdefault(ParameterKeys.MODEL, {}).setdefault(ParameterKeys.CFG, {})[v] = (
                hopt_trial_params[k]
            )

        for section_key, param_mapping_dict in mapping_params.items():
            for k, v in param_mapping_dict.items():
                if section_key in trial_run_params and isinstance(trial_run_params[section_key], dict):
                    if ParameterKeys.CFG in trial_run_params[section_key]:
                        trial_run_params[section_key][ParameterKeys.CFG][v] = hopt_trial_params[k]
                    else:
                        trial_run_params[section_key][v] = hopt_trial_params[k]
                elif (
                    ParameterKeys.MODEL in trial_run_params
                    and ParameterKeys.CFG in trial_run_params[ParameterKeys.MODEL]
                    and section_key in trial_run_params[ParameterKeys.MODEL][ParameterKeys.CFG]
                ):
                    target = trial_run_params[ParameterKeys.MODEL][ParameterKeys.CFG][section_key]
                    if isinstance(target, dict) and ParameterKeys.CFG in target:
                        target[ParameterKeys.CFG][v] = hopt_trial_params[k]
                    elif isinstance(target, dict):
                        target[v] = hopt_trial_params[k]

        if (
            ParameterKeys.DATA in trial_run_params
            and isinstance(trial_run_params[ParameterKeys.DATA], dict)
            and ParameterKeys.VAL in trial_run_params[ParameterKeys.DATA]
            and ParameterKeys.TEST in trial_run_params[ParameterKeys.DATA]
        ):
            trial_run_params[ParameterKeys.DATA][ParameterKeys.TEST][ParameterKeys.CFG] = deepcopy(
                trial_run_params[ParameterKeys.DATA][ParameterKeys.VAL][ParameterKeys.CFG]
            )

        return trial_run_params

    def objective(self, trial: optuna.Trial) -> float:
        trial_run_parameters = self._apply_params(trial)
        run = run_pkg.__dict__[self.run_name](trial_run_parameters)
        save_ruamel(
            trial_run_parameters,
            f"{trial_run_parameters[ParameterKeys.GENERAL][ParameterKeys.OUT_DIR]}/parameters.yaml",
        )
        try:
            out = run.launch()
            return float(out[self.metric_to_watch])
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            direction = self.study.direction
            return 0.0 if direction == optuna.study.StudyDirection.MAXIMIZE else 1.0

    def optimize(self):
        n_trials = self.optimization_parameters.get(ParameterKeys.N_TRIALS, 20)
        n_remaining_trials = n_trials - len(self.study.trials)
        if n_remaining_trials > 0:
            self.optimization_parameters.update({ParameterKeys.N_TRIALS: n_remaining_trials})
            self.study.optimize(self.objective, **self.optimization_parameters)
        else:
            print("Trials already completed. Skipping")

    def test(self) -> None:
        best_trial_id = self.study.best_trial.number
        best_run_params_path = f"{self.out_dir}/{best_trial_id}/parameters.yaml"
        best_run_params = load_ruamel(best_run_params_path)
        best_run_params[ParameterKeys.GENERAL][ParameterKeys.OUT_DIR] = self.out_dir
        if (
            ParameterKeys.DATA in self.run_config
            and isinstance(self.run_config[ParameterKeys.DATA], dict)
            and ParameterKeys.TEST in self.run_config[ParameterKeys.DATA]
        ):
            best_run_params[ParameterKeys.DATA][ParameterKeys.TEST][ParameterKeys.CFG] = deepcopy(
                self.run_config[ParameterKeys.DATA][ParameterKeys.TEST][ParameterKeys.CFG]
            )
        run = run_pkg.__dict__[self.run_name](best_run_params)
        return run.test()

    def launch(self) -> None:
        self.optimize()
        self.test()
        self.study.trials_dataframe().to_csv(f"{self.out_dir}/trials.csv", index=False)

