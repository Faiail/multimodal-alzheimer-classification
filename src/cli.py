import click
from src.utils import load_ruamel, ParameterKeys


@click.group()
def main():
    pass


@main.command("preprocess")
@click.option("--parameters", help="Path to the parameters file", required=True)
@click.option(
    "--cls", help="Class to be used for running the preprocessing", required=True
)
def preprocess(
    parameters,
    cls,
):
    from src.preprocess import exe_preprocess_command

    exe_preprocess_command(
        parameters=load_ruamel(parameters),
        cls=cls,
    )


@main.command("experiment")
@click.option("--parameters", help="Path to the parameters file", required=True)
@click.option(
    "--cls", help="Class to be used for running the experiment", required=True
)
@click.option("--fold", help="Fold id (for cross validation)", default=0, type=int, required=False)
def experiment(parameters, cls, fold=0):
    from src.experiment import exe_experiment

    parameters = load_ruamel(parameters)
    parameters[ParameterKeys.GENERAL][ParameterKeys.FOLD] = fold
    exe_experiment(parameters, cls)


@main.command("test")
@click.option("--parameters", help="Path to the parameters file", required=True)
@click.option("--cls", help="Class to be used for running the test", required=True)
def test(parameters, cls):
    from src.experiment import exe_test

    exe_test(load_ruamel(parameters), cls)
