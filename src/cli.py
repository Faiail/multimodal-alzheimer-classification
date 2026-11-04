import click
from src.utils import load_ruamel


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
def experiment(parameters, cls):
    from src.experiment import exe_experiment

    exe_experiment(load_ruamel(parameters), cls)

@main.command("test")
@click.option("--parameters", help="Path to the parameters file", required=True)
@click.option("--cls", help="Class to be used for running the test", required=True)
def test(parameters, cls):
    from src.experiment import exe_test

    exe_test(load_ruamel(parameters), cls)
