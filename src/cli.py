import click
from src.utils import load_ruamel


@click.group()
def main():
    pass


@main.command("preprocess")
@click.option("--parameters", help="Path to the parameters file", required=True)
@click.option("--discover", is_flag=True, help="Optional flag to discover raw DICOM subjects and modalities")
@click.option("--dicom", is_flag=True, help="Optional flag to convert raw DICOM files to NIfTI")
@click.option("--smri", is_flag=True, help="Optional flag to run sMRI N4 bias correction and SPM segmentation")
@click.option("--pet", is_flag=True, help="Optional flag to run PET 4D-to-3D averaging and sMRI coregistration")
@click.option("--features", is_flag=True, help="Optional flag to extract ROI GM volume and PET SUVR features")
@click.option("--coords", is_flag=True, help="Optional flag to extract AAL3 anatomical ROI MNI centroids")
@click.option("--clinical", is_flag=True, help="Optional flag to build MCI conversion clinical dataset")
@click.option("--cn_ad", is_flag=True, help="Optional flag to build binary CN vs AD dataset")
@click.option("--multiclass", is_flag=True, help="Optional flag to build 3-class CN vs MCI vs AD dataset")
@click.option("--zhang", is_flag=True, help="Optional flag to build Zhang individual morphological network dataset")
@click.option("--ontology", is_flag=True, help="Optional flag to run FMA ontology mapping, KG build, or validation")
@click.option("--visualize", is_flag=True, help="Optional flag to generate visualization figures and tables")
def preprocess(
    parameters,
    discover,
    dicom,
    smri,
    pet,
    features,
    coords,
    clinical,
    cn_ad,
    multiclass,
    zhang,
    ontology,
    visualize,
):
    from src.preprocess import exe_preprocess_command

    exe_preprocess_command(
        parameters=load_ruamel(parameters),
        discover=discover,
        dicom=dicom,
        smri=smri,
        pet=pet,
        features=features,
        coords=coords,
        clinical=clinical,
        cn_ad=cn_ad,
        multiclass=multiclass,
        zhang=zhang,
        ontology=ontology,
        visualize=visualize,
    )


@main.command("experiment")
@click.option("--parameters", help="Path to the parameters file", required=True)
@click.option("--cls", help="Class to be used for running the experiment", required=True)
def experiment(parameters, cls):
    from src.experiment import exe_experiment

    exe_experiment(load_ruamel(parameters), cls)


@main.command("optimize")
@click.option("--parameters", help="Path to the parameters file", required=True)
@click.option("--cls", help="Class to be used for running the optimization", required=True)
def optimize(parameters, cls):
    from src.optimization import optimize_fn as exe_optimization

    exe_optimization(load_ruamel(parameters), cls)


@main.command("test")
@click.option("--parameters", help="Path to the parameters file", required=True)
@click.option("--cls", help="Class to be used for running the test", required=True)
def test(parameters, cls):
    from src.experiment import exe_test

    exe_test(load_ruamel(parameters), cls)


@main.command("explain")
@click.option("--parameters", help="Path to the parameters file", required=True)
@click.option("--cls", help="Class to be used for running the explanations", required=True)
def explain(parameters, cls):
    from src.explainability import exe_explain

    exe_explain(load_ruamel(parameters), cls)

