import os
import subprocess
from importlib.metadata import version

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from pd_model.config import resolve_path


def configure_tracking(config, profile="full"):
    uri = os.environ.get("MLFLOW_TRACKING_URI", config["tracking"]["uri"])
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = "sqlite:///" + str(resolve_path(config, uri[len("sqlite:///") :]))
    mlflow.set_tracking_uri(uri)
    name = config["tracking"]["experiment"]
    if profile != "full":
        name += f"_{profile}"
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        artifacts = resolve_path(config, config["tracking"]["artifacts"]).resolve()
        artifacts.mkdir(parents=True, exist_ok=True)
        experiment_id = mlflow.create_experiment(
            name, artifact_location=artifacts.as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def git_commit(root):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def log_pipeline(pipeline, sample, root):
    packages = ["scikit-learn", "numpy", "pandas", "scipy", "joblib", "cloudpickle"]
    return mlflow.sklearn.log_model(
        pipeline,
        name="model",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        signature=infer_signature(sample, pipeline.predict(sample)),
        input_example=sample,
        code_paths=[str(root / "src/pd_model")],
        pip_requirements=[f"{package}=={version(package)}" for package in packages],
    )
