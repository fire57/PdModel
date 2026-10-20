import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from pd_model.config import (
    FEATURES,
    TARGET,
    load_config,
    resolve_path,
    write_json,
)
from pd_model.data.validation import load_csv, require_valid
from pd_model.models.metrics import classification_metrics
from pd_model.models.predict import load_bundle, positive_probability
from pd_model.tracking.mlflow_utils import configure_tracking


def evaluate(config):
    data = load_csv(resolve_path(config, config["data"]["processed"]) / "test.csv")
    require_valid(data, "processed")
    models = resolve_path(config, config["training"]["models"])
    _, champion = load_bundle(models / "champion/pipeline.joblib")
    configure_tracking(config)
    figures = resolve_path(config, config["evaluation"]["figures"])
    figures.mkdir(parents=True, exist_ok=True)
    comparison = []
    all_probabilities = {}
    evaluation_run_ids = {}
    for path in sorted((models / "candidates").glob("*/pipeline.joblib")):
        pipeline, metadata = load_bundle(path)
        identifier = metadata["scenario_id"]
        probabilities = positive_probability(pipeline, data[FEATURES])
        metrics = classification_metrics(
            data[TARGET], probabilities, metadata["threshold"]
        )
        all_probabilities[identifier] = probabilities
        comparison.append({"scenario_id": identifier, **metrics})
        plot_path = figures / f"{identifier}_roc.png"
        fig, ax = plt.subplots(figsize=(7, 5))
        RocCurveDisplay.from_predictions(
            data[TARGET], probabilities, name=identifier, ax=ax
        )
        ax.plot([0, 1], [0, 1], "--", color="#767676", label="Случайный классификатор")
        ax.set(
            title="ROC на отложенной test",
            xlabel="False positive rate",
            ylabel="True positive rate",
        )
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        try:
            mlflow.get_run(metadata["mlflow_run_id"])
            run_args = {"run_id": metadata["mlflow_run_id"]}
        except mlflow.exceptions.MlflowException:
            run_args = {"run_name": f"evaluate_{identifier}"}
        with mlflow.start_run(**run_args) as run:
            evaluation_run_ids[identifier] = run.info.run_id
            mlflow.set_tag("source_training_run", metadata["mlflow_run_id"])
            mlflow.set_tag("test_used_for_selection", "false")
            mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
            mlflow.log_artifact(str(plot_path), "evaluation")
        print(
            f"TEST {identifier}: ROC-AUC={metrics['roc_auc']:.5f}, "
            f"F1={metrics['f1']:.5f}"
        )
    selected = next(
        item for item in comparison if item["scenario_id"] == champion["scenario_id"]
    )
    champion_probabilities = all_probabilities[champion["scenario_id"]]
    predictions = (champion_probabilities >= champion["threshold"]).astype(int)
    for name, display in [
        (
            "champion_pr",
            lambda ax: PrecisionRecallDisplay.from_predictions(
                data[TARGET], champion_probabilities, ax=ax
            ),
        ),
        (
            "champion_confusion",
            lambda ax: ConfusionMatrixDisplay.from_predictions(
                data[TARGET], predictions, ax=ax, colorbar=False
            ),
        ),
        (
            "champion_calibration",
            lambda ax: CalibrationDisplay.from_predictions(
                data[TARGET], champion_probabilities, n_bins=10, ax=ax
            ),
        ),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        display(ax)
        ax.set_title(f"{name.replace('champion_', '').capitalize()} - test")
        fig.tight_layout()
        fig.savefig(figures / f"{name}.png", dpi=160)
        plt.close(fig)
    result = {
        "champion": champion["scenario_id"],
        "model_version": champion["model_version"],
        "source_sha256": champion["source_sha256"],
        "selection": "Chosen using train CV before test evaluation",
        "metrics": {
            key: value for key, value in selected.items() if key != "scenario_id"
        },
    }
    write_json(resolve_path(config, config["evaluation"]["metrics"]), result)
    output = resolve_path(config, "reports/experiments/test_comparison.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison).to_csv(output, index=False)
    with mlflow.start_run(run_id=evaluation_run_ids[champion["scenario_id"]]):
        mlflow.log_artifact(
            str(resolve_path(config, config["evaluation"]["metrics"])), "evaluation"
        )
        for path in figures.glob("champion_*.png"):
            mlflow.log_artifact(str(path), "evaluation")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="params.yaml")
    evaluate(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
