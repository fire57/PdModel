from pd_model.config import write_json
from pd_model.models.predict import positive_probability
from pd_model.monitoring.psi import make_distribution


def build_reference(pipeline, frame, metadata, config, output):
    transformed = pipeline.named_steps["features"].transform(frame)
    numeric = ["AGE", "LIMIT_BAL", "BILL_AMT1"]
    if "max_delay" in transformed:
        numeric += ["max_delay"]
    distributions = {
        column: make_distribution(transformed[column], config["bins"])
        for column in numeric
    }
    distributions["PAY_0"] = make_distribution(transformed["PAY_0"], categorical=True)
    distributions["default_probability"] = make_distribution(
        positive_probability(pipeline, frame), config["bins"]
    )
    reference = {
        "model_version": metadata["model_version"],
        "source_sha256": metadata["source_sha256"],
        "rows": len(frame),
        "epsilon": config["epsilon"],
        "distributions": distributions,
        "probability_reference": (
            "Fitted champion predictions on train; in-sample reference"
        ),
    }
    write_json(output, reference)
    return reference
