import argparse
import time

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pd_model.config import (
    FEATURES,
    INTEGER_COLUMNS,
    TARGET,
    load_config,
    read_json,
    resolve_path,
    write_json,
)
from pd_model.data.validation import load_csv, require_valid
from pd_model.models.metrics import classification_metrics
from pd_model.models.predict import load_bundle
from pd_model.monitoring.psi import population_stability_index


def simulate(config, scenario="normal", api_url="http://127.0.0.1:8000", rows=None):
    data = load_csv(resolve_path(config, config["data"]["processed"]) / "test.csv")
    count = rows or config["monitoring"]["batch_size"]
    data = data.sample(n=min(count, len(data)), random_state=config["training"]["seed"])
    frame = data[FEATURES].copy()
    if scenario == "shifted":
        frame["LIMIT_BAL"] = frame["LIMIT_BAL"] * config["monitoring"]["shift_factor"]
    require_valid(frame, "inference")
    models = resolve_path(config, config["training"]["models"])
    pipeline, metadata = load_bundle(models / "champion/pipeline.joblib")
    reference = read_json(models / "monitoring_reference.json")
    if reference["model_version"] != metadata["model_version"]:
        raise ValueError("Reference принадлежит другой версии модели")
    responses, errors = [], []
    started = time.perf_counter()
    with httpx.Client(base_url=api_url, timeout=10) as client:
        for index, (_, row) in enumerate(frame.iterrows()):
            payload = {
                column: (
                    int(row[column])
                    if column in INTEGER_COLUMNS
                    else float(row[column])
                )
                for column in FEATURES
            }
            last_error = None
            for attempt in range(3):
                try:
                    response = client.post("/predict", json=payload)
                    response.raise_for_status()
                    result = response.json()
                    if result["model_version"] != reference["model_version"]:
                        raise ValueError("API использует другую версию модели")
                    if not 0 <= result["default_probability"] <= 1:
                        raise ValueError("API вернул вероятность вне диапазона")
                    responses.append(result)
                    last_error = None
                    break
                except (httpx.HTTPError, ValueError, KeyError) as error:
                    last_error = str(error)
                    time.sleep(0.1 * (attempt + 1))
            if last_error:
                errors.append({"position": index, "error": last_error})
            if (index + 1) % 200 == 0:
                print(f"{scenario}: {index + 1}/{len(frame)} HTTP requests", flush=True)
    report_path = resolve_path(config, f"reports/monitoring/{scenario}.json")
    if errors:
        write_json(
            report_path,
            {"success": False, "errors": errors, "successful_requests": len(responses)},
        )
        raise RuntimeError(f"{len(errors)} API requests failed; report: {report_path}")
    transformed = pipeline.named_steps["features"].transform(frame)
    probabilities = [item["default_probability"] for item in responses]
    results = {}
    for column, distribution in reference["distributions"].items():
        values = (
            probabilities if column == "default_probability" else transformed[column]
        )
        result = population_stability_index(distribution, values, reference["epsilon"])
        score = result["psi"]
        result["status"] = (
            "alert"
            if score >= config["monitoring"]["alert"]
            else "warning" if score >= config["monitoring"]["warning"] else "low"
        )
        results[column] = result
    quality = None
    if scenario == "normal":
        quality = classification_metrics(
            data[TARGET], probabilities, metadata["threshold"]
        )
    report = {
        "success": True,
        "scenario": scenario,
        "model_version": metadata["model_version"],
        "rows": len(frame),
        "successful_requests": len(responses),
        "failed_requests": 0,
        "seconds": time.perf_counter() - started,
        "shift_factor": (
            config["monitoring"]["shift_factor"] if scenario == "shifted" else 1
        ),
        "psi": results,
        "normal_quality": quality,
        "interpretation": (
            "Heuristic distribution signal; does not establish concept drift"
        ),
    }
    write_json(report_path, report)
    write_json(
        resolve_path(config, f"reports/monitoring/{scenario}_responses.json"), responses
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, column in zip(axes, ["LIMIT_BAL", "default_probability"]):
        result = results[column]
        positions = np.arange(len(result["reference_share"]))
        ax.bar(
            positions - 0.2,
            result["reference_share"],
            width=0.4,
            color="#285A8E",
            label="Train reference",
        )
        ax.bar(
            positions + 0.2,
            result["current_share"],
            width=0.4,
            color="#B48324",
            label=scenario,
        )
        ax.set(
            title=f"{column}; PSI={result['psi']:.3f}",
            xlabel="Фиксированный train bin",
            ylabel="Доля наблюдений",
        )
        ax.legend()
    fig.tight_layout()
    figure = resolve_path(config, f"reports/monitoring/{scenario}_distribution.png")
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    print(
        f"{scenario}: LIMIT_BAL PSI={results['LIMIT_BAL']['psi']:.4f}; "
        f"report={report_path}"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--scenario", choices=["normal", "shifted"], default="normal")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--rows", type=int)
    args = parser.parse_args()
    simulate(load_config(args.config), args.scenario, args.api_url, args.rows)
