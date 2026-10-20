import argparse
import time

import httpx
import pandas as pd

from pd_model.config import read_json, write_json
from pd_model.models.predict import load_bundle, predict_records


def smoke(
    url,
    model="models/champion/pipeline.joblib",
    output="reports/evidence/api_smoke.json",
):
    request = read_json("examples/sample_request.json")
    pipeline, metadata = load_bundle(model)
    expected = predict_records(pipeline, metadata, pd.DataFrame([request]))[0]
    with httpx.Client(base_url=url, timeout=10) as client:
        deadline = time.monotonic() + 60
        while True:
            try:
                ready = client.get("/ready")
                ready.raise_for_status()
                break
            except (httpx.HTTPError, OSError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("API не стал ready за 60 секунд") from None
                time.sleep(1)
        response = client.post("/predict", json=request)
        response.raise_for_status()
        actual = response.json()
        if abs(actual["default_probability"] - expected["default_probability"]) > 1e-10:
            raise AssertionError("HTTP и batch probabilities различаются")
        for key in (
            "default_prediction",
            "model_version",
            "threshold",
            "schema_version",
        ):
            if actual[key] != expected[key]:
                raise AssertionError(f"HTTP и batch {key} различаются")
        invalid = {**request, "AGE": 150}
        invalid_status = client.post("/predict", json=invalid).status_code
        missing = {key: value for key, value in request.items() if key != "LIMIT_BAL"}
        missing_status = client.post("/predict", json=missing).status_code
        extra_status = client.post("/predict", json={**request, "ID": 1}).status_code
        if (invalid_status, missing_status, extra_status) != (422, 422, 422):
            raise AssertionError("Некорректные запросы должны получить 422")
        report = {
            "success": True,
            "ready": ready.json(),
            "prediction": actual,
            "invalid_age_status": invalid_status,
            "missing_field_status": missing_status,
            "extra_field_status": extra_status,
            "batch_probability_match": True,
        }
        write_json(output, report)
        print(report)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="models/champion/pipeline.joblib")
    parser.add_argument("--output", default="reports/evidence/api_smoke.json")
    args = parser.parse_args()
    smoke(args.url, args.model, args.output)
