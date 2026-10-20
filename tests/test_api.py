import pytest
from fastapi.testclient import TestClient

from pd_model.api.app import create_app
from pd_model.config import FEATURES, INTEGER_COLUMNS
from pd_model.models.predict import predict_records


def request_from_row(raw_sample):
    row = raw_sample.iloc[0]
    return {
        column: int(row[column]) if column in INTEGER_COLUMNS else float(row[column])
        for column in FEATURES
    }


def test_valid_prediction_matches_batch(bundle, raw_sample):
    path, pipeline, metadata = bundle
    payload = request_from_row(raw_sample)
    with TestClient(create_app(path)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").json()["model_version"] == "pytest-fixture"
        result = client.post("/predict", json=dict(reversed(list(payload.items()))))
        assert result.status_code == 200
        expected = predict_records(pipeline, metadata, raw_sample[FEATURES].iloc[:1])[0]
        assert result.json() == expected


@pytest.mark.parametrize(
    "field,value",
    [
        ("AGE", 150),
        ("AGE", "24"),
        ("SEX", 3),
        ("EDUCATION", 99),
        ("LIMIT_BAL", -1),
        ("PAY_AMT1", -1),
        ("ID", 1),
    ],
)
def test_api_rejects_invalid_json(bundle, raw_sample, field, value):
    path, _, _ = bundle
    request = {**request_from_row(raw_sample), field: value}
    with TestClient(create_app(path)) as client:
        assert client.post("/predict", json=request).status_code == 422


def test_api_requires_all_features(bundle, raw_sample):
    request = request_from_row(raw_sample)
    del request["PAY_AMT6"]
    with TestClient(create_app(bundle[0])) as client:
        assert client.post("/predict", json=request).status_code == 422


def test_missing_model_fails_startup(tmp_path):
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(tmp_path / "missing.joblib")):
            pass
