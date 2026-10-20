import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from sklearn.base import clone

from pd_model.config import FEATURES, TARGET, project_root, write_json
from pd_model.models.pipeline import create_pipeline
from pd_model.models.predict import load_bundle, positive_probability
from pd_model.tracking.mlflow_utils import configure_tracking, log_pipeline


def test_round_trip(bundle, raw_sample):
    path, original, _ = bundle
    loaded, _ = load_bundle(path)
    np.testing.assert_allclose(
        positive_probability(original, raw_sample[FEATURES]),
        positive_probability(loaded, raw_sample[FEATURES]),
        atol=1e-12,
    )
    assert list(loaded.named_steps) == ["features", "preprocessor", "classifier"]


def test_imputer_fits_train_only(raw_sample):
    X = raw_sample[FEATURES].iloc[:4].copy()
    X["LIMIT_BAL"] = [100, 200, 300, 400]
    pipeline = create_pipeline(aggregates=False, params={"max_iter": 500})
    clone(pipeline).fit(X, [0, 1, 0, 1])
    pipeline.fit(X, [0, 1, 0, 1])
    imputer = pipeline["preprocessor"].named_transformers_["numeric"]["imputer"]
    assert imputer.statistics_[0] == 250
    test = X.copy()
    test["LIMIT_BAL"] = [np.nan, 99999, 99999, 99999]
    probabilities = positive_probability(pipeline, test)
    assert np.isfinite(probabilities).all()
    assert imputer.statistics_[0] == 250


def test_real_mlflow_model_roundtrip(tmp_path, raw_sample):
    config = {
        "_root": tmp_path,
        "tracking": {
            "uri": "sqlite:///test.db",
            "experiment": "pytest",
            "artifacts": "artifacts",
        },
    }
    configure_tracking(config)
    model = create_pipeline(params={"max_iter": 500})
    X = raw_sample[FEATURES]
    model.fit(X, raw_sample[TARGET])
    with mlflow.start_run():
        info = log_pipeline(model, X.head(2), project_root())
    loaded = mlflow.sklearn.load_model(info.model_uri)
    np.testing.assert_allclose(
        positive_probability(model, X), positive_probability(loaded, X), atol=1e-12
    )


def test_corrupted_bundle_is_rejected(bundle):
    path, _, _ = bundle
    with path.open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="SHA-256"):
        load_bundle(path)


def test_incompatible_schema_is_rejected(bundle):
    path, _, metadata = bundle
    write_json(path.parent / "metadata.json", {**metadata, "schema_version": "other"})
    with pytest.raises(ValueError, match="схемы"):
        load_bundle(path)
