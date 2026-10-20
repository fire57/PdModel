import joblib
import pandas as pd
import pytest

from pd_model.config import (
    FEATURES,
    SCHEMA_VERSION,
    TARGET,
    file_sha256,
    project_root,
    write_json,
)
from pd_model.models.pipeline import create_pipeline


@pytest.fixture
def raw_sample():
    return pd.read_csv(project_root() / "tests/fixtures/credit_sample.csv")


@pytest.fixture
def bundle(tmp_path, raw_sample):
    pipeline = create_pipeline(params={"max_iter": 500})
    pipeline.fit(raw_sample[FEATURES], raw_sample[TARGET])
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "features": FEATURES,
        "threshold": 0.5,
        "model_version": "pytest-fixture",
        "model_sha256": file_sha256(path),
        "profile": "test",
    }
    write_json(tmp_path / "metadata.json", metadata)
    return path, pipeline, metadata
