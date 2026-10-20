import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pd_model.config import FEATURES, SCHEMA_VERSION, file_sha256, read_json
from pd_model.data.validation import load_csv, require_valid


def positive_probability(pipeline, frame):
    classes = list(pipeline.classes_)
    if 1 not in classes:
        raise ValueError("Обученная модель не содержит положительного класса 1")
    return pipeline.predict_proba(frame.loc[:, FEATURES])[:, classes.index(1)]


def load_bundle(path):
    path = Path(path)
    metadata = read_json(path.parent / "metadata.json")
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Несовместимая версия входной схемы")
    if metadata["features"] != FEATURES:
        raise ValueError("Несовместимый список raw-признаков")
    if file_sha256(path) != metadata["model_sha256"]:
        raise ValueError("Модель повреждена: SHA-256 не совпадает")
    pipeline = joblib.load(path)
    if list(pipeline.classes_) != [0, 1]:
        raise ValueError("Ожидались классы 0 и 1")
    return pipeline, metadata


def predict_records(pipeline, metadata, frame):
    probabilities = positive_probability(pipeline, frame)
    if not np.isfinite(probabilities).all():
        raise ValueError("Модель вернула некорректные вероятности")
    threshold = metadata["threshold"]
    return [
        {
            "default_prediction": int(probability >= threshold),
            "default_probability": float(probability),
            "threshold": float(threshold),
            "model_version": metadata["model_version"],
            "schema_version": metadata["schema_version"],
        }
        for probability in probabilities
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="models/champion/pipeline.joblib")
    parser.add_argument("--output", default="reports/predictions.csv")
    args = parser.parse_args()
    frame = load_csv(args.data)
    require_valid(frame, "inference")
    pipeline, metadata = load_bundle(args.model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predict_records(pipeline, metadata, frame)).to_csv(output, index=False)
    print(f"Predicted {len(frame)} rows → {output}")


if __name__ == "__main__":
    main()
