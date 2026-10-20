import argparse

from sklearn.model_selection import StratifiedGroupKFold

from pd_model.config import (
    FEATURES,
    INTEGER_COLUMNS,
    TARGET,
    file_sha256,
    load_config,
    resolve_path,
    write_json,
)
from pd_model.data.validation import load_csv, require_valid
from pd_model.features.build_features import feature_groups


def split_frame(frame, seed=42, folds=5, test_fold=0):
    data = frame.copy()
    data["feature_group"] = feature_groups(data[FEATURES])
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(data[FEATURES], data[TARGET], data["feature_group"]))
    train_index, test_index = splits[test_fold]
    train, test = data.iloc[train_index].copy(), data.iloc[test_index].copy()
    if set(train.ID) & set(test.ID):
        raise ValueError("Пересечение ID между train/test")
    if set(train.feature_group) & set(test.feature_group):
        raise ValueError("Пересечение одинаковых входов между train/test")
    return train, test


def prepare(config):
    source = resolve_path(config, config["data"]["raw"])
    digest = file_sha256(source)
    if digest != config["data"]["source_sha256"]:
        raise ValueError("Исходный snapshot изменён: SHA-256 не совпадает")
    frame = load_csv(source)
    raw_report = require_valid(frame, "raw")
    for column in INTEGER_COLUMNS:
        frame[column] = frame[column].astype("int64")
    train, test = split_frame(
        frame,
        config["split"]["seed"],
        config["split"]["folds"],
        config["split"]["test_fold"],
    )
    train_report = require_valid(train, "processed")
    test_report = require_valid(test, "processed")
    output = resolve_path(config, config["data"]["processed"])
    output.mkdir(parents=True, exist_ok=True)
    train.to_csv(output / "train.csv", index=False)
    test.to_csv(output / "test.csv", index=False)
    write_json(
        output / "split_manifest.json",
        {
            "source_sha256": digest,
            "method": "StratifiedGroupKFold / normalized input SHA-256 groups",
            "parameters": config["split"],
            "train_rows": len(train),
            "test_rows": len(test),
            "train_positive_rate": train[TARGET].mean(),
            "test_positive_rate": test[TARGET].mean(),
            "train_ids": train.ID.tolist(),
            "test_ids": test.ID.tolist(),
            "group_overlap": 0,
            "id_overlap": 0,
        },
    )
    write_json(
        resolve_path(config, "reports/validation/prepare.json"),
        {
            "success": True,
            "raw": raw_report,
            "train": train_report,
            "test": test_report,
        },
    )
    print(f"Prepared train={len(train)}, test={len(test)}; group overlap=0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="params.yaml")
    prepare(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
