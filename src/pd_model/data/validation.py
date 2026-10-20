import argparse
import csv
from pathlib import Path

import great_expectations as gx
import numpy as np
import pandas as pd

from pd_model.config import (
    INTEGER_COLUMNS,
    SCHEMA_VERSION,
    TARGET,
    project_root,
    read_json,
    write_json,
)


class DataValidationError(ValueError):
    pass


def load_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise DataValidationError("Повторяющиеся имена столбцов CSV")
    return pd.read_csv(path)


def validate_frame(frame, suite="raw", report_path=None, rules_dir=None):
    directory = rules_dir or project_root() / "data/expectations"
    rules = read_json(directory / f"{suite}.json")
    expected = set(rules["columns"])
    checks = [
        {"name": "nonempty_batch", "success": len(frame) > 0},
        {
            "name": "exact_columns",
            "success": set(frame.columns) == expected and frame.columns.is_unique,
        },
    ]
    for column in expected.intersection(frame.columns):
        if column == "feature_group":
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        checks.append(
            {
                "name": f"{column}:finite_numeric",
                "success": bool(
                    pd.api.types.is_numeric_dtype(frame[column])
                    and np.isfinite(numeric).all()
                    and not pd.api.types.is_bool_dtype(frame[column])
                ),
            }
        )
        if column in INTEGER_COLUMNS:
            checks.append(
                {
                    "name": f"{column}:integer_values",
                    "success": bool((numeric.dropna() % 1 == 0).all()),
                }
            )
    if suite == "processed" and TARGET in frame:
        checks.append(
            {"name": "both_target_classes", "success": set(frame[TARGET]) == {0, 1}}
        )
    if "feature_group" in frame:
        checks.append(
            {
                "name": "feature_group:sha256",
                "success": bool(
                    frame["feature_group"]
                    .astype(str)
                    .str.fullmatch(r"[0-9a-f]{64}")
                    .all()
                ),
            }
        )
    report = {
        "suite": rules["name"],
        "schema_version": SCHEMA_VERSION,
        "rows": len(frame),
        "preflight": checks,
        "gx": None,
        "success": False,
    }
    if len(frame) and set(frame.columns) == expected and frame.columns.is_unique:
        context = gx.get_context(mode="ephemeral")
        source = context.data_sources.add_pandas(name="credit_dataframe")
        asset = source.add_dataframe_asset(name="credit")
        batch = asset.add_batch_definition_whole_dataframe(name="whole_batch")
        expectations = [
            getattr(gx.expectations, item["type"])(**item["kwargs"])
            for item in rules["expectations"]
        ]
        expectation_suite = context.suites.add(
            gx.ExpectationSuite(name=rules["name"], expectations=expectations)
        )
        validation = gx.ValidationDefinition(
            name=f"{suite}_validation", data=batch, suite=expectation_suite
        )
        try:
            result = validation.run(
                batch_parameters={"dataframe": frame},
                result_format={
                    "result_format": "SUMMARY",
                    "partial_unexpected_count": 10,
                },
            )
            report["gx"] = result.to_json_dict()
            report["success"] = bool(result.success) and all(
                item["success"] for item in checks
            )
        except (TypeError, ValueError) as error:
            report["gx_error"] = str(error)
    if report_path:
        write_json(report_path, report)
    return report


def require_valid(frame, suite="raw", report_path=None, rules_dir=None):
    report = validate_frame(frame, suite, report_path, rules_dir)
    if not report["success"]:
        names = [item["name"] for item in report["preflight"] if not item["success"]]
        if report["gx"]:
            names += [
                str(item["expectation_config"])
                for item in report["gx"]["results"]
                if not item["success"]
            ]
        raise DataValidationError("Валидация не пройдена: " + "; ".join(names))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--suite", choices=["raw", "processed", "inference"], default="raw"
    )
    parser.add_argument("--report", default="reports/validation/raw.json")
    args = parser.parse_args()
    input_loaded = False
    try:
        frame = load_csv(args.data)
        input_loaded = True
        report = require_valid(frame, args.suite, args.report)
    except Exception as error:
        if (
            not input_loaded
            or not isinstance(error, DataValidationError)
            or not Path(args.report).exists()
        ):
            write_json(args.report, {"success": False, "error": str(error)})
        print(f"VALIDATION FAILED: {error}")
        return 2
    print(f"GX PASS: {report['rows']} rows, suite={report['suite']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
