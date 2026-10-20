import json
import subprocess
import sys

import numpy as np
import pytest

from pd_model.config import TARGET, project_root
from pd_model.data.validation import load_csv, validate_frame


def test_real_gx_accepts_source(raw_sample):
    report = validate_frame(raw_sample)
    assert report["success"]
    assert report["gx"]["success"]
    assert len(report["gx"]["results"]) > 50


@pytest.mark.parametrize(
    "column,value",
    [
        ("AGE", 150),
        ("AGE", 20.5),
        ("AGE", "text"),
        ("LIMIT_BAL", -1),
        ("LIMIT_BAL", np.nan),
        ("LIMIT_BAL", np.inf),
        ("PAY_AMT1", -1),
        ("SEX", 3),
        ("EDUCATION", 99),
        ("PAY_0", 10),
        (TARGET, 2),
    ],
)
def test_anomalies_are_rejected(raw_sample, column, value):
    frame = raw_sample.copy()
    frame[column] = frame[column].astype(object)
    frame.loc[frame.index[0], column] = value
    assert not validate_frame(frame)["success"]


def test_legitimate_special_values(raw_sample):
    frame = raw_sample.copy()
    frame.loc[0, "BILL_AMT1"] = -100
    frame.loc[0, "EDUCATION"] = 5
    frame.loc[1, "EDUCATION"] = 6
    frame.loc[2, "EDUCATION"] = 0
    frame.loc[0, "MARRIAGE"] = 0
    frame.loc[0, "PAY_0"] = -2
    frame.loc[1, "PAY_0"] = 0
    assert validate_frame(frame)["success"]


def test_missing_empty_and_duplicate_id(raw_sample):
    assert not validate_frame(raw_sample.drop(columns="AGE"))["success"]
    assert not validate_frame(raw_sample.iloc[:0])["success"]
    duplicate = raw_sample.copy()
    duplicate.loc[1, "ID"] = duplicate.loc[0, "ID"]
    assert not validate_frame(duplicate)["success"]


def test_cli_fails_and_writes_real_gx_report(tmp_path, raw_sample):
    raw_sample.loc[0, "AGE"] = 150
    source, output = tmp_path / "anomalies.csv", tmp_path / "report.json"
    raw_sample.to_csv(source, index=False)
    command = [
        sys.executable,
        "-m",
        "pd_model.data.validation",
        "--data",
        str(source),
        "--report",
        str(output),
    ]
    result = subprocess.run(command, cwd=project_root(), capture_output=True, text=True)
    assert result.returncode == 2
    report = json.loads(output.read_text())
    assert report["success"] is False
    assert any(not item["success"] for item in report["gx"]["results"])


def test_duplicate_csv_headers_rejected(tmp_path):
    path = tmp_path / "duplicate.csv"
    path.write_text("ID,ID\n1,2\n")
    with pytest.raises(ValueError, match="Повторяющиеся"):
        load_csv(path)


def test_cli_replaces_existing_report_after_csv_read_error(tmp_path):
    source = tmp_path / "duplicate.csv"
    output = tmp_path / "report.json"
    source.write_text("ID,ID\n1,2\n", encoding="utf-8")
    output.write_text(json.dumps({"success": True}), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pd_model.data.validation",
            "--data",
            str(source),
            "--report",
            str(output),
        ],
        cwd=project_root(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["success"] is False
    assert "Повторяющиеся имена столбцов" in report["error"]
