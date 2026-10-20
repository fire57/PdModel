import hashlib
import json
import math
import os
from pathlib import Path

import yaml

TARGET = "default.payment.next.month"
STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"] + STATUS_COLUMNS
NUMERIC = ["LIMIT_BAL", "AGE"] + BILL_COLUMNS + PAYMENT_COLUMNS
FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
FEATURES += STATUS_COLUMNS + BILL_COLUMNS + PAYMENT_COLUMNS
INTEGER_COLUMNS = ["ID", "AGE"] + CATEGORICAL + [TARGET]
SCHEMA_VERSION = "credit-v1"


def project_root():
    if os.environ.get("PD_MODEL_ROOT"):
        return Path(os.environ["PD_MODEL_ROOT"]).resolve()
    if (Path.cwd() / "params.yaml").exists():
        return Path.cwd()
    return Path(__file__).resolve().parents[2]


def load_config(path="params.yaml"):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["_root"] = path.parent
    return config


def resolve_path(config, path):
    return config["_root"] / path


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
