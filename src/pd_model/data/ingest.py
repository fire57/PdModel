import argparse
import shutil
from pathlib import Path

from pd_model.config import file_sha256, load_config, write_json
from pd_model.data.validation import load_csv, require_valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/raw/UCI_Credit_Card.csv")
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    source = Path(args.input).resolve()
    digest = file_sha256(source)
    if digest != config["data"]["source_sha256"]:
        raise ValueError("SHA-256 не совпадает с предоставленным исходным snapshot")
    frame = load_csv(source)
    require_valid(frame, report_path="reports/validation/ingest.json")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if source != output.resolve():
        shutil.copy2(source, output)
    write_json(
        "data/source_manifest.json",
        {
            "filename": output.name,
            "sha256": digest,
            "rows": len(frame),
            "columns": frame.columns.tolist(),
            "source": "User-provided UCI_Credit_Card.csv",
            "uci": (
                "https://archive.ics.uci.edu/dataset/350/"
                "default+of+credit+card+clients"
            ),
        },
    )
    print(f"Imported {len(frame)} rows; sha256={digest}")


if __name__ == "__main__":
    main()
