import argparse
import shutil
import sqlite3
import tarfile
from pathlib import Path

from pd_model.config import file_sha256, read_json, write_json


def export_tracking():
    root = Path.cwd().resolve()
    build = root / ".cache/tracking-export"
    build.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(root / "mlflow.db") as source:
        with sqlite3.connect(build / "mlflow.db") as destination:
            source.backup(destination)
    archive = root / "artifacts/mlflow-tracking.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(build / "mlflow.db", arcname="mlflow.db")
        stream.add(root / "mlartifacts", arcname="mlartifacts")
    write_json(
        root / "artifacts/mlflow-tracking-manifest.json",
        {
            "file": archive.name,
            "sha256": file_sha256(archive),
            "bytes": archive.stat().st_size,
            "source_root": str(root),
            "backend": "SQLite backup",
        },
    )
    print(f"MLflow runs exported: {archive}")


def restore_tracking(archive):
    root = Path.cwd().resolve()
    if (root / "mlflow.db").exists() or (root / "mlartifacts").exists():
        raise FileExistsError(
            "Восстановление возможно в папку без существующего MLflow store"
        )
    manifest = read_json(root / "artifacts/mlflow-tracking-manifest.json")
    if file_sha256(archive) != manifest["sha256"]:
        raise ValueError("SHA-256 MLflow archive не совпадает")
    staging = root / ".cache/tracking-restore"
    staging.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream.getmembers():
            parts = Path(member.name).parts
            if (
                not parts
                or parts[0] not in {"mlflow.db", "mlartifacts"}
                or ".." in parts
                or Path(member.name).is_absolute()
                or member.issym()
                or member.islnk()
                or not (member.isdir() or member.isfile())
            ):
                raise ValueError(f"Недопустимый archive member: {member.name}")
        stream.extractall(staging, filter="data")
    shutil.copy2(staging / "mlflow.db", root / "mlflow.db")
    shutil.copytree(staging / "mlartifacts", root / "mlartifacts")
    old_root, new_root = manifest["source_root"], str(root)
    replacements = [
        (Path(old_root).as_uri(), root.as_uri()),
        (old_root, new_root),
    ]
    with sqlite3.connect(root / "mlflow.db") as connection:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        ]
        for table in tables:
            escaped_table = '"' + table.replace('"', '""') + '"'
            columns = connection.execute(
                f"PRAGMA table_info({escaped_table})"
            ).fetchall()
            for _, column, kind, *_ in columns:
                if any(name in kind.upper() for name in ("TEXT", "CHAR")):
                    escaped_column = '"' + column.replace('"', '""') + '"'
                    for old, new in replacements:
                        connection.execute(
                            f"UPDATE {escaped_table} SET {escaped_column} = "
                            f"replace({escaped_column}, ?, ?) "
                            f"WHERE instr({escaped_column}, ?) > 0",
                            (old, new, old),
                        )
    print("MLflow restored; absolute artifact paths migrated to current project root")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["export", "restore"])
    parser.add_argument("--archive", default="artifacts/mlflow-tracking.tar.gz")
    args = parser.parse_args()
    if args.operation == "export":
        export_tracking()
    else:
        restore_tracking(Path(args.archive))
