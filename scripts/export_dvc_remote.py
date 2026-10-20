import argparse
import tarfile
from pathlib import Path

from pd_model.config import file_sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default=".dvc_remote")
    parser.add_argument("--output", default="artifacts/dvc-remote.tar.gz")
    args = parser.parse_args()
    source, archive = Path(args.remote), Path(args.output)
    if not source.is_dir() or not any(source.rglob("*")):
        raise FileNotFoundError("Remote пуст: сначала выполните dvc push")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(source, arcname="remote", recursive=True)
    write_json(
        Path("artifacts/dvc-remote-manifest.json"),
        {
            "file": archive.name,
            "sha256": file_sha256(archive),
            "bytes": archive.stat().st_size,
            "layout": "remote/files/md5",
            "source_remote": str(source),
            "download_url_variable": "PD_MODEL_DVC_ARCHIVE_URL",
        },
    )
    print(f"Exported {archive}: {archive.stat().st_size} bytes")


if __name__ == "__main__":
    main()
