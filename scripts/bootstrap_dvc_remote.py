import argparse
import os
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from pd_model.config import file_sha256, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="artifacts/dvc-remote.tar.gz")
    parser.add_argument("--url", default=os.environ.get("PD_MODEL_DVC_ARCHIVE_URL"))
    parser.add_argument("--remote", default=".dvc_remote")
    args = parser.parse_args()
    archive = Path(args.archive)
    manifest = read_json(Path("artifacts/dvc-remote-manifest.json"))
    if not archive.exists():
        if not args.url or not args.url.startswith("https://"):
            raise ValueError("Передайте локальный архив или HTTPS URL через --url")
        archive.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(args.url, timeout=120) as source:
            with archive.open("wb") as destination:
                while block := source.read(1024 * 1024):
                    destination.write(block)
    if file_sha256(archive) != manifest["sha256"]:
        raise ValueError("SHA-256 архива remote не совпадает с manifest")
    staging = Path(".cache/dvc-restore")
    staging.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream.getmembers():
            parts = Path(member.name).parts
            if (
                not parts
                or parts[0] != "remote"
                or ".." in parts
                or Path(member.name).is_absolute()
                or member.issym()
                or member.islnk()
                or not (member.isdir() or member.isfile())
            ):
                raise ValueError(f"Недопустимый путь в архиве: {member.name}")
        stream.extractall(staging, filter="data")
    remote = Path(args.remote).resolve()
    remote.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copytree(staging / "remote", remote, dirs_exist_ok=True)
    relative_url = os.path.relpath(remote, Path.cwd())
    subprocess.run(
        ["dvc", "remote", "modify", "local", "url", relative_url, "--local"],
        check=True,
    )
    subprocess.run(["dvc", "pull"], check=True)
    print("SHA-256 OK; DVC remote восстановлен, dvc pull выполнен")


if __name__ == "__main__":
    main()
