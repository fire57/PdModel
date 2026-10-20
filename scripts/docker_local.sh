#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
image="${PD_MODEL_IMAGE:-pd-model:local}"
container="${PD_MODEL_CONTAINER:-pd-model-local}"
test -f models/champion/pipeline.joblib
test -f models/champion/metadata.json
test -f models/monitoring_reference.json
docker build --tag "$image" .
docker run --detach --name "$container" --publish 127.0.0.1:8000:8000 "$image"
.venv/bin/python scripts/smoke_api.py --url http://127.0.0.1:8000 --output reports/evidence/docker_smoke.json
printf 'API: http://127.0.0.1:8000/docs\nContainer: %s\n' "$container"
