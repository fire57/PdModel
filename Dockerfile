FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    MODEL_PATH=/app/models/champion/pipeline.joblib

WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir --no-deps . \
    && useradd --create-home --uid 10001 pdmodel

COPY params.yaml ./
COPY config ./config
COPY data/expectations ./data/expectations
COPY models/champion ./models/champion
COPY models/monitoring_reference.json ./models/monitoring_reference.json
RUN chown -R pdmodel:pdmodel /app

USER pdmodel
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)"
CMD ["uvicorn", "pd_model.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
