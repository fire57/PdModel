import logging
import os
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from pd_model.api.schemas import ClientData, PredictionResponse
from pd_model.config import FEATURES, project_root
from pd_model.models.predict import load_bundle, predict_records

logger = logging.getLogger("pd_model.api")


def create_app(model_path=None):
    @asynccontextmanager
    async def lifespan(application):
        path = model_path or os.environ.get(
            "MODEL_PATH", str(project_root() / "models/champion/pipeline.joblib")
        )
        application.state.pipeline, application.state.metadata = load_bundle(path)
        yield
        application.state.pipeline = None

    application = FastAPI(
        title="Credit Default Prediction API", version="0.1.0", lifespan=lifespan
    )

    @application.middleware("http")
    async def request_logging(request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        logger.info(
            "%s %s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response

    @application.get("/health")
    def health():
        return {"status": "ok"}

    @application.get("/ready")
    def ready():
        if application.state.pipeline is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        metadata = application.state.metadata
        return {
            "status": "ready",
            "model_version": metadata["model_version"],
            "schema_version": metadata["schema_version"],
            "profile": metadata.get("profile", "full"),
        }

    @application.post("/predict", response_model=PredictionResponse)
    def predict(data: ClientData):
        frame = pd.DataFrame([data.model_dump()], columns=FEATURES)
        try:
            return predict_records(
                application.state.pipeline, application.state.metadata, frame
            )[0]
        except Exception:
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail="Inference failed") from None

    return application


app = create_app()
