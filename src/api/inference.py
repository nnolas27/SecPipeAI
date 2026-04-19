"""
SecPipeAI Inference API — cloud-native DevSecOps integration endpoint.

Exposes trained network intrusion detection classifiers as an HTTP service
so that CI/CD pipelines, container runtime monitors, and network probes
can submit feature vectors and receive real-time binary threat classifications.

Endpoints:
    POST /detect          — classify one network flow feature vector
    POST /detect/batch    — classify a batch of feature vectors
    GET  /health          — liveness probe for Kubernetes / load balancers
    GET  /models          — list available trained models and their metadata

Usage:
    uvicorn src.api.inference:app --host 0.0.0.0 --port 8000
    make serve
    docker run --rm -p 8000:8000 -v $(pwd)/outputs:/app/outputs secpipeai-api

DevSecOps integration pattern:
    1. Train models offline via: make all
    2. Start this server in a sidecar container or as a pipeline step
    3. Send network flow feature vectors from your monitoring agent to /detect
    4. Act on the returned prediction + confidence in your pipeline logic

The server loads models lazily on first request and caches them in memory
to avoid repeated disk I/O in high-throughput pipeline environments.
"""

import json
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import yaml

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise ImportError(
        "FastAPI and Pydantic are required for the inference API.\n"
        "Install with: pip install fastapi uvicorn pydantic>=2.0"
    ) from exc

# ── Configuration ─────────────────────────────────────────────────────────────

_CONFIG_PATH = Path("configs/experiment.yaml")
_SUPPORTED_DATASETS = ("cicids2017", "unsw_nb15")
_DEFAULT_MODEL = "xgboost"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


# ── FastAPI application ────────────────────────────────────────────────────────

app = FastAPI(
    title="SecPipeAI Threat Detection API",
    description=(
        "Real-time network intrusion detection for cloud-native DevSecOps pipelines. "
        "Submit network flow feature vectors; receive binary threat classifications "
        "with confidence scores."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── In-memory model cache ──────────────────────────────────────────────────────

_model_cache: dict[str, object] = {}
_feature_names_cache: dict[str, list[str]] = {}
_model_meta_cache: dict[str, dict] = {}


def _model_key(dataset: str, model_name: str) -> str:
    return f"{dataset}/{model_name}"


def _load_model(dataset: str, model_name: str):
    """Load a trained model from disk into the in-memory cache."""
    key = _model_key(dataset, model_name)
    if key in _model_cache:
        return _model_cache[key]

    cfg = _load_config()
    model_path = Path(cfg["paths"]["models"]) / dataset / f"{model_name}.joblib"
    meta_path = Path(cfg["paths"]["models"]) / dataset / f"{model_name}_meta.json"
    feat_path = Path(cfg["paths"]["processed_data"]) / dataset / "feature_names.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. "
            f"Run 'make train DATASET={dataset}' first."
        )

    model = joblib.load(model_path)
    _model_cache[key] = model

    if meta_path.exists():
        _model_meta_cache[key] = json.loads(meta_path.read_text())

    if feat_path.exists():
        _feature_names_cache[dataset] = json.loads(feat_path.read_text())

    return model


# ── Request / Response schemas ────────────────────────────────────────────────

class FlowDetectionRequest(BaseModel):
    """A single network flow feature vector for threat classification."""

    features: list[float] = Field(
        ...,
        description=(
            "Normalized feature vector. Must match the dimensionality of the "
            "trained model for the specified dataset: 77 features for cicids2017, "
            "190 features for unsw_nb15 (after one-hot encoding of categoricals). "
            "Features must be preprocessed (imputed, scaled) using the pipeline "
            "produced by 'make preprocess_<dataset>'."
        ),
        min_length=1,
    )
    dataset: str = Field(
        default="cicids2017",
        description="Dataset the model was trained on. One of: cicids2017, unsw_nb15.",
    )
    model_name: str = Field(
        default="xgboost",
        description="Classifier to use. One of: xgboost, random_forest, logistic_regression, dummy.",
    )
    pipeline_id: Optional[str] = Field(
        default=None,
        description="Optional CI/CD pipeline or run identifier for traceability.",
    )
    source_component: Optional[str] = Field(
        default=None,
        description="Optional label for the DevSecOps component submitting the request (e.g., 'github-actions', 'k8s-daemonset').",
    )


class FlowDetectionResult(BaseModel):
    """Threat classification result for a single network flow."""

    prediction: int = Field(
        ...,
        description="Binary classification: 0 = benign traffic, 1 = malicious/attack traffic.",
    )
    label: str = Field(
        ...,
        description="Human-readable label: 'BENIGN' or 'ATTACK'.",
    )
    confidence: float = Field(
        ...,
        description="Model's probability estimate that the flow is an attack (range: 0.0–1.0).",
    )
    model_name: str
    dataset: str
    n_features: int = Field(..., description="Number of input features processed.")
    inference_latency_ms: float = Field(..., description="Server-side inference time in milliseconds.")
    timestamp_utc: str = Field(..., description="ISO 8601 UTC timestamp of the detection.")
    pipeline_id: Optional[str] = None
    source_component: Optional[str] = None
    alert: bool = Field(
        ...,
        description="True when prediction=1 (attack detected). Use this field to gate pipeline actions.",
    )


class BatchDetectionRequest(BaseModel):
    """A batch of network flow feature vectors for bulk classification."""

    flows: list[list[float]] = Field(
        ...,
        description="List of feature vectors. Each vector must match the dataset's feature dimensionality.",
        min_length=1,
        max_length=10000,
    )
    dataset: str = Field(default="cicids2017")
    model_name: str = Field(default="xgboost")
    pipeline_id: Optional[str] = None
    source_component: Optional[str] = None


class BatchDetectionResult(BaseModel):
    n_flows: int
    n_attacks: int
    attack_rate: float = Field(..., description="Fraction of flows classified as attacks.")
    predictions: list[int]
    confidences: list[float]
    model_name: str
    dataset: str
    inference_latency_ms: float
    timestamp_utc: str
    pipeline_id: Optional[str] = None
    source_component: Optional[str] = None
    alert: bool = Field(..., description="True when any flow in the batch is classified as an attack.")


class ModelInfo(BaseModel):
    model_name: str
    dataset: str
    available: bool
    train_shape: Optional[list[int]] = None
    train_time_s: Optional[float] = None
    n_features: Optional[int] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Operations"])
async def health():
    """Liveness and readiness probe for Kubernetes and load balancers."""
    return {
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "models_loaded": list(_model_cache.keys()),
    }


@app.get("/models", response_model=list[ModelInfo], tags=["Operations"])
async def list_models():
    """List all available trained models and their metadata."""
    cfg = _load_config()
    results = []
    model_names = [m["name"] for m in cfg["models"]]

    for dataset in _SUPPORTED_DATASETS:
        feat_path = Path(cfg["paths"]["processed_data"]) / dataset / "feature_names.json"
        n_features = None
        if feat_path.exists():
            n_features = len(json.loads(feat_path.read_text()))

        for name in model_names:
            model_path = Path(cfg["paths"]["models"]) / dataset / f"{name}.joblib"
            meta_path = Path(cfg["paths"]["models"]) / dataset / f"{name}_meta.json"
            available = model_path.exists()
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            results.append(ModelInfo(
                model_name=name,
                dataset=dataset,
                available=available,
                train_shape=meta.get("train_shape"),
                train_time_s=meta.get("train_time_s"),
                n_features=n_features,
            ))
    return results


@app.post("/detect", response_model=FlowDetectionResult, tags=["Detection"])
async def detect_single(request: FlowDetectionRequest):
    """
    Classify a single network flow feature vector as benign or attack.

    This is the primary integration point for CI/CD pipelines and runtime monitors.
    The caller is responsible for preprocessing the feature vector using the same
    imputation and scaling pipeline produced by 'make preprocess_<dataset>'.

    DevSecOps usage example (GitHub Actions):
        curl -X POST https://your-host:8000/detect \\
          -H 'Content-Type: application/json' \\
          -d '{"features": [...], "dataset": "cicids2017", "source_component": "github-actions"}'
    """
    if request.dataset not in _SUPPORTED_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported dataset '{request.dataset}'. Must be one of: {_SUPPORTED_DATASETS}",
        )

    try:
        model = _load_model(request.dataset, request.model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    X = np.array(request.features, dtype=np.float32).reshape(1, -1)

    t0 = time.perf_counter()
    prediction = int(model.predict(X)[0])
    confidence = float(model.predict_proba(X)[0, 1])
    latency_ms = (time.perf_counter() - t0) * 1000

    return FlowDetectionResult(
        prediction=prediction,
        label="ATTACK" if prediction == 1 else "BENIGN",
        confidence=confidence,
        model_name=request.model_name,
        dataset=request.dataset,
        n_features=len(request.features),
        inference_latency_ms=round(latency_ms, 3),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        pipeline_id=request.pipeline_id,
        source_component=request.source_component,
        alert=prediction == 1,
    )


@app.post("/detect/batch", response_model=BatchDetectionResult, tags=["Detection"])
async def detect_batch(request: BatchDetectionRequest):
    """
    Classify a batch of network flow feature vectors.

    Use this endpoint for bulk scanning of network captures, CI build artifacts,
    or container image traffic logs. Returns aggregate attack statistics and
    per-flow predictions suitable for pipeline gating decisions.

    The 'alert' field is True when at least one flow in the batch is classified
    as an attack. Gate your pipeline on this field.
    """
    if request.dataset not in _SUPPORTED_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported dataset '{request.dataset}'. Must be one of: {_SUPPORTED_DATASETS}",
        )

    try:
        model = _load_model(request.dataset, request.model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    X = np.array(request.flows, dtype=np.float32)

    t0 = time.perf_counter()
    predictions = model.predict(X).tolist()
    confidences = model.predict_proba(X)[:, 1].tolist()
    latency_ms = (time.perf_counter() - t0) * 1000

    n_attacks = int(sum(predictions))

    return BatchDetectionResult(
        n_flows=len(predictions),
        n_attacks=n_attacks,
        attack_rate=round(n_attacks / len(predictions), 6),
        predictions=[int(p) for p in predictions],
        confidences=[round(float(c), 6) for c in confidences],
        model_name=request.model_name,
        dataset=request.dataset,
        inference_latency_ms=round(latency_ms, 3),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        pipeline_id=request.pipeline_id,
        source_component=request.source_component,
        alert=n_attacks > 0,
    )
