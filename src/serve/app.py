import os
import sys
import threading
import random
import logging
from typing import List, Dict, Any, Optional

import pandas as pd
import mlflow
import mlflow.pyfunc as pyfunc
from mlflow.exceptions import MlflowException

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Make sure we can import your preprocessing module
sys.path.append("/app/src")
from utils.preprocessing.s1 import preprocess_data  # type: ignore

# ------------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("taxi-serving")

# ------------------------------------------------------------------------------
# Config / env
# ------------------------------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = os.getenv("MODEL_NAME", "taxi-duration")
ALIAS_STABLE = os.getenv("MODEL_ALIAS_STABLE", "production")
ALIAS_CAND = (
    os.getenv("MODEL_ALIAS_CAND")
    or os.getenv("MODEL_ALIAS_CANDIDATE")
    or "staging"
)

if not MLFLOW_TRACKING_URI:
    logger.warning("MLFLOW_TRACKING_URI is not set! Model loading will fail.")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logger.info("Starting Taxi Canary server")
logger.info("CONFIG: MODEL_NAME=%r", MODEL_NAME)
logger.info("CONFIG: ALIAS_STABLE=%r", ALIAS_STABLE)
logger.info("CONFIG: ALIAS_CAND=%r", ALIAS_CAND)
logger.info("CONFIG: MLFLOW_TRACKING_URI=%r", MLFLOW_TRACKING_URI)

# ------------------------------------------------------------------------------
# FastAPI app / models
# ------------------------------------------------------------------------------

app = FastAPI(title="Taxi Canary", version="1.0")


class PredictPayload(BaseModel):
    records: List[Dict[str, Any]]


class WeightsBody(BaseModel):
    stable_weight: Optional[int] = None
    cand_weight: Optional[int] = None


_models: Dict[str, pyfunc.PyFuncModel] = {}
_ml = threading.Lock()
_st = threading.Lock()
state = {"stable_weight": 100, "cand_weight": 0}


# ------------------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------------------

def _load(alias: str) -> pyfunc.PyFuncModel:
    """
    Load a model for a given alias, with logging and error details.
    """
    with _ml:
        if alias in _models:
            logger.debug("Using cached model for alias=%s", alias)
            return _models[alias]

        model_uri = f"models:/{MODEL_NAME}@{alias}"
        logger.info("Loading model: alias=%s, uri=%s", alias, model_uri)

        try:
            m = pyfunc.load_model(model_uri)
        except MlflowException as e:
            logger.exception(
                "MLflow error while loading model alias=%s (uri=%s)", alias, model_uri
            )
            # re-raise so /predict can turn it into a 5xx HTTP error with details
            raise
        except Exception as e:
            logger.exception(
                "Unexpected error while loading model alias=%s (uri=%s)", alias, model_uri
            )
            raise

        _models[alias] = m
        logger.info("Successfully loaded model alias=%s", alias)
        return m


def _pick() -> str:
    """
    Pick 'stable' or 'candidate' based on current weights.
    """
    with _st:
        cw = int(state["cand_weight"])
        sw = int(state["stable_weight"])
    choice = "candidate" if random.randrange(100) < cw else "stable"
    logger.debug(
        "Backend pick: choice=%s (stable_weight=%s, cand_weight=%s)",
        choice,
        sw,
        cw,
    )
    return choice


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@app.get("/health")
def health():
    with _st:
        sw, cw = state["stable_weight"], state["cand_weight"]
    logger.info("Health check: stable_weight=%s, cand_weight=%s", sw, cw)
    return {"ok": True, "weights": {"stable": sw, "candidate": cw}}


@app.post("/predict")
def predict(p: PredictPayload):
    logger.info("Received /predict request with %d record(s)", len(p.records))

    try:
        # Preprocessing
        try:
            df = preprocess_data(pd.DataFrame(p.records))
            logger.debug("Preprocessing done. Columns=%s", list(df.columns))
        except Exception as e:
            logger.exception("Error during preprocessing in /predict")
            raise HTTPException(
                status_code=400,
                detail=f"Preprocessing error: {e}",
            )

        if "trip_duration" in df.columns:
            df = df.drop(columns=["trip_duration"])

        # Routing
        backend = _pick()
        alias = ALIAS_STABLE if backend == "stable" else ALIAS_CAND
        logger.info("Routing request to backend=%s alias=%s", backend, alias)

        # Load model & predict
        try:
            m = _load(alias)
        except Exception as e:
            # Already logged in _load
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load model alias '{alias}': {e}",
            )

        try:
            y = m.predict(df)
        except Exception as e:
            logger.exception(
                "Error during model.predict for alias=%s", alias
            )
            raise HTTPException(
                status_code=500,
                detail=f"Prediction error for alias '{alias}': {e}",
            )

        preds = [float(v) for v in y]
        logger.info(
            "Prediction successful: backend=%s alias=%s n=%d",
            backend,
            alias,
            len(preds),
        )
        return JSONResponse(
            {
                "backend": backend,
                "alias": alias,
                "predictions": preds,
            }
        )

    except HTTPException:
        # Already logged and structured
        raise
    except Exception as e:
        logger.exception("Unexpected error in /predict")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error in /predict: {e}",
        )


@app.post("/admin/weights")
def weights(body: WeightsBody):
    logger.info("Received /admin/weights body=%s", body.dict())
    if body.stable_weight is None and body.cand_weight is None:
        logger.warning("/admin/weights called without any weights set")
        raise HTTPException(status_code=400, detail="set a weight")

    with _st:
        old_state = dict(state)
        if body.stable_weight is not None:
            state["stable_weight"] = int(body.stable_weight)
        if body.cand_weight is not None:
            state["cand_weight"] = int(body.cand_weight)

        if body.stable_weight is None:
            state["stable_weight"] = 100 - state["cand_weight"]
        if body.cand_weight is None:
            state["cand_weight"] = 100 - state["stable_weight"]

        t = state["stable_weight"] + state["cand_weight"]
        if t != 100:
            sw = round(100 * state["stable_weight"] / t)
            state["stable_weight"] = sw
            state["cand_weight"] = 100 - sw

        logger.info(
            "Weights updated: old=%s new=%s", old_state, state
        )

    return {"ok": True, "weights": state}


@app.post("/admin/reload")
def reload_models():
    logger.info("Received /admin/reload – clearing model cache")
    with _ml:
        _models.clear()
    return {"ok": True}


@app.get("/debug/state")
def debug_state():
    """
    Optional helper endpoint so you can inspect state via curl while debugging.
    DO NOT expose this in a real public deployment.
    """
    with _st:
        sw, cw = state["stable_weight"], state["cand_weight"]
    loaded_aliases = list(_models.keys())
    logger.info(
        "Debug state requested: stable_weight=%s, cand_weight=%s, loaded_aliases=%s",
        sw,
        cw,
        loaded_aliases,
    )
    return {
        "weights": {"stable_weight": sw, "cand_weight": cw},
        "loaded_aliases": loaded_aliases,
        "config": {
            "MODEL_NAME": MODEL_NAME,
            "ALIAS_STABLE": ALIAS_STABLE,
            "ALIAS_CAND": ALIAS_CAND,
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
        },
    }

