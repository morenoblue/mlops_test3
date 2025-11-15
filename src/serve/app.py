from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os, threading, random, pandas as pd, mlflow, mlflow.pyfunc as pyfunc
import sys; sys.path.append("/app/src")
from utils.preprocessing.s1 import preprocess_data

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI"); mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_NAME = os.getenv("MODEL_NAME","taxi-duration")
ALIAS_STABLE = os.getenv("MODEL_ALIAS_STABLE","production")
ALIAS_CAND = os.getenv("MODEL_ALIAS_CAND") or os.getenv("MODEL_ALIAS_CANDIDATE") or "staging"

app = FastAPI(title="Taxi Canary", version="1.0")

class PredictPayload(BaseModel):
    records: List[Dict[str, Any]]

_models: Dict[str, pyfunc.PyFuncModel] = {}
_ml = threading.Lock()
_st = threading.Lock()
state = {"stable_weight":100,"cand_weight":0}

def _load(alias:str)->pyfunc.PyFuncModel:
    with _ml:
        m=_models.get(alias)
        if m: return m
        m=pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}")
        _models[alias]=m
        return m

def _pick()->str:
    with _st: cw=int(state["cand_weight"])
    return "candidate" if random.randrange(100)<cw else "stable"

@app.get("/health")
def health():
    with _st: sw, cw = state["stable_weight"], state["cand_weight"]
    return {"ok":True,"weights":{"stable":sw,"candidate":cw}}

@app.post("/predict")
def predict(p: PredictPayload):
    df = preprocess_data(pd.DataFrame(p.records))
    if "trip_duration" in df.columns: df=df.drop(columns=["trip_duration"])
    backend=_pick(); alias = ALIAS_STABLE if backend=="stable" else ALIAS_CAND
    m=_load(alias); y=m.predict(df)
    return JSONResponse({"backend":backend,"alias":alias,"predictions":[float(v) for v in y]})

class W(BaseModel):
    stable_weight: Optional[int]=None
    cand_weight: Optional[int]=None

@app.post("/admin/weights")
def weights(b:W):
    if b.stable_weight is None and b.cand_weight is None: raise HTTPException(400,"set a weight")
    with _st:
        if b.stable_weight is not None: state["stable_weight"]=int(b.stable_weight)
        if b.cand_weight is not None: state["cand_weight"]=int(b.cand_weight)
        if b.stable_weight is None: state["stable_weight"]=100-state["cand_weight"]
        if b.cand_weight is None: state["cand_weight"]=100-state["stable_weight"]
        t=state["stable_weight"]+state["cand_weight"]
        if t!=100:
            sw=round(100*state["stable_weight"]/t); state["stable_weight"]=sw; state["cand_weight"]=100-sw
    return {"ok":True,"weights":state}

@app.post("/admin/reload")
def reload():
    with _ml: _models.clear()
    return {"ok":True}
