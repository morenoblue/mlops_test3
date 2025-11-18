This is my project:


.
├── Dockerfile
├── Dockerfile.serve
├── .dockerignore
├── .git
│   ├── COMMIT_EDITMSG
│   ├── config
│   ├── description
│   ├── HEAD
│   ├── hooks
│   │   ├── applypatch-msg.sample
│   │   ├── commit-msg.sample
│   │   ├── fsmonitor-watchman.sample
│   │   ├── post-update.sample
│   │   ├── pre-applypatch.sample
│   │   ├── pre-commit.sample
│   │   ├── pre-merge-commit.sample
│   │   ├── prepare-commit-msg.sample
│   │   ├── pre-push.sample
│   │   ├── pre-rebase.sample
│   │   ├── pre-receive.sample
│   │   ├── push-to-checkout.sample
│   │   ├── sendemail-validate.sample
│   │   └── update.sample
│   ├── index
│   ├── info
│   │   └── exclude
│   ├── logs
│   │   ├── HEAD
│   │   └── refs
│   │       ├── heads
│   │       └── remotes
│   ├── objects
│   └── refs
│       ├── heads
│       │   └── main
│       ├── remotes
│       │   └── origin
│       └── tags
├── .github
│   └── workflows
│       ├── cd.yml
│       └── ci.yml
├── .gitignore
├── prompt.md
├── quick_references.md
├── requirements-serve.txt
├── requirements.txt
├── scripts
│   ├── make_smoke_sample.py
│   └── mlops.py
├── src
│   ├── data
│   │   └── smoke_sample.parquet
│   ├── models
│   │   └── random_forest.py
│   ├── pipelines
│   │   └── pipeline_taxi_duration.py
│   ├── serve
│   │   └── app.py
│   ├── train.py
│   └── utils
│       ├── io.py
│       └── preprocessing
│           └── s1.py
├── tutorial.md
└── yellow_tripdata_2010-01.parquet



app.py
```python
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

```

random_forest.py
```python
from sklearn.ensemble import RandomForestRegressor
def get(): return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
```

io.py
```python
import os, glob
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

def _read_from_file(fp: str, need: int | None) -> pd.DataFrame:
    if need is None:
        return pd.read_parquet(fp, engine="pyarrow")
    out, got = [], 0
    pf = pq.ParquetFile(fp)
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg)
        out.append(t); got += t.num_rows
        if got >= need: break
    tbl = pa.concat_tables(out) if out else pa.table({})
    df = tbl.to_pandas()
    return df.iloc[:need] if need is not None else df

def read_parquet(path: str, nrows: int | None = None) -> pd.DataFrame:
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        if not files: return pd.DataFrame()
        if nrows is None:
            return pd.concat([pd.read_parquet(f, engine="pyarrow") for f in files], ignore_index=True)
        need = nrows; parts = []
        for f in files:
            left = need - sum(len(p) for p in parts)
            if left <= 0: break
            parts.append(_read_from_file(f, left))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return _read_from_file(path, nrows)
```

s1.py
```python
import pandas as pd, numpy as np

def _tod(h):
    return 0 if 6<=h<12 else (1 if 12<=h<16 else (2 if 16<=h<22 else 3))

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["trip_distance","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude",
              "fare_amount","surcharge","mta_tax","tip_amount","tolls_amount","total_amount"]:
        if c in df.columns: df[c] = df[c].astype("float32")
    if "vendor_id" in df.columns: df = df.drop(columns=["vendor_id"])
    if "pickup_datetime" in df.columns:
        ts = pd.to_datetime(df["pickup_datetime"])
        df["pickup_year"]=ts.dt.year.astype("uint16")
        df["pickup_month"]=ts.dt.month.astype("uint8")
        df["pickup_day"]=ts.dt.day.astype("uint8")
        df["pickup_weekday"]=ts.dt.weekday.astype("uint8")
        df["pickup_hour"]=ts.dt.hour.astype("uint8")
        df["pickup_part_of_day"]=df["pickup_hour"].apply(_tod).astype("uint8")
    if "dropoff_datetime" in df.columns and "pickup_datetime" in df.columns:
        dur=(pd.to_datetime(df["dropoff_datetime"])-pd.to_datetime(df["pickup_datetime"])).dt.total_seconds().astype("float32")
        df["trip_duration"]=dur
        df=df[df["trip_duration"]>0]
    for c in ["store_and_fwd_flag","payment_type","pickup_datetime","dropoff_datetime"]:
        if c in df.columns: df = df.drop(columns=[c])
    return df.astype(np.float32)
```

pipeline_taxi_duration.py
```python
    df = read_parquet(path, nrows=nrows)
    dfp = preprocess_data(df)
    y = dfp["trip_duration"].astype(np.float32)
    X = dfp.drop(columns=["trip_duration"])
    return X, y

def build(model):
    return Pipeline(steps=[("scaler", StandardScaler()), ("model", model)])
```

train.py
```python
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
EXP = "taxi-duration"
NAME = "taxi-duration"

get_model = importlib.import_module("models.random_forest").get
pl = importlib.import_module("pipelines.pipeline_taxi_duration")


def main():
    X, y = pl.load_data()
    model = pl.build(get_model())

    mlflow.set_experiment(EXP)

    with mlflow.start_run() as run:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        t = time.time()
        model.fit(Xtr, ytr)
        ypred = model.predict(Xte)
        dur = time.time() - t

        # Compute RMSE in a way that works on all sklearn versions
        mse = mean_squared_error(yte, ypred)
        rmse = float(sqrt(mse))

        mlflow.log_metrics(
            {
                "rmse": rmse,
                "fit_seconds": float(dur),
            }
        )

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=NAME,
        )

        rid = run.info.run_id

    from mlflow.tracking import MlflowClient

    c = MlflowClient()
    mv = c.search_model_versions(f"name='{NAME}' and run_id='{rid}'")[0]

    try:
        prod = c.get_model_version_by_alias(NAME, "production")
        prod_rmse = c.get_run(prod.run_id).data.metrics["rmse"]
        better = rmse <= prod_rmse
    except Exception:
        better = True

    if better:
        c.set_registered_model_alias(NAME, "staging", mv.version)
        print("PROMOTE=1")
    else:
        print("PROMOTE=0")


if __name__ == "__main__":
    main()

```

requirements.txt
```requirements
scikit-learn>=1.0
pandas>=1.5
pyarrow>=10.0
mlflow>=2.8
```

requirements-serve.txt
```text
fastapi>=0.110
uvicorn>=0.23
pandas>=1.5
pyarrow>=10.0
scikit-learn>=1.0
mlflow>=2.8
```

Dockerfile.serve
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements-serve.txt /app/
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r requirements-serve.txt
COPY src/ /app/src/
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY src/ /app/src/
ENTRYPOINT ["python", "/app/src/train.py"]
```

mlops.py
```python
#!/usr/bin/env python3
import os, sys
from mlflow.tracking import MlflowClient

NAME = "taxi-duration"
client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])

def promote_staging_to_production():
    v = client.get_model_version_by_alias(NAME, "staging")
    client.set_registered_model_alias(NAME, "production", v.version)
    print("production <- staging (v%s)" % v.version)

def has_production():
    try:
        client.get_model_version_by_alias(NAME, "production")
        print("1")
    except Exception:
        print("0")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "promote-staging-to-production":
        promote_staging_to_production()
    elif cmd == "has-production":
        has_production()
```

make_smoke_sample.py
```python
SRC  = Path(os.getenv("SMOKE_SRC", BASE / "yellow_tripdata_2010-01.parquet"))
OUT  = BASE / "src" / "data" / "smoke_sample.parquet"
ROWS = int(os.getenv("SMOKE_ROWS", "256"))

def parquet_files(p: Path):
    return [p] if p.is_file() else sorted(p.glob("*.parquet"))

need = ROWS
chunks = []
for fp in parquet_files(SRC):
    pf = pq.ParquetFile(fp)
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg)           # read one row-group only
        chunks.append(t)
        need -= t.num_rows
        if need <= 0: break
    if need <= 0: break

tbl = pa.concat_tables(chunks) if chunks else pa.table({})
tbl = tbl.slice(0, ROWS)                    # trim any overshoot
OUT.parent.mkdir(parents=True, exist_ok=True)
pq.write_table(tbl, OUT, compression="snappy")
print(f"wrote {OUT} ({tbl.num_rows} rows)")
```

.dockerignore
```gitignore
.git
.github
__pycache__/
*.pyc
.venv
dist
build
*.egg-info
.pytest_cache
```


ci.yml
```yml

# dummy code version, just to avoid increasing wait times between the iterations since we are focusin on the cd.yml rn:

name: CI
run-name: "${{ github.workflow }}"

permissions: write-all

on:
  push:
  pull_request:
  workflow_dispatch:

jobs:
  dummy:
    runs-on: ubuntu-latest
    steps:
      - name: Fast dummy CI
        run: echo "Dummy CI run – just here to trigger CD"


# original code:

# name: CI

# on:
#   push:
#   pull_request:
#
# jobs:
#   build-train-serve:
#     runs-on: ubuntu-latest
#     permissions: { contents: read, packages: write }
#     steps:
#       - uses: actions/checkout@v4
#       - uses: docker/setup-buildx-action@v3
#
#       - name: Build train
#         uses: docker/build-push-action@v6
#         with:
#           context: .
#           file: Dockerfile
#           load: true
#           tags: |
#             ghcr.io/${{ github.repository }}:training-${{ github.sha }}
#
#       - name: Smoke train (local file store)
#         run: |
#           docker run --rm \
#             -e MLFLOW_TRACKING_URI="file:/tmp/mlruns" \
#             -e DATA_PATH="/app/src/data/smoke_sample.parquet" \
#             -e TRAIN_NROWS="256" \
#             ghcr.io/${{ github.repository }}:training-${{ github.sha }}
#
#       - name: Build serve
#         uses: docker/build-push-action@v6
#         with:
#           context: .
#           file: Dockerfile.serve
#           load: true
#           tags: |
#             ghcr.io/${{ github.repository }}:serving-${{ github.sha }}
#
#       - name: Smoke serve
#         run: |
#           docker run -d --name serve -p 18000:8000 ghcr.io/${{ github.repository }}:serving-${{ github.sha }}
#           sleep 2
#           curl -fsS http://localhost:18000/health
#           docker rm -f serve
#
#       - name: Login GHCR
#         if: github.event_name != 'pull_request'
#         uses: docker/login-action@v3
#         with:
#           registry: ghcr.io
#           username: ${{ github.actor }}
#           password: ${{ secrets.GITHUB_TOKEN }}
#
#       - name: Push images
#         if: github.event_name != 'pull_request'
#         run: |
#           docker push ghcr.io/${{ github.repository }}:training-${{ github.sha }}
#           docker push ghcr.io/${{ github.repository }}:serving-${{ github.sha }}
```

cd.yml
```yml

name: CD
run-name: "${{ github.workflow }}"

permissions: write-all

on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]

jobs:
  train_big:
    if: ${{ github.event.workflow_run.conclusion == 'success' && github.event.workflow_run.event != 'pull_request' }}
    runs-on: [self-hosted, trainer]
    outputs:
      PROMOTE: ${{ steps.promote.outputs.PROMOTE }}
    steps:
      - uses: actions/checkout@v4

      - name: Login GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Pull train image
        # run: docker pull ghcr.io/${{ github.repository }}:training-${{ github.event.workflow_run.head_sha }}
        run: docker pull ghcr.io/${{ github.repository }}:training-72f8e0f8de9e1b325f52b7dfe21650bdd862275c

      - name: Big train (register + maybe alias=staging)
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
          HOME_DIR: ${{ env.HOME }}
        run: |
          docker run --rm --network host \
            -v "$HOME/train_data:/data:ro" \
            -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
            -e DATA_PATH="/data" \
            -e TRAIN_NROWS="1000" \
            ghcr.io/${{ github.repository }}:training-72f8e0f8de9e1b325f52b7dfe21650bdd862275c | tee out.txt

      - name: Output promote flag
        id: promote
        run: |
          v=$(grep -o 'PROMOTE=[01]' out.txt | tail -n1 | cut -d= -f2 || echo 1)
          echo "PROMOTE=$([ "$v" = "1" ] && echo true || echo false)" >> $GITHUB_OUTPUT

  staging:
    needs: [train_big]
    if: ${{ needs.train_big.outputs.PROMOTE == 'true' }}
    runs-on: [self-hosted, staging]
    steps:
      - uses: actions/checkout@v4

      - name: Login GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Pull serve image
        # run: docker pull ghcr.io/${{ github.repository }}:serving-${{ github.event.workflow_run.head_sha }}
        run: docker pull ghcr.io/${{ github.repository }}:serving-72f8e0f8de9e1b325f52b7dfe21650bdd862275c

      - name: Run serving (staging alias)
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          docker rm -f model-staging || true
          docker run -d --name model-staging -p 8000:8000 \
            -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
            -e MODEL_NAME="taxi-duration" \
            -e MODEL_ALIAS_STABLE="production" \
            -e MODEL_ALIAS_CAND="staging" \
            ghcr.io/${{ github.repository }}:serving-72f8e0f8de9e1b325f52b7dfe21650bdd862275c 

          echo "Containers right after docker run:"
          docker ps -a

          sleep 5
          # Make candidate 100% for staging smoke test
          curl -fsS -X POST http://localhost:8000/admin/weights \
               -H "Content-Type: application/json" \
               -d '{"cand_weight":100}'

          curl -fsS http://localhost:8000/health
          curl -fsS -X POST http://localhost:8000/predict \
            -H "Content-Type: application/json" \
            -d '{"records":[{"vendor_id":1,"pickup_datetime":"2010-01-15T08:30:00","dropoff_datetime":"2010-01-15T08:45:00","passenger_count":1,"trip_distance":2.3,"pickup_longitude":-73.99,"pickup_latitude":40.73,"rate_code":1,"dropoff_longitude":-73.98,"dropoff_latitude":40.75,"fare_amount":12.5,"surcharge":0.5,"mta_tax":0.5,"tip_amount":2.0,"tolls_amount":0.0,"total_amount":15.5,"store_and_fwd_flag":"N","payment_type":"CRD"}]}'

      - name: Dump model-staging logs
        if: failure()
        run: |
          echo "=== docker ps -a ==="
          docker ps -a || true
          echo "=== logs for model-staging ==="
          docker logs model-staging || true


  prod_canary:
    needs: [train_big, staging]
    if: ${{ needs.train_big.outputs.PROMOTE == 'true' }}
    runs-on: [self-hosted, production]
    steps:
      - uses: actions/checkout@v4

      - name: Login GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Pull serve image
        # run: docker pull ghcr.io/${{ github.repository }}:serving-${{ github.event.workflow_run.head_sha }}
        run: docker pull ghcr.io/${{ github.repository }}:serving-72f8e0f8de9e1b325f52b7dfe21650bdd862275c

      - name: Start gateway on :8000
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          docker rm -f model-gateway || true
          docker run -d --name model-gateway -p 8000:8000 \
            -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
            -e MODEL_NAME="taxi-duration" \
            -e MODEL_ALIAS_STABLE="production" \
            -e MODEL_ALIAS_CAND="staging" \
            ghcr.io/${{ github.repository }}:serving-72f8e0f8de9e1b325f52b7dfe21650bdd862275c

          sleep 2
          curl -fsS http://localhost:8000/health

      - name: Check if Production alias exists
        id: hasprod
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          docker run --rm --network host \
            -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
            -v "$PWD:/w" -w /w python:3.11-slim bash -lc \
            "python -m pip install -q mlflow>=2.8 && python scripts/mlops.py has-production" > has_prod.txt
          echo "HAS_PROD=$(tr -d '\n\r' < has_prod.txt)" >> $GITHUB_OUTPUT

      # First run: 100% candidate, then promote
      - name: First run → 100% candidate immediately
        if: ${{ steps.hasprod.outputs.HAS_PROD == '0' }}
        run: |
          curl -fsS -X POST http://localhost:8000/admin/weights \
            -H "Content-Type: application/json" \
            -d '{"cand_weight":100}'
          curl -fsS http://localhost:8000/health

      - name: First run → promote staging → production
        if: ${{ steps.hasprod.outputs.HAS_PROD == '0' }}
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          docker run --rm --network host \
            -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
            -v "$PWD:/w" -w /w python:3.11-slim bash -lc \
            "python -m pip install -q mlflow>=2.8 && python scripts/mlops.py promote-staging-to-production"

      # Subsequent runs: 10% → soak 5m → 50% → soak 5m → 100%, then promote
      - name: Canary 10%
        if: ${{ steps.hasprod.outputs.HAS_PROD == '1' }}
        run: |
          curl -fsS -X POST http://localhost:8000/admin/weights \
            -H "Content-Type: application/json" \
            -d '{"stable_weight":90,"cand_weight":10}'
          curl -fsS http://localhost:8000/health

      - name: Soak at 10% (5 minutes)
        if: ${{ steps.hasprod.outputs.HAS_PROD == '1' }}
        run: sleep 200

      - name: Canary 50%
        if: ${{ steps.hasprod.outputs.HAS_PROD == '1' }}
        run: |
          curl -fsS -X POST http://localhost:8000/admin/weights \
            -H "Content-Type: application/json" \
            -d '{"cand_weight":50}'
          curl -fsS http://localhost:8000/health

      - name: Soak at 50% (5 minutes)
        if: ${{ steps.hasprod.outputs.HAS_PROD == '1' }}
        run: sleep 200

      - name: Canary 100%
        if: ${{ steps.hasprod.outputs.HAS_PROD == '1' }}
        run: |
          curl -fsS -X POST http://localhost:8000/admin/weights \
            -H "Content-Type: application/json" \
            -d '{"cand_weight":100}'
          curl -fsS http://localhost:8000/health

      - name: Promote staging → production
        if: ${{ steps.hasprod.outputs.HAS_PROD == '1' }}
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          docker run --rm --network host \
            -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
            -v "$PWD:/w" -w /w python:3.11-slim bash -lc \
            "python -m pip install -q mlflow>=2.8 && python scripts/mlops.py promote-staging-to-production"

```





-------------------------------------------------------------------


The ci runs fine. the problem is the cd.yml. its showing this:


```txt

Run docker run --rm --network host \
  docker run --rm --network host \
    -v "$HOME/train_data:/data:ro" \
    -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
    -e DATA_PATH="/data" \
    -e TRAIN_NROWS="1000" \
    ghcr.io/morenoblue/mlops_test3:training-72f8e0f8de9e1b325f52b7dfe21650bdd862275c | tee out.txt
  shell: /usr/bin/bash -e {0}
  env:
    MLFLOW_TRACKING_URI: ***
    HOME_DIR: 
2025/11/18 17:52:04 INFO mlflow.tracking.fluent: Experiment with name 'taxi-duration' does not exist. Creating a new experiment.
2025/11/18 17:52:05 WARNING mlflow.utils.git_utils: Failed to import Git (the Git executable is probably not on your PATH), so Git SHA is not available. Error: Failed to initialize: Bad git executable.
The git executable must be specified in one of the following ways:
    - be included in your $PATH
    - be set via $GIT_PYTHON_GIT_EXECUTABLE
    - explicitly set via git.refresh(<full-path-to-git-executable>)
All git commands will error until this is rectified.
This initial message can be silenced or aggravated in the future by setting the
$GIT_PYTHON_REFRESH environment variable. Use one of the following values:
    - quiet|q|silence|s|silent|none|n|0: for no message or exception
    - warn|w|warning|log|l|1: for a warning message (logging level CRITICAL, displayed by default)
    - error|e|exception|raise|r|2: for a raised exception
Example:
    export GIT_PYTHON_REFRESH=quiet
urllib3.exceptions.ResponseError: too many 503 error responses
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/requests/adapters.py", line 644, in send
    resp = conn.urlopen(
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/urllib3/connectionpool.py", line 942, in urlopen
    return self.urlopen(
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/urllib3/connectionpool.py", line 942, in urlopen
    return self.urlopen(
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/urllib3/connectionpool.py", line 942, in urlopen
    return self.urlopen(
           ^^^^^^^^^^^^^
  [Previous line repeated 4 more times]
  File "/usr/local/lib/python3.11/site-packages/urllib3/connectionpool.py", line 932, in urlopen
    retries = retries.increment(method, url, response=response, _pool=self)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/urllib3/util/retry.py", line 519, in increment
    raise MaxRetryError(_pool, url, reason) from reason  # type: ignore[arg-type]
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
urllib3.exceptions.MaxRetryError: HTTPConnectionPool(host='10.17.0.223', port=5000): Max retries exceeded with url: /api/2.0/mlflow/runs/create (Caused by ResponseError('too many 503 error responses'))
During handling of the above exception, another exception occurred:
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/mlflow/utils/rest_utils.py", line 236, in http_request
    return _get_http_response_with_retries(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/mlflow/utils/request_utils.py", line 237, in _get_http_response_with_retries
    return session.request(method, url, allow_redirects=allow_redirects, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/requests/sessions.py", line 589, in request
    resp = self.send(prep, **send_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/requests/sessions.py", line 703, in send
    r = adapter.send(request, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/requests/adapters.py", line 668, in send
    raise RetryError(e, request=request)
requests.exceptions.RetryError: HTTPConnectionPool(host='10.17.0.223', port=5000): Max retries exceeded with url: /api/2.0/mlflow/runs/create (Caused by ResponseError('too many 503 error responses'))
During handling of the above exception, another exception occurred:
Traceback (most recent call last):
  File "/app/src/train.py", line 75, in <module>
    main()
  File "/app/src/train.py", line 26, in main
    with mlflow.start_run() as run:
         ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/mlflow/tracking/fluent.py", line 478, in start_run
    active_run_obj = client.create_run(
                     ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/mlflow/tracking/client.py", line 479, in create_run
    return self._tracking_client.create_run(experiment_id, start_time, tags, run_name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/mlflow/telemetry/track.py", line 30, in wrapper
    result = func(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/mlflow/tracking/_tracking_service/client.py", line 183, in create_run
    return self.store.create_run(
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/mlflow/store/tracking/rest_store.py", line 340, in create_run
    response_proto = self._call_endpoint(CreateRun, req_body)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/mlflow/store/tracking/rest_store.py", line 203, in _call_endpoint
    return call_endpoint(
           ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/mlflow/utils/rest_utils.py", line 594, in call_endpoint
    response = http_request(**call_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/mlflow/utils/rest_utils.py", line 259, in http_request
    raise MlflowException(f"API request to {url} failed with exception {e}")
mlflow.exceptions.MlflowException: API request to ***/api/2.0/mlflow/runs/create failed with exception HTTPConnectionPool(host='10.17.0.223', port=5000): Max retries exceeded with url: /api/2.0/mlflow/runs/create (Caused by ResponseError('too many 503 error responses'))
```

Do you know what's going on?
