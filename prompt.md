When I push this, to my repo (assuming there is nothing in my mlflow yet) will it work. in the end if I manually ssh into my prod machine
will I be able to curl to localhost:8000/health and get a reponse?

will both my ci and cd run?


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
name: CI (build+smoke+push)

on:
  push:
  pull_request:

jobs:
  build-train-serve:
    runs-on: ubuntu-latest
    permissions: { contents: read, packages: write }
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3

      - name: Build train
        uses: docker/build-push-action@v6
        with:
          context: .
          file: Dockerfile
          load: true
          tags: |
            ghcr.io/${{ github.repository }}:training-${{ github.sha }}

      - name: Smoke train (local file store)
        run: |
          docker run --rm \
            -e MLFLOW_TRACKING_URI="file:/tmp/mlruns" \
            -e DATA_PATH="/app/src/data/smoke_sample.parquet" \
            -e TRAIN_NROWS="256" \
            ghcr.io/${{ github.repository }}:training-${{ github.sha }}

      - name: Build serve
        uses: docker/build-push-action@v6
        with:
          context: .
          file: Dockerfile.serve
          load: true
          tags: |
            ghcr.io/${{ github.repository }}:serving-${{ github.sha }}

      - name: Smoke serve
        run: |
          docker run -d --name serve -p 18000:8000 ghcr.io/${{ github.repository }}:serving-${{ github.sha }}
          sleep 2
          curl -fsS http://localhost:18000/health
          docker rm -f serve

      - name: Login GHCR
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Push images
        if: github.event_name != 'pull_request'
        run: |
          docker push ghcr.io/${{ github.repository }}:training-${{ github.sha }}
          docker push ghcr.io/${{ github.repository }}:serving-${{ github.sha }}
```

cd.yml
```yml
name: CD (train→staging→prod canary)

on:
  workflow_run:
    workflows: ["CI (build+smoke+push)"]
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
        run: docker pull ghcr.io/${{ github.repository }}:training-${{ github.event.workflow_run.head_sha }}

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
            ghcr.io/${{ github.repository }}:training-${{ github.event.workflow_run.head_sha }} | tee out.txt

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
        run: docker pull ghcr.io/${{ github.repository }}:serving-${{ github.event.workflow_run.head_sha }}

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
            ghcr.io/${{ github.repository }}:serving-${{ github.event.workflow_run.head_sha }}
          sleep 3
          curl -fsS http://localhost:8000/health
          curl -fsS -X POST http://localhost:8000/predict \
            -H "Content-Type: application/json" \
            -d '{"records":[{"vendor_id":1,"pickup_datetime":"2010-01-15T08:30:00","dropoff_datetime":"2010-01-15T08:45:00","passenger_count":1,"trip_distance":2.3,"pickup_longitude":-73.99,"pickup_latitude":40.73,"rate_code":1,"dropoff_longitude":-73.98,"dropoff_latitude":40.75,"fare_amount":12.5,"surcharge":0.5,"mta_tax":0.5,"tip_amount":2.0,"tolls_amount":0.0,"total_amount":15.5,"store_and_fwd_flag":"N","payment_type":"CRD"}]}'

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
        run: docker pull ghcr.io/${{ github.repository }}:serving-${{ github.event.workflow_run.head_sha }}

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
            ghcr.io/${{ github.repository }}:serving-${{ github.event.workflow_run.head_sha }}
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
