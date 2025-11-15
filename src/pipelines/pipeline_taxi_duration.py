import os
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils.io import read_parquet
from utils.preprocessing.s1 import preprocess_data

BASE = Path(__file__).resolve().parents[1]
DEFAULT_DATA = BASE / "data" / "smoke_sample.parquet"

def load_data():
    path = os.getenv("DATA_PATH", str(DEFAULT_DATA))
    nrows_env = os.getenv("TRAIN_NROWS", "")
    nrows = int(nrows_env) if (nrows_env and nrows_env.isdigit()) else 0
    nrows = None if nrows == 0 else nrows

    df = read_parquet(path, nrows=nrows)
    dfp = preprocess_data(df)
    y = dfp["trip_duration"].astype(np.float32)
    X = dfp.drop(columns=["trip_duration"])
    return X, y

def build(model):
    return Pipeline(steps=[("scaler", StandardScaler()), ("model", model)])
