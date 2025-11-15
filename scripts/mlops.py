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
