import os
import time
import importlib
import mlflow
from math import sqrt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# I'm adding this cuz I want the workflow to run again

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

