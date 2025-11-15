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
