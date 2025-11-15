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
