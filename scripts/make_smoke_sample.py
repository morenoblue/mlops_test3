#!/usr/bin/env python3
# Read just enough Parquet row-groups to reach N rows, then write a tiny sample.

from pathlib import Path
import os
import pyarrow as pa
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parents[1]
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
