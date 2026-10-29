"""Tiny Parquet table store.

Each table is a folder of Parquet files, one per partition (a season, a date, ...).
Writing a partition replaces that file, so re-running a job is safe (idempotent).
DuckDB reads the whole folder as one table.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from nhlmodel.config import data_dir


def table_dir(table: str) -> Path:
    p = data_dir() / table
    p.mkdir(parents=True, exist_ok=True)
    return p


def write(table: str, df: pd.DataFrame, partition: str) -> Path:
    """Replace one partition of a table."""
    path = table_dir(table) / f"{partition}.parquet"
    tmp = path.with_suffix(".tmp")
    df.reset_index(drop=True).to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def append(table: str, df: pd.DataFrame, partition: str, key: list[str] | None = None) -> Path:
    """Add rows to a partition. With `key`, newer rows replace older rows with the same key."""
    path = table_dir(table) / f"{partition}.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)
        if key:
            df = df.drop_duplicates(subset=key, keep="last")
    return write(table, df, partition)


def exists(table: str, partition: str | None = None) -> bool:
    d = data_dir() / table
    if partition is None:
        return d.exists() and any(d.glob("*.parquet"))
    return (d / f"{partition}.parquet").exists()


def read(table: str, where: str | None = None, partitions: list[str] | None = None) -> pd.DataFrame:
    """Read a whole table (or some partitions) into pandas. Empty frame if missing."""
    d = data_dir() / table
    if partitions is not None:
        # a partition name also matches its sub-partitions: "20262027" -> 20262027_2026-10-27.parquet
        files = []
        for p in partitions:
            files += [str(f) for f in [d / f"{p}.parquet"] if f.exists()]
            files += sorted(str(f) for f in d.glob(f"{p}_*.parquet"))
    else:
        files = sorted(str(f) for f in d.glob("*.parquet")) if d.exists() else []
    if not files:
        return pd.DataFrame()
    sql = "SELECT * FROM read_parquet($files, union_by_name=true)"
    if where:
        sql += f" WHERE {where}"
    con = duckdb.connect()
    try:
        return con.execute(sql, {"files": files}).df()
    finally:
        con.close()


def query(sql: str, **tables: str) -> pd.DataFrame:
    """Run SQL where each keyword maps a view name to a table, e.g. query("select ...", s="shots")."""
    con = duckdb.connect()
    try:
        for name, table in tables.items():
            d = data_dir() / table
            files = sorted(str(f) for f in d.glob("*.parquet")) if d.exists() else []
            if not files:
                raise FileNotFoundError(f"table {table!r} has no data")
            con.execute(
                f"CREATE VIEW {name} AS SELECT * FROM read_parquet({files!r}, union_by_name=true)"
            )
        return con.execute(sql).df()
    finally:
        con.close()
