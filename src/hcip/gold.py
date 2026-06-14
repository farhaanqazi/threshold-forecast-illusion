"""Gold layer: build the dimensional star schema from the Silver unified table and
load it into a relational store.

Defaults to a local SQLite database (zero-setup, runs anywhere). Switching to the
production PostgreSQL warehouse is a single change to the engine URL — no other code
changes — because all access goes through SQLAlchemy.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

STANDARD = 0.92
INT_MEASURES = ["total_waiting", "within_18wk", "over_18wk", "over_52wk", "over_104wk"]


def _financial_year(ts: pd.Timestamp) -> str:
    """NHS financial year (April–March), e.g. 2025-06 -> '2025/26'."""
    return f"{ts.year}/{(ts.year + 1) % 100:02d}" if ts.month >= 4 else f"{ts.year - 1}/{ts.year % 100:02d}"


def build_gold(silver: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the five Gold tables as DataFrames with surrogate keys."""
    s = silver.copy()

    # dim_region (one row per ICB)
    dim_region = (s[["icb_code", "icb_name"]].drop_duplicates()
                  .sort_values("icb_code").reset_index(drop=True))
    dim_region.insert(0, "region_key", range(1, len(dim_region) + 1))

    # dim_hospital (one row per provider; ICB taken from its most recent period)
    prov = (s.sort_values("period_date")
            .groupby("provider_code", as_index=False)
            .agg(provider_name=("provider_name", "last"), icb_code=("icb_code", "last")))
    dim_hospital = prov.merge(dim_region[["region_key", "icb_code"]], on="icb_code", how="left")
    dim_hospital.insert(0, "hospital_key", range(1, len(dim_hospital) + 1))
    dim_hospital = dim_hospital[["hospital_key", "provider_code", "provider_name", "region_key"]]

    # dim_specialty (one row per treatment function)
    dim_specialty = (s[["specialty_code", "specialty_name"]].drop_duplicates()
                     .sort_values("specialty_code").reset_index(drop=True))
    dim_specialty.insert(0, "specialty_key", range(1, len(dim_specialty) + 1))

    # dim_date (one row per month)
    dd = pd.DataFrame({"period_date": sorted(s["period_date"].unique())})
    dd["period_date"] = pd.to_datetime(dd["period_date"])
    dd["year"] = dd["period_date"].dt.year.astype("int16")
    dd["quarter"] = dd["period_date"].dt.quarter.astype("int8")
    dd["month"] = dd["period_date"].dt.month.astype("int8")
    dd["month_name"] = dd["period_date"].dt.strftime("%B")
    dd["financial_year"] = dd["period_date"].apply(_financial_year)
    # Compute at full int precision (int16 year * 100 would overflow).
    dd["date_key"] = (dd["period_date"].dt.year * 100 + dd["period_date"].dt.month).astype("int32")
    dim_date = dd[["date_key", "period_date", "year", "quarter", "month", "month_name", "financial_year"]]

    # fact_waiting_list (grain: hospital x specialty x month)
    fact = (s.merge(dim_date[["date_key", "period_date"]], on="period_date")
            .merge(dim_hospital[["hospital_key", "provider_code"]], on="provider_code")
            .merge(dim_specialty[["specialty_key", "specialty_code"]], on="specialty_code")
            .merge(dim_region[["region_key", "icb_code"]], on="icb_code"))
    fact[INT_MEASURES] = fact[INT_MEASURES].astype("int64")
    fact["is_breach"] = fact["pct_within_18wk"] < STANDARD
    fact = fact[["date_key", "hospital_key", "specialty_key", "region_key", *INT_MEASURES,
                 "pct_within_18wk", "breach_rate", "is_breach"]].reset_index(drop=True)
    fact.insert(0, "waiting_list_id", range(1, len(fact) + 1))

    return {
        "dim_date": dim_date,
        "dim_region": dim_region,
        "dim_hospital": dim_hospital,
        "dim_specialty": dim_specialty,
        "fact_waiting_list": fact,
    }


def write_gold(tables: dict[str, pd.DataFrame], engine_url: str, parquet_dir: Path | None = None) -> None:
    """Load the Gold tables into the database; optionally also export Parquet copies."""
    engine = create_engine(engine_url)
    with engine.begin() as conn:
        for name, frame in tables.items():
            frame.to_sql(name, conn, if_exists="replace", index=False)
    if parquet_dir is not None:
        parquet_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            frame.to_parquet(parquet_dir / f"{name}.parquet", index=False)
