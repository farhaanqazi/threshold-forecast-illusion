"""Data-validation gates for the feature table (schema, ranges, grain).

The models are only as trustworthy as their inputs; these checks fail the build if the
feature contract is violated.
"""
try:
    import pandera.pandas as pa
except ImportError:  # older pandera
    import pandera as pa

from hcip import modeling as M

FEATURE_SCHEMA = pa.DataFrameSchema(
    {
        "provider_code": pa.Column(nullable=False, required=True),
        "specialty_code": pa.Column(nullable=False, required=True),
        "period_date": pa.Column(nullable=False, required=True),
        "total_waiting": pa.Column(nullable=True, checks=pa.Check.ge(0)),
        "within_18wk": pa.Column(nullable=True, checks=pa.Check.ge(0)),
        "over_18wk": pa.Column(nullable=True, checks=pa.Check.ge(0)),
        "over_52wk": pa.Column(nullable=True, checks=pa.Check.ge(0)),
        "over_104wk": pa.Column(nullable=True, checks=pa.Check.ge(0)),
        "pct_within_18wk": pa.Column(nullable=True, checks=pa.Check.in_range(0, 1)),
        "breach_rate": pa.Column(nullable=True, checks=pa.Check.in_range(0, 1)),
        "over_52_share": pa.Column(nullable=True, checks=pa.Check.in_range(0, 1)),
        "over_104_share": pa.Column(nullable=True, checks=pa.Check.in_range(0, 1)),
    },
    strict=False,
)


def test_feature_schema_holds(df):
    FEATURE_SCHEMA.validate(df, lazy=True)


def test_grain_is_unique(df):
    dupes = df.duplicated(["period_date", "provider_code", "specialty_code"]).sum()
    assert dupes == 0, f"{dupes} duplicate rows at the modelling grain"


def test_reconciliation(df):
    """within + over must equal total (the wait-band partition must be exhaustive)."""
    sub = df.dropna(subset=["within_18wk", "over_18wk", "total_waiting"])
    assert (sub["within_18wk"] + sub["over_18wk"] == sub["total_waiting"]).all()
