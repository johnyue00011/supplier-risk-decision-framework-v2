#!/usr/bin/env python3
"""Build historical vendor-month feature tables for NCR supplier risk.

This stage intentionally stops at feature construction. It does not define
targets, train models, create scorecards, or assign risk bands.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEVERE_NOTIFICATION_TYPES = {"F2", "F3", "Q2"}
MISSING_RATE_COLUMNS = ["ncr_rate_3m", "severe_rate_3m", "loss_rate_3m"]


def normalize_id(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    text = text.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return text.str.replace(r"\.0$", "", regex=True)


def parse_month(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.astype("float64")
    return numerator.astype("float64").div(denominator.where(denominator > 0))


def responsibility_flags(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    text = series.astype("string").str.strip()
    known = text.notna() & text.ne("")
    supplier = text.str.contains("supplier", case=False, na=False)
    unknown = ~known
    internal = known & ~supplier
    return supplier.astype("int64"), internal.astype("int64"), unknown.astype("int64")


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["vendor_id", "month"]).copy()
    grouped = out.groupby("vendor_id", group_keys=False)

    out["ncr_count_1m"] = grouped["ncr_count"].shift(1).fillna(0)
    for window in [3, 6]:
        shifted = grouped[
            [
                "ncr_count",
                "severe_ncr_count",
                "material_value_sum",
                "received_qty_net",
                "supplier_fault_count",
                "internal_fault_count",
                "unknown_fault_count",
                "responsibility_known_count",
                "material_value_missing_count",
            ]
        ].shift(1)
        rolled = shifted.groupby(out["vendor_id"], group_keys=False).rolling(window=window, min_periods=1).sum()
        rolled = rolled.reset_index(level=0, drop=True).fillna(0)
        out[f"ncr_count_{window}m"] = rolled["ncr_count"]
        out[f"severe_ncr_count_{window}m"] = rolled["severe_ncr_count"]
        out[f"material_value_sum_{window}m"] = rolled["material_value_sum"]
        out[f"exposure_qty_{window}m"] = rolled["received_qty_net"]
        out[f"material_value_missing_count_{window}m"] = rolled["material_value_missing_count"]
        if window == 6:
            out["supplier_fault_count_6m"] = rolled["supplier_fault_count"]
            out["internal_fault_count_6m"] = rolled["internal_fault_count"]
            out["unknown_fault_count_6m"] = rolled["unknown_fault_count"]
            out["responsibility_known_count_6m"] = rolled["responsibility_known_count"]

    shifted_max = grouped[["severity_score_max", "material_value_max"]].shift(1)
    rolled_max = shifted_max.groupby(out["vendor_id"], group_keys=False).rolling(window=6, min_periods=1).max()
    rolled_max = rolled_max.reset_index(level=0, drop=True)
    out["max_severity_6m"] = rolled_max["severity_score_max"].fillna(0)
    out["material_value_max_6m"] = rolled_max["material_value_max"].fillna(0)

    out["ncr_rate_3m"] = safe_divide(out["ncr_count_3m"], out["exposure_qty_3m"])
    out["severe_rate_3m"] = safe_divide(out["severe_ncr_count_3m"], out["exposure_qty_3m"])
    out["loss_rate_3m"] = safe_divide(out["material_value_sum_3m"], out["exposure_qty_3m"])

    previous_3m_counts = grouped["ncr_count"].shift(4).groupby(out["vendor_id"], group_keys=False).rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)
    previous_3m_severe = grouped["severe_ncr_count"].shift(4).groupby(out["vendor_id"], group_keys=False).rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)
    previous_3m_exposure = grouped["received_qty_net"].shift(4).groupby(out["vendor_id"], group_keys=False).rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)
    previous_ncr_rate_3m = safe_divide(previous_3m_counts.fillna(0), previous_3m_exposure.fillna(0))
    previous_severe_rate_3m = safe_divide(previous_3m_severe.fillna(0), previous_3m_exposure.fillna(0))
    out["ncr_rate_change"] = out["ncr_rate_3m"] - previous_ncr_rate_3m
    out["severe_rate_change"] = out["severe_rate_3m"] - previous_severe_rate_3m

    responsibility_total = out[["supplier_fault_count_6m", "internal_fault_count_6m", "unknown_fault_count_6m"]].sum(axis=1)
    out["supplier_fault_share_6m"] = safe_divide(out["supplier_fault_count_6m"], responsibility_total)
    out["internal_fault_share_6m"] = safe_divide(out["internal_fault_count_6m"], responsibility_total)
    out["unknown_fault_share_6m"] = safe_divide(out["unknown_fault_count_6m"], responsibility_total)
    return out


def add_recency_features(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    out = panel.sort_values(["vendor_id", "month_start"]).copy()
    out["days_since_last_ncr"] = np.nan
    out["recent_severe_flag"] = False

    event_subset = events[events["vendor_id"].notna() & events["created_date"].notna()].copy()
    event_subset = event_subset.sort_values(["vendor_id", "created_date"])
    for vendor_id, idx in out.groupby("vendor_id").groups.items():
        vendor_events = event_subset[event_subset["vendor_id"].eq(vendor_id)]
        if vendor_events.empty:
            continue
        dates = vendor_events["created_date"].to_numpy(dtype="datetime64[ns]")
        severe_dates = vendor_events.loc[vendor_events["severe_ncr_flag"].eq(1), "created_date"].to_numpy(dtype="datetime64[ns]")
        month_starts = out.loc[idx, "month_start"].to_numpy(dtype="datetime64[ns]")

        positions = np.searchsorted(dates, month_starts, side="left") - 1
        has_prior = positions >= 0
        days = np.full(len(month_starts), np.nan)
        days[has_prior] = (month_starts[has_prior] - dates[positions[has_prior]]).astype("timedelta64[D]").astype(float)
        out.loc[idx, "days_since_last_ncr"] = days

        if len(severe_dates):
            severe_positions = np.searchsorted(severe_dates, month_starts, side="left") - 1
            has_prior_severe = severe_positions >= 0
            severe_days = np.full(len(month_starts), np.inf)
            severe_days[has_prior_severe] = (month_starts[has_prior_severe] - severe_dates[severe_positions[has_prior_severe]]).astype("timedelta64[D]").astype(float)
            out.loc[idx, "recent_severe_flag"] = severe_days <= 92
    return out


def build_monthly_event_features(ncr_event: pd.DataFrame) -> pd.DataFrame:
    events = ncr_event[ncr_event["vendor_id"].notna() & ncr_event["created_month"].notna()].copy()
    events["notification_type"] = events["notification_type"].astype("string").str.upper().str.strip()
    events["material_value"] = pd.to_numeric(events["material_value"], errors="coerce")
    events["material_value_nonmissing"] = events["material_value"].fillna(0)
    events["material_value_missing_flag"] = events["material_value"].isna().astype("int64")
    events["severe_ncr_flag"] = events["notification_type"].isin(SEVERE_NOTIFICATION_TYPES).astype("int64")
    severity_map = {"F2": 4, "F3": 4, "Q2": 4, "F4": 3, "F6": 2, "F7": 2, "E5": 1}
    events["severity_score"] = events["notification_type"].map(severity_map).fillna(0).astype("int64")
    supplier, internal, unknown = responsibility_flags(events["cause_responsibility"])
    events["supplier_fault_flag"] = supplier
    events["internal_fault_flag"] = internal
    events["unknown_fault_flag"] = unknown
    events["responsibility_known_flag"] = 1 - unknown

    monthly = events.groupby(["vendor_id", "created_month"], dropna=False).agg(
        ncr_count=("notification_number", "nunique"),
        severe_ncr_count=("severe_ncr_flag", "sum"),
        severity_score_max=("severity_score", "max"),
        material_value_sum=("material_value_nonmissing", "sum"),
        material_value_max=("material_value_nonmissing", "max"),
        material_value_missing_count=("material_value_missing_flag", "sum"),
        supplier_fault_count=("supplier_fault_flag", "sum"),
        internal_fault_count=("internal_fault_flag", "sum"),
        unknown_fault_count=("unknown_fault_flag", "sum"),
        responsibility_known_count=("responsibility_known_flag", "sum"),
    ).reset_index().rename(columns={"created_month": "month"})
    return monthly


def build_features(processed_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    ncr_event = pd.read_csv(processed_dir / "ncr_event.csv")
    exposure = pd.read_csv(processed_dir / "vendor_month_exposure.csv")
    base = pd.read_csv(processed_dir / "vendor_month_base.csv")

    for df in [ncr_event, exposure, base]:
        df["vendor_id"] = normalize_id(df["vendor_id"])
    ncr_event["created_month"] = parse_month(ncr_event["created_month"])
    ncr_event["created_date"] = pd.to_datetime(ncr_event["created_date"], errors="coerce")
    exposure["month"] = parse_month(exposure["month"])
    base["month"] = parse_month(base["month"])

    if base.duplicated(["vendor_id", "month"]).any():
        raise ValueError("vendor_month_base.csv contains duplicate vendor_id/month rows")
    if exposure.duplicated(["vendor_id", "month"]).any():
        raise ValueError("vendor_month_exposure.csv contains duplicate vendor_id/month rows")

    monthly_events = build_monthly_event_features(ncr_event)
    vendors = pd.Index(base["vendor_id"].dropna().unique(), name="vendor_id")
    months = pd.period_range(base["month"].min(), base["month"].max(), freq="M", name="month")
    panel = pd.MultiIndex.from_product([vendors, months]).to_frame(index=False)

    monthly = panel.merge(monthly_events, on=["vendor_id", "month"], how="left")
    monthly = monthly.merge(
        exposure[["vendor_id", "month", "received_qty_net"]],
        on=["vendor_id", "month"],
        how="left",
    )
    fill_zero = [
        "ncr_count",
        "severe_ncr_count",
        "severity_score_max",
        "material_value_sum",
        "material_value_max",
        "material_value_missing_count",
        "supplier_fault_count",
        "internal_fault_count",
        "unknown_fault_count",
        "responsibility_known_count",
        "received_qty_net",
    ]
    monthly[fill_zero] = monthly[fill_zero].fillna(0)
    monthly = add_rolling_features(monthly)
    monthly["month_start"] = monthly["month"].dt.to_timestamp()
    ncr_for_recency = ncr_event.copy()
    ncr_for_recency["notification_type"] = ncr_for_recency["notification_type"].astype("string").str.upper().str.strip()
    ncr_for_recency["severe_ncr_flag"] = ncr_for_recency["notification_type"].isin(SEVERE_NOTIFICATION_TYPES).astype("int64")
    monthly = add_recency_features(monthly, ncr_for_recency)

    feature_cols = [
        "ncr_count_1m",
        "ncr_count_3m",
        "ncr_count_6m",
        "severe_ncr_count_3m",
        "severe_ncr_count_6m",
        "max_severity_6m",
        "exposure_qty_3m",
        "exposure_qty_6m",
        "ncr_rate_3m",
        "severe_rate_3m",
        "loss_rate_3m",
        "material_value_sum_3m",
        "material_value_sum_6m",
        "material_value_max_6m",
        "material_value_missing_count_3m",
        "material_value_missing_count_6m",
        "days_since_last_ncr",
        "recent_severe_flag",
        "supplier_fault_share_6m",
        "internal_fault_share_6m",
        "unknown_fault_share_6m",
        "responsibility_known_count_6m",
        "ncr_rate_change",
        "severe_rate_change",
    ]
    features = base[["vendor_id", "month", "internal_supplier_candidate"]].merge(
        monthly[["vendor_id", "month", *feature_cols]],
        on=["vendor_id", "month"],
        how="left",
    )
    features["month"] = features["month"].astype("string")
    for col in ["recent_severe_flag", "internal_supplier_candidate"]:
        features[col] = features[col].astype(bool)

    features["exposure_3m_missing"] = features["exposure_qty_3m"].le(0)
    for col in MISSING_RATE_COLUMNS:
        features[f"{col}_missing"] = features[col].isna()
    features["days_since_last_ncr_missing"] = features["days_since_last_ncr"].isna()
    responsibility_cols = ["supplier_fault_share_6m", "internal_fault_share_6m", "unknown_fault_share_6m"]
    features["responsibility_share_6m_missing"] = features[responsibility_cols].isna().any(axis=1)

    if features.duplicated(["vendor_id", "month"]).any():
        raise ValueError("vendor_month_features.csv would contain duplicate vendor_id/month rows")
    if not features["internal_supplier_candidate"].equals(base["internal_supplier_candidate"].astype(bool).reset_index(drop=True)):
        raise ValueError("internal_supplier_candidate was not preserved from vendor_month_base.csv")

    stats = {
        "base_rows": len(base),
        "feature_rows": len(features),
        "unique_vendors": features["vendor_id"].nunique(),
        "month_min": features["month"].min(),
        "month_max": features["month"].max(),
        "feature_columns": feature_cols,
        "duplicate_key_rows": int(features.duplicated(["vendor_id", "month"]).sum()),
        "internal_supplier_rows": int(features["internal_supplier_candidate"].sum()),
    }
    return features, stats


def missing_table(features: pd.DataFrame) -> str:
    rows = []
    for col in features.columns:
        missing = int(features[col].isna().sum())
        if missing:
            rows.append((col, missing, missing / len(features)))
    if not rows:
        return "No missing values were found in the feature table.\n"
    text = "| Feature | Missing rows | Missing % |\n|---|---:|---:|\n"
    for col, count, pct in rows:
        text += f"| `{col}` | {count:,} | {pct:.1%} |\n"
    return text


def coverage_table(features: pd.DataFrame) -> str:
    indicators = [
        "exposure_3m_missing",
        "ncr_rate_3m_missing",
        "severe_rate_3m_missing",
        "loss_rate_3m_missing",
        "days_since_last_ncr_missing",
        "responsibility_share_6m_missing",
    ]
    text = "| Coverage indicator | Rows flagged | Coverage |\n|---|---:|---:|\n"
    for col in indicators:
        flagged = int(features[col].sum())
        coverage = 1 - flagged / len(features)
        text += f"| `{col}` | {flagged:,} | {coverage:.1%} |\n"
    return text


def write_summary(path: Path, features: pd.DataFrame, stats: dict[str, object]) -> None:
    definitions = [
        ("`ncr_count_1m`", "NCR notification count in the prior calendar month."),
        ("`ncr_count_3m`, `ncr_count_6m`", "NCR notification count in the prior 3 or 6 completed calendar months."),
        ("`severe_ncr_count_3m`, `severe_ncr_count_6m`", f"Prior-window count where `notification_type` is one of {sorted(SEVERE_NOTIFICATION_TYPES)}."),
        ("`max_severity_6m`", "Maximum prior-6-month severity score; F2/F3/Q2=4, F4=3, F6/F7=2, E5=1, unknown=0."),
        ("`exposure_qty_3m`, `exposure_qty_6m`", "Prior-window sum of net GR receipt quantity from movement type 101 minus 102."),
        ("`ncr_rate_3m`", "`ncr_count_3m / exposure_qty_3m`; missing when exposure is zero or unavailable."),
        ("`severe_rate_3m`", "`severe_ncr_count_3m / exposure_qty_3m`; missing when exposure is zero or unavailable."),
        ("`loss_rate_3m`", "`material_value_sum_3m / exposure_qty_3m`; missing when exposure is zero or unavailable."),
        ("`material_value_sum_3m`, `material_value_sum_6m`", "Prior-window sum of NCR `material_value`, treating missing event values as zero and retaining missing-count indicators."),
        ("`material_value_max_6m`", "Maximum NCR `material_value` in the prior 6 completed calendar months."),
        ("`material_value_missing_count_3m`, `material_value_missing_count_6m`", "Prior-window count of NCR events with missing `material_value`."),
        ("`days_since_last_ncr`", "Days from the first day of the vendor-month to the most recent earlier NCR created date."),
        ("`recent_severe_flag`", "True when a severe NCR occurred in the prior 92 days before the vendor-month starts."),
        ("`supplier_fault_share_6m`", "Prior-6-month share of NCRs whose `cause_responsibility` contains 'supplier'."),
        ("`internal_fault_share_6m`", "Prior-6-month share of NCRs with known non-supplier `cause_responsibility`."),
        ("`unknown_fault_share_6m`", "Prior-6-month share of NCRs with missing or blank `cause_responsibility`."),
        ("`responsibility_known_count_6m`", "Prior-6-month count of NCRs with nonmissing responsibility."),
        ("`ncr_rate_change`", "Current prior-3-month NCR rate minus the preceding 3-month NCR rate."),
        ("`severe_rate_change`", "Current prior-3-month severe NCR rate minus the preceding 3-month severe NCR rate."),
    ]
    definition_table = "| Feature | Definition |\n|---|---|\n"
    for feature, definition in definitions:
        definition_table += f"| {feature} | {definition} |\n"

    content = f"""# Feature Engineering Summary

Generated by `src/features.py`.

## Scope

This stage uses only preprocessing outputs:

- `data/processed/ncr_event.csv`
- `data/processed/vendor_month_exposure.csv`
- `data/processed/vendor_month_base.csv`

No targets, models, scores, scorecards, or risk bands are created.

## Output

| Output | Rows | Unique vendors | Month range |
|---|---:|---:|---|
| `vendor_month_features.csv` | {stats['feature_rows']:,} | {stats['unique_vendors']:,} | {stats['month_min']} to {stats['month_max']} |

## Leakage Controls

- The feature key is `vendor_id` + `month`.
- Rolling windows are calculated from completed months before the row month. For example, a `2025-06` row can use activity through `2025-05`, not `2025-06`.
- Recency features use NCR created dates strictly before the first day of the row month.
- `internal_supplier_candidate` is preserved directly from `vendor_month_base.csv`.
- Duplicate `vendor_id` + `month` rows are validated before writing output. Duplicate key rows found: {stats['duplicate_key_rows']:,}.

## Feature Definitions

{definition_table}

## Missing Values

{missing_table(features)}

## Feature Coverage

{coverage_table(features)}

## Notes

- Severe NCRs are defined from available current-event metadata, using `notification_type` only. This keeps the feature deterministic and avoids future disposition or completion outcomes.
- Responsibility shares use historical `cause_responsibility`; missing responsibility is kept as its own share rather than dropped.
- Exposure-normalized rates are intentionally missing when prior net exposure is not positive, with explicit missing indicators included.
"""
    path.write_text(content, encoding="utf-8")


def run(processed_dir: Path, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    features, stats = build_features(processed_dir)
    features.to_csv(processed_dir / "vendor_month_features.csv", index=False)
    write_summary(reports_dir / "feature_engineering_summary.md", features, stats)
    print("Feature engineering complete")
    print(f"vendor_month_features rows: {len(features):,}")
    print(f"unique vendors: {features['vendor_id'].nunique():,}")
    print(f"outputs: {processed_dir / 'vendor_month_features.csv'}, {reports_dir / 'feature_engineering_summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build historical NCR supplier risk features.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    run(args.processed_dir, args.reports_dir)


if __name__ == "__main__":
    main()
