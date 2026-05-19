#!/usr/bin/env python3
"""Build preprocessing outputs for the NCR supplier risk project.

This script intentionally stops at clean event/exposure/base tables. It does
not define targets, train models, create rolling features, or build scorecards.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_FILES = {
    "ncr_all": "2.11.26all.xlsx",
    "ncr_task_report": "All Open&Closed NCR report.xlsx",
    "gr_history": "Gred 2.9.25-2.9.26.xlsx",
    "ncr_vendor": "NCR_Closed_Open_Supplier_Vendor_ID.xlsx",
}

INTERNAL_SUPPLIER_CANDIDATES = {"1000200000", "2000004123", "2000044404"}


@dataclass(frozen=True)
class BuildStats:
    ncr_all_rows: int
    ncr_vendor_rows: int
    ncr_task_rows: int
    gr_rows: int
    ncr_event_rows: int
    vendor_month_exposure_rows: int
    vendor_month_base_rows: int
    duplicate_ncr_vendor_rows_removed: int
    duplicate_ncr_all_rows_removed: int
    duplicate_task_rows_collapsed: int
    notification_overlap: int
    notification_overlap_vendor_match: int
    notification_overlap_vendor_conflict: int
    task_notifications_covered: int
    ncr_events_missing_vendor: int
    ncr_events_with_vendor: int
    gr_rows_missing_vendor: int
    gr_rows_excluded_bad_movement: int
    vendor_month_keys_with_both_gr_and_ncr: int


def snake_case(value: object) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip())
    return re.sub(r"_+", "_", text).strip("_").lower() or "unnamed"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [snake_case(col) for col in out.columns]
    return out


def normalize_id(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    text = text.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return text.str.replace(r"\.0$", "", regex=True)


def get_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(pd.NA, index=df.index, dtype="object")


def coalesce(*series: pd.Series) -> pd.Series:
    result = series[0].copy()
    for next_series in series[1:]:
        result = result.combine_first(next_series)
    return result


def first_non_null(values: pd.Series):
    non_null = values.dropna()
    return pd.NA if non_null.empty else non_null.iloc[0]


def mode_or_first(values: pd.Series):
    non_null = values.dropna()
    if non_null.empty:
        return pd.NA
    modes = non_null.mode()
    return non_null.iloc[0] if modes.empty else modes.iloc[0]


def parse_date(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce").dt.date
    numeric = pd.to_numeric(series, errors="coerce")
    valid_serial = numeric.where(numeric.between(20000, 60000))
    from_serial = pd.to_datetime(valid_serial, unit="D", origin="1899-12-30", errors="coerce")
    generic_input = series.mask(numeric.notna())
    generic = pd.to_datetime(generic_input, errors="coerce")
    return generic.combine_first(from_serial).dt.date


def parse_month(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").astype("string")


def read_excel(raw_dir: Path, key: str, **kwargs) -> pd.DataFrame:
    path = raw_dir / RAW_FILES[key]
    if not path.exists():
        raise FileNotFoundError(f"Missing raw file: {path}")
    return pd.read_excel(path, **kwargs)


def collapse_by_notification(df: pd.DataFrame, notification_col: str, aggregations: dict[str, str]) -> pd.DataFrame:
    working = df.copy()
    working[notification_col] = normalize_id(working[notification_col])
    working = working[working[notification_col].notna()]
    agg_map = {}
    for column, method in aggregations.items():
        if column not in working.columns:
            continue
        if method == "first":
            agg_map[column] = first_non_null
        elif method == "mode":
            agg_map[column] = mode_or_first
        elif method == "sum":
            agg_map[column] = "sum"
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    collapsed = working.groupby(notification_col, dropna=False).agg(agg_map).reset_index()
    sizes = working.groupby(notification_col, dropna=False).size().rename("source_row_count").reset_index()
    return collapsed.merge(sizes, on=notification_col, how="left")


def build_ncr_event(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    ncr_all_raw = read_excel(raw_dir, "ncr_all", sheet_name="Sheet1")
    ncr_vendor_raw = read_excel(raw_dir, "ncr_vendor", sheet_name="Sheet1")
    task_raw = read_excel(raw_dir, "ncr_task_report", sheet_name="Sheet1")
    ncr_all = normalize_columns(ncr_all_raw)
    ncr_vendor = normalize_columns(ncr_vendor_raw)
    task = normalize_columns(task_raw)

    vendor = collapse_by_notification(ncr_vendor, "notification", {
        "vendor_id": "first",
        "supplier_name": "mode",
        "material_number": "first",
        "material_description": "first",
        "notification_created_date": "first",
        "notification_completion_date": "first",
        "notification_qty": "first",
        "material_value": "first",
        "disposition": "mode",
        "cause_responsibility": "mode",
        "notification_type": "first",
        "project": "first",
        "notification_status": "first",
        "dashboard_status": "mode",
    }).rename(columns={"notification": "notification_number", "source_row_count": "ncr_vendor_source_row_count"})
    vendor["vendor_id"] = normalize_id(vendor["vendor_id"])

    ncr_all_event = collapse_by_notification(ncr_all, "notification", {
        "supplier": "first",
        "material": "first",
        "material_description": "first",
        "created_on": "first",
        "completion_by_date": "first",
        "notification_date": "first",
        "complaint_quantity": "first",
        "notification_type": "first",
        "notification_status": "first",
        "coding_code_text": "mode",
        "code_group_text": "mode",
    }).rename(columns={"notification": "notification_number", "supplier": "supplier_vendor_id", "source_row_count": "ncr_all_source_row_count"})
    ncr_all_event["supplier_vendor_id"] = normalize_id(ncr_all_event["supplier_vendor_id"])

    task_event = collapse_by_notification(task, "notification_number", {
        "supplier_name": "mode",
        "material_number": "first",
        "material_description": "first",
        "notification_created_date": "first",
        "notification_completion_date": "first",
        "notification_qty": "first",
        "material_value": "first",
        "disposition": "mode",
        "cause_responsibility": "mode",
        "notification_type": "first",
        "project": "first",
        "notification_status": "first",
        "dashboard_status": "mode",
    }).rename(columns={"source_row_count": "task_source_row_count"})

    event = vendor.merge(ncr_all_event, on="notification_number", how="outer", suffixes=("_vendor", "_all"), indicator="ncr_vendor_all_merge")
    event = event.merge(task_event, on="notification_number", how="left", suffixes=("", "_task"))
    event["in_ncr_vendor_file"] = event["ncr_vendor_all_merge"].isin(["left_only", "both"])
    event["in_ncr_all_file"] = event["ncr_vendor_all_merge"].isin(["right_only", "both"])
    event["in_task_report"] = event["task_source_row_count"].notna()

    event["vendor_id"] = coalesce(get_series(event, "vendor_id"), get_series(event, "supplier_vendor_id"))
    event["supplier_name"] = coalesce(get_series(event, "supplier_name"), get_series(event, "supplier_name_task"))
    event["material"] = coalesce(get_series(event, "material_number"), get_series(event, "material"), get_series(event, "material_number_task"))
    event["material_description"] = coalesce(get_series(event, "material_description"), get_series(event, "material_description_all"), get_series(event, "material_description_task"))
    event["created_date"] = coalesce(parse_date(get_series(event, "notification_created_date")), parse_date(get_series(event, "created_on")), parse_date(get_series(event, "notification_date")), parse_date(get_series(event, "notification_created_date_task")))
    event["completion_date"] = coalesce(parse_date(get_series(event, "notification_completion_date")), parse_date(get_series(event, "completion_by_date")), parse_date(get_series(event, "notification_completion_date_task")))
    event["qty"] = coalesce(pd.to_numeric(get_series(event, "notification_qty"), errors="coerce"), pd.to_numeric(get_series(event, "complaint_quantity"), errors="coerce"), pd.to_numeric(get_series(event, "notification_qty_task"), errors="coerce"))
    event["material_value"] = coalesce(pd.to_numeric(get_series(event, "material_value"), errors="coerce"), pd.to_numeric(get_series(event, "material_value_task"), errors="coerce"))
    event["disposition"] = coalesce(get_series(event, "disposition"), get_series(event, "disposition_task"))
    event["cause_responsibility"] = coalesce(get_series(event, "cause_responsibility"), get_series(event, "cause_responsibility_task"))
    event["notification_type"] = coalesce(get_series(event, "notification_type"), get_series(event, "notification_type_all"), get_series(event, "notification_type_task"))
    event["project"] = coalesce(get_series(event, "project"), get_series(event, "project_task"))
    event["notification_status"] = coalesce(get_series(event, "notification_status"), get_series(event, "notification_status_all"), get_series(event, "notification_status_task"))
    event["dashboard_status"] = coalesce(get_series(event, "dashboard_status"), get_series(event, "dashboard_status_task"))
    event["internal_supplier_candidate"] = event["vendor_id"].isin(INTERNAL_SUPPLIER_CANDIDATES)
    event["created_month"] = parse_month(event["created_date"])

    overlap = event[event["in_ncr_vendor_file"] & event["in_ncr_all_file"]].copy()
    output_cols = [
        "notification_number", "vendor_id", "supplier_name", "material", "material_description",
        "created_date", "completion_date", "created_month", "qty", "material_value",
        "disposition", "cause_responsibility", "notification_type", "project",
        "notification_status", "dashboard_status", "internal_supplier_candidate",
        "in_ncr_vendor_file", "in_ncr_all_file", "in_task_report",
        "ncr_vendor_source_row_count", "ncr_all_source_row_count", "task_source_row_count",
    ]
    event = event[output_cols].sort_values("notification_number").reset_index(drop=True)
    stats = {
        "ncr_all_rows": len(ncr_all_raw),
        "ncr_vendor_rows": len(ncr_vendor_raw),
        "ncr_task_rows": len(task_raw),
        "duplicate_ncr_vendor_rows_removed": len(ncr_vendor_raw) - vendor["notification_number"].nunique(),
        "duplicate_ncr_all_rows_removed": len(ncr_all_raw) - ncr_all_event["notification_number"].nunique(),
        "duplicate_task_rows_collapsed": len(task_raw) - task_event["notification_number"].nunique(),
        "notification_overlap": len(overlap),
        "notification_overlap_vendor_match": int((overlap["vendor_id"].notna() & overlap["supplier_vendor_id"].notna() & (overlap["vendor_id"] == overlap["supplier_vendor_id"])).sum()) if "supplier_vendor_id" in overlap.columns else 0,
        "notification_overlap_vendor_conflict": int((overlap["vendor_id"].notna() & overlap["supplier_vendor_id"].notna() & (overlap["vendor_id"] != overlap["supplier_vendor_id"])).sum()) if "supplier_vendor_id" in overlap.columns else 0,
        "task_notifications_covered": int(event["in_task_report"].sum()),
    }
    return event, stats


def build_vendor_month_exposure(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    gr_raw = read_excel(raw_dir, "gr_history", sheet_name="Sheet1")
    gr = normalize_columns(gr_raw)
    gr["vendor_id"] = normalize_id(gr["vendor"])
    gr["movement_type"] = normalize_id(gr["movement_type"])
    gr["posting_date"] = parse_date(gr["posting_date"])
    gr["month"] = parse_month(gr["posting_date"])
    gr["quantity_numeric"] = pd.to_numeric(gr["quantity"], errors="coerce")
    valid_movement = gr["movement_type"].isin(["101", "102"])
    valid_vendor = gr["vendor_id"].notna()
    lines = gr[valid_movement & valid_vendor].copy()
    lines["abs_qty"] = lines["quantity_numeric"].abs()
    lines["signed_qty"] = np.where(lines["movement_type"].eq("101"), lines["abs_qty"], -lines["abs_qty"])
    lines["received_qty_101_line"] = np.where(lines["movement_type"].eq("101"), lines["abs_qty"], 0.0)
    lines["received_qty_102_line"] = np.where(lines["movement_type"].eq("102"), lines["abs_qty"], 0.0)
    exposure = lines.groupby(["vendor_id", "month"], dropna=False).agg(
        received_qty_101=("received_qty_101_line", "sum"),
        received_qty_102=("received_qty_102_line", "sum"),
        received_qty_net=("signed_qty", "sum"),
        gr_line_count=("signed_qty", "size"),
        gr_101_line_count=("movement_type", lambda s: int((s == "101").sum())),
        gr_102_line_count=("movement_type", lambda s: int((s == "102").sum())),
        unique_material_count=("material", pd.Series.nunique),
        unique_purchase_order_count=("purchase_order", pd.Series.nunique),
    ).reset_index().sort_values(["vendor_id", "month"])
    stats = {
        "gr_rows": len(gr_raw),
        "gr_rows_missing_vendor": int(gr["vendor_id"].isna().sum()),
        "gr_rows_excluded_bad_movement": int((~valid_movement).sum()),
    }
    return exposure, stats


def build_vendor_month_base(ncr_event: pd.DataFrame, exposure: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    ncr_with_vendor = ncr_event[ncr_event["vendor_id"].notna() & ncr_event["created_month"].notna()].copy()
    ncr_month = ncr_with_vendor.groupby(["vendor_id", "created_month"], dropna=False).agg(
        ncr_event_count=("notification_number", "nunique"),
        ncr_qty_total=("qty", "sum"),
        ncr_material_value_total=("material_value", "sum"),
        supplier_responsible_ncr_count=("cause_responsibility", lambda s: int(s.astype("string").str.contains("supplier", case=False, na=False).sum())),
        open_ncr_count=("dashboard_status", lambda s: int(s.astype("string").str.contains("open", case=False, na=False).sum())),
        closed_ncr_count=("dashboard_status", lambda s: int(s.astype("string").str.contains("closed", case=False, na=False).sum())),
    ).reset_index().rename(columns={"created_month": "month"})

    base_index = pd.concat(
        [
            exposure[["vendor_id", "month"]],
            ncr_month[["vendor_id", "month"]],
        ],
        ignore_index=True,
    ).drop_duplicates().sort_values(["vendor_id", "month"]).reset_index(drop=True)
    base = base_index.merge(exposure, on=["vendor_id", "month"], how="left", indicator="exposure_merge")
    base = base.merge(ncr_month, on=["vendor_id", "month"], how="left", indicator="ncr_merge")
    for col in ["received_qty_101", "received_qty_102", "received_qty_net", "gr_line_count", "gr_101_line_count", "gr_102_line_count", "unique_material_count", "unique_purchase_order_count", "ncr_event_count", "ncr_qty_total", "ncr_material_value_total", "supplier_responsible_ncr_count", "open_ncr_count", "closed_ncr_count"]:
        base[col] = base[col].fillna(0)
    for col in ["gr_line_count", "gr_101_line_count", "gr_102_line_count", "unique_material_count", "unique_purchase_order_count", "ncr_event_count", "supplier_responsible_ncr_count", "open_ncr_count", "closed_ncr_count"]:
        base[col] = base[col].astype("int64")
    base["has_gr_exposure"] = base["exposure_merge"].eq("both")
    base["has_ncr_event_same_month"] = base["ncr_merge"].eq("both")
    base["internal_supplier_candidate"] = base["vendor_id"].isin(INTERNAL_SUPPLIER_CANDIDATES)
    base = base.drop(columns=["exposure_merge", "ncr_merge"]).sort_values(["vendor_id", "month"]).reset_index(drop=True)
    both_keys = set(zip(exposure["vendor_id"], exposure["month"])) & set(zip(ncr_month["vendor_id"], ncr_month["month"]))
    return base, {"vendor_month_keys_with_both_gr_and_ncr": len(both_keys)}


def date_range_text(series: pd.Series) -> str:
    parsed = pd.to_datetime(series, errors="coerce")
    return "none" if parsed.notna().sum() == 0 else f"{parsed.min().date()} to {parsed.max().date()}"


def write_quality_summary(path: Path, stats: BuildStats, ncr_event: pd.DataFrame, exposure: pd.DataFrame, base: pd.DataFrame) -> None:
    missing = ncr_event[["vendor_id", "supplier_name", "material", "created_date", "completion_date", "qty", "material_value", "cause_responsibility"]].isna().sum()
    content = f"""# Data Quality Summary

Generated by `src/preprocess.py`.

## Output Row Counts

| Output | Rows | Unique vendors |
|---|---:|---:|
| `ncr_event.csv` | {stats.ncr_event_rows:,} | {ncr_event['vendor_id'].nunique():,} |
| `vendor_month_exposure.csv` | {stats.vendor_month_exposure_rows:,} | {exposure['vendor_id'].nunique():,} |
| `vendor_month_base.csv` | {stats.vendor_month_base_rows:,} | {base['vendor_id'].nunique():,} |

## Raw Input Row Counts

| Raw file | Rows used |
|---|---:|
| NCR All | {stats.ncr_all_rows:,} |
| NCR Open & Closed vendor map | {stats.ncr_vendor_rows:,} |
| NCR task-level report `Sheet1` | {stats.ncr_task_rows:,} |
| GR History | {stats.gr_rows:,} |

## Date Ranges

| Dataset | Date range |
|---|---|
| NCR event created date | {date_range_text(ncr_event['created_date'])} |
| NCR event completion date | {date_range_text(ncr_event['completion_date'])} |
| GR posting month | {exposure['month'].min()} to {exposure['month'].max()} |
| Vendor-month base | {base['month'].min()} to {base['month'].max()} |

## Duplicate Handling

- `ncr_event.csv` is keyed by `notification_number`, using Notification Number as the NCR event key.
- NCR vendor-map duplicates collapsed: {stats.duplicate_ncr_vendor_rows_removed:,} rows beyond one row per notification.
- NCR All duplicates collapsed: {stats.duplicate_ncr_all_rows_removed:,} rows beyond one row per notification.
- Task-level NCR report rows collapsed: {stats.duplicate_task_rows_collapsed:,} rows beyond one row per notification.
- Task-level rows were used only for notification-level enrichment and QA, not as event rows.

## Merge Coverage

- NCR All / vendor-map overlapping notifications: {stats.notification_overlap:,}.
- Overlapping notifications with matching vendor IDs: {stats.notification_overlap_vendor_match:,}.
- Overlapping notifications with conflicting vendor IDs: {stats.notification_overlap_vendor_conflict:,}.
- Event notifications covered by task-level report: {stats.task_notifications_covered:,}.
- Vendor-month keys with both GR exposure and NCR activity in the same month: {stats.vendor_month_keys_with_both_gr_and_ncr:,}.

## Missing Vendor Coverage

- NCR events with vendor ID: {stats.ncr_events_with_vendor:,}.
- NCR events missing vendor ID: {stats.ncr_events_missing_vendor:,}.
- GR rows missing vendor ID and excluded from vendor exposure: {stats.gr_rows_missing_vendor:,}.
- GR rows excluded due to movement type outside explicit 101/102 logic: {stats.gr_rows_excluded_bad_movement:,}.

## NCR Missing Value Counts

| Field | Missing rows |
|---|---:|
"""
    for field, count in missing.items():
        content += f"| `{field}` | {int(count):,} |\n"
    content += """
## GR Movement-Type Logic

- `received_qty_101 = sum(abs(Quantity))` for movement type `101`.
- `received_qty_102 = sum(abs(Quantity))` for movement type `102` reversal QA volume.
- `received_qty_net = received_qty_101 - received_qty_102`.

## Internal / Strategic Suppliers

Internal or strategic supplier candidates were not excluded. The outputs include `internal_supplier_candidate` for known candidates: `1000200000`, `2000004123`, `2000044404`.

## Known Limitations

- NCR history spans several years, while GR exposure is available only for the supplied GR extract period.
- Some notification/vendor mappings conflict between NCR All and the vendor-mapped NCR file; the vendor-mapped file is preferred when available and conflicts are summarized above.
- Supplier names are retained for audit context, but `vendor_id` is the supplier key in processed outputs.
- Material value and quantity outliers are preserved for downstream analysis; no target definition or modeling treatment is applied here.
"""
    path.write_text(content, encoding="utf-8")


def run(raw_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ncr_event, ncr_stats = build_ncr_event(raw_dir)
    exposure, exposure_stats = build_vendor_month_exposure(raw_dir)
    base, base_stats = build_vendor_month_base(ncr_event, exposure)
    if ncr_event["notification_number"].duplicated().any():
        raise ValueError("ncr_event.csv would contain duplicate notification_number values")
    if exposure.duplicated(["vendor_id", "month"]).any():
        raise ValueError("vendor_month_exposure.csv would contain duplicate vendor_id/month rows")
    if base.duplicated(["vendor_id", "month"]).any():
        raise ValueError("vendor_month_base.csv would contain duplicate vendor_id/month rows")
    stats = BuildStats(
        ncr_all_rows=ncr_stats["ncr_all_rows"],
        ncr_vendor_rows=ncr_stats["ncr_vendor_rows"],
        ncr_task_rows=ncr_stats["ncr_task_rows"],
        gr_rows=exposure_stats["gr_rows"],
        ncr_event_rows=len(ncr_event),
        vendor_month_exposure_rows=len(exposure),
        vendor_month_base_rows=len(base),
        duplicate_ncr_vendor_rows_removed=ncr_stats["duplicate_ncr_vendor_rows_removed"],
        duplicate_ncr_all_rows_removed=ncr_stats["duplicate_ncr_all_rows_removed"],
        duplicate_task_rows_collapsed=ncr_stats["duplicate_task_rows_collapsed"],
        notification_overlap=ncr_stats["notification_overlap"],
        notification_overlap_vendor_match=ncr_stats["notification_overlap_vendor_match"],
        notification_overlap_vendor_conflict=ncr_stats["notification_overlap_vendor_conflict"],
        task_notifications_covered=ncr_stats["task_notifications_covered"],
        ncr_events_missing_vendor=int(ncr_event["vendor_id"].isna().sum()),
        ncr_events_with_vendor=int(ncr_event["vendor_id"].notna().sum()),
        gr_rows_missing_vendor=exposure_stats["gr_rows_missing_vendor"],
        gr_rows_excluded_bad_movement=exposure_stats["gr_rows_excluded_bad_movement"],
        vendor_month_keys_with_both_gr_and_ncr=base_stats["vendor_month_keys_with_both_gr_and_ncr"],
    )
    ncr_event.to_csv(output_dir / "ncr_event.csv", index=False)
    exposure.to_csv(output_dir / "vendor_month_exposure.csv", index=False)
    base.to_csv(output_dir / "vendor_month_base.csv", index=False)
    write_quality_summary(output_dir / "data_quality_summary.md", stats, ncr_event, exposure, base)
    print("Preprocessing complete")
    print(f"ncr_event rows: {len(ncr_event):,}")
    print(f"vendor_month_exposure rows: {len(exposure):,}")
    print(f"vendor_month_base rows: {len(base):,}")
    print(f"outputs: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NCR preprocessing outputs.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    run(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
