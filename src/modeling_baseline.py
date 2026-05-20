#!/usr/bin/env python3
"""Baseline point-in-time supplier risk ranking model.

This stage builds a deliberately simple logistic regression baseline for
ranking vendor-month observations by disproportionate future operational loss.
It avoids scorecards, boosted trees, deep learning, and aggressive tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FUTURE_WINDOW_MONTHS = 3
RISK_THRESHOLD_QUANTILE = 0.80
MODEL_ITERATIONS = 4_000
LEARNING_RATE = 0.05
L2_PENALTY = 0.5

DISPOSITION_SEVERE_PATTERN = r"scrap|rtv|mrb"
SUPPLIER_SEVERE_DISPOSITION_PATTERN = r"scrap|rtv|mrb|repair|rework|reject|concession"

FEATURE_COLUMNS = [
    "internal_supplier_candidate",
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
    "exposure_3m_missing",
    "ncr_rate_3m_missing",
    "severe_rate_3m_missing",
    "loss_rate_3m_missing",
    "days_since_last_ncr_missing",
    "responsibility_share_6m_missing",
]

LOG1P_COLUMNS = {
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
    "responsibility_known_count_6m",
    "ncr_rate_change",
    "severe_rate_change",
}


def normalize_id(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    text = text.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return text.str.replace(r"\.0$", "", regex=True)


def parse_month(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.PeriodDtype):
        return series
    return pd.to_datetime(series, errors="coerce").dt.to_period("M")


def sigmoid(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0, 1 / (1 + np.exp(-values)), np.exp(values) / (1 + np.exp(values)))


def month_add(month: pd.Period, months: int) -> pd.Period:
    return month + months


def add_point_in_time_severe_flags(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["material_value"] = pd.to_numeric(out["material_value"], errors="coerce")
    out["material_value_nonmissing"] = out["material_value"].fillna(0)
    out["created_month"] = parse_month(out["created_month"])
    disposition = out["disposition"].astype("string").fillna("").str.lower()
    responsibility = out["cause_responsibility"].astype("string").fillna("").str.lower()

    material_thresholds: dict[pd.Period, float] = {}
    prior_values = out.loc[out["material_value"].notna(), ["created_month", "material_value"]].sort_values("created_month")
    for month in sorted(out["created_month"].dropna().unique()):
        material_thresholds[month] = prior_values.loc[prior_values["created_month"].lt(month), "material_value"].quantile(0.80)

    out["prior_material_value_p80"] = out["created_month"].map(material_thresholds).astype("float64")
    high_material = out["material_value"].gt(out["prior_material_value_p80"])
    severe_disposition = disposition.str.contains(DISPOSITION_SEVERE_PATTERN, case=False, regex=True, na=False)
    supplier_responsible_severe = (
        responsibility.str.contains("supplier", case=False, regex=False, na=False)
        & disposition.str.contains(SUPPLIER_SEVERE_DISPOSITION_PATTERN, case=False, regex=True, na=False)
    )
    out["severe_ncr_operational_flag"] = (high_material | severe_disposition | supplier_responsible_severe).astype("int64")
    return out


def future_sum_by_vendor_month(panel: pd.DataFrame, value_column: str) -> pd.Series:
    grouped = panel.groupby("vendor_id", sort=False)[value_column]
    future_parts = [grouped.shift(-offset).fillna(0) for offset in range(1, FUTURE_WINDOW_MONTHS + 1)]
    return sum(future_parts)


def build_modeling_dataset(processed_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    features = pd.read_csv(processed_dir / "vendor_month_features.csv")
    events = pd.read_csv(processed_dir / "ncr_event.csv")
    exposure = pd.read_csv(processed_dir / "vendor_month_exposure.csv")

    for frame in [features, events, exposure]:
        frame["vendor_id"] = normalize_id(frame["vendor_id"])
    features["month"] = parse_month(features["month"])
    events["created_month"] = parse_month(events["created_month"])
    exposure["month"] = parse_month(exposure["month"])
    exposure["received_qty_net"] = pd.to_numeric(exposure["received_qty_net"], errors="coerce").fillna(0)

    if features.duplicated(["vendor_id", "month"]).any():
        raise ValueError("vendor_month_features.csv contains duplicate vendor_id/month rows")
    if exposure.duplicated(["vendor_id", "month"]).any():
        raise ValueError("vendor_month_exposure.csv contains duplicate vendor_id/month rows")

    events = add_point_in_time_severe_flags(events)
    valid_events = events[events["vendor_id"].notna() & events["created_month"].notna()].copy()
    monthly_events = valid_events.groupby(["vendor_id", "created_month"], dropna=False).agg(
        future_material_value=("material_value_nonmissing", "sum"),
        future_ncr_count=("notification_number", "nunique"),
        future_severe_ncr_count=("severe_ncr_operational_flag", "sum"),
        future_severe_material_value=(
            "material_value_nonmissing",
            lambda values: values[valid_events.loc[values.index, "severe_ncr_operational_flag"].eq(1)].sum(),
        ),
    ).reset_index().rename(columns={"created_month": "month"})

    monthly_exposure = exposure.groupby(["vendor_id", "month"], dropna=False).agg(
        future_gr_exposure=("received_qty_net", "sum")
    ).reset_index()

    vendors = pd.Index(features["vendor_id"].dropna().unique(), name="vendor_id")
    first_month = features["month"].min()
    last_outcome_month = min(events["created_month"].max(), exposure["month"].max())
    panel_months = pd.period_range(first_month, last_outcome_month, freq="M", name="month")
    panel = pd.MultiIndex.from_product([vendors, panel_months]).to_frame(index=False)
    panel = panel.merge(monthly_events, on=["vendor_id", "month"], how="left")
    panel = panel.merge(monthly_exposure, on=["vendor_id", "month"], how="left")
    fill_columns = [
        "future_material_value",
        "future_ncr_count",
        "future_severe_ncr_count",
        "future_severe_material_value",
        "future_gr_exposure",
    ]
    panel[fill_columns] = panel[fill_columns].fillna(0)
    panel = panel.sort_values(["vendor_id", "month"]).reset_index(drop=True)
    for column in fill_columns:
        panel[f"{column}_{FUTURE_WINDOW_MONTHS}m"] = future_sum_by_vendor_month(panel, column)

    targets = panel[
        [
            "vendor_id",
            "month",
            f"future_material_value_{FUTURE_WINDOW_MONTHS}m",
            f"future_ncr_count_{FUTURE_WINDOW_MONTHS}m",
            f"future_severe_ncr_count_{FUTURE_WINDOW_MONTHS}m",
            f"future_severe_material_value_{FUTURE_WINDOW_MONTHS}m",
            f"future_gr_exposure_{FUTURE_WINDOW_MONTHS}m",
        ]
    ].copy()
    targets["future_gr_exposure_raw_3m"] = targets["future_gr_exposure_3m"]
    targets["future_gr_exposure_3m"] = targets["future_gr_exposure_3m"].clip(lower=0)
    targets["future_loss_rate_3m"] = targets["future_material_value_3m"] / (targets["future_gr_exposure_3m"] + 1)
    targets["future_window_end_month"] = targets["month"].map(lambda month: month_add(month, FUTURE_WINDOW_MONTHS))

    modeling = features.merge(targets, on=["vendor_id", "month"], how="left")
    min_exposure_month = exposure["month"].min()
    modeling["has_complete_future_window"] = modeling["month"].map(lambda month: month + 1 >= min_exposure_month and month + FUTURE_WINDOW_MONTHS <= last_outcome_month)
    modeling = modeling[modeling["has_complete_future_window"]].copy()
    modeling["month"] = modeling["month"].astype("string")
    modeling["future_window_end_month"] = modeling["future_window_end_month"].astype("string")

    stats = {
        "last_outcome_month": str(last_outcome_month),
        "min_exposure_month": str(min_exposure_month),
        "modeling_rows": int(len(modeling)),
        "modeling_vendors": int(modeling["vendor_id"].nunique()),
        "modeling_month_min": modeling["month"].min(),
        "modeling_month_max": modeling["month"].max(),
        "severe_disposition_pattern": DISPOSITION_SEVERE_PATTERN,
        "supplier_severe_disposition_pattern": SUPPLIER_SEVERE_DISPOSITION_PATTERN,
    }
    return modeling, stats


def transform_features(frame: pd.DataFrame, medians: pd.Series | None = None, means: pd.Series | None = None, stds: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    transformed = pd.DataFrame(index=frame.index)
    for column in FEATURE_COLUMNS:
        values = frame[column]
        if values.dtype == bool:
            values = values.astype("float64")
        else:
            values = pd.to_numeric(values, errors="coerce")
        if column in LOG1P_COLUMNS:
            values = np.sign(values) * np.log1p(np.abs(values))
            transformed[f"log1p_{column}"] = values
        else:
            transformed[column] = values.astype("float64")

    if medians is None:
        medians = transformed.median(numeric_only=True).fillna(0)
    transformed = transformed.fillna(medians)

    if means is None:
        means = transformed.mean()
    if stds is None:
        stds = transformed.std(ddof=0).replace(0, 1).fillna(1)
    transformed = (transformed - means) / stds
    return transformed, medians, means, stds


def fit_logistic_regression(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[np.ndarray, list[float]]:
    x = np.column_stack([np.ones(len(x_train)), x_train.to_numpy(dtype="float64")])
    y = y_train.to_numpy(dtype="float64")
    positive_rate = y.mean()
    if positive_rate <= 0 or positive_rate >= 1:
        raise ValueError("Training target must contain both positive and negative examples")
    weights = np.where(y == 1, 0.5 / positive_rate, 0.5 / (1 - positive_rate))
    beta = np.zeros(x.shape[1], dtype="float64")
    loss_history: list[float] = []

    for iteration in range(MODEL_ITERATIONS):
        predictions = sigmoid(x @ beta)
        error = (predictions - y) * weights
        gradient = (x.T @ error) / len(y)
        gradient[1:] += (L2_PENALTY / len(y)) * beta[1:]
        beta -= LEARNING_RATE * gradient
        if iteration % 250 == 0 or iteration == MODEL_ITERATIONS - 1:
            clipped = np.clip(predictions, 1e-8, 1 - 1e-8)
            loss = -np.average(y * np.log(clipped) + (1 - y) * np.log(1 - clipped), weights=weights)
            loss += (L2_PENALTY / (2 * len(y))) * float(np.sum(beta[1:] ** 2))
            loss_history.append(float(loss))
    return beta, loss_history


def predict_scores(x_frame: pd.DataFrame, beta: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(x_frame)), x_frame.to_numpy(dtype="float64")])
    return sigmoid(x @ beta)


def roc_auc(y_true: pd.Series, scores: pd.Series) -> float:
    y = y_true.to_numpy(dtype="int64")
    s = scores.to_numpy(dtype="float64")
    positives = y == 1
    negatives = y == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    rank_sum_pos = ranks[positives].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def add_score_deciles(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    rank = out["risk_score"].rank(method="first", ascending=True)
    out["score_decile"] = pd.qcut(rank, q=10, labels=False, duplicates="drop") + 1
    out["score_decile"] = out["score_decile"].astype("int64")
    return out


def decile_analysis(scored: pd.DataFrame) -> pd.DataFrame:
    total_loss = scored["future_material_value_3m"].sum()
    overall_target_rate = scored["high_future_loss_flag"].mean()
    overall_loss_rate = scored["future_loss_rate_3m"].mean()
    grouped = scored.groupby("score_decile", as_index=False).agg(
        rows=("vendor_id", "size"),
        unique_vendors=("vendor_id", "nunique"),
        min_score=("risk_score", "min"),
        avg_score=("risk_score", "mean"),
        max_score=("risk_score", "max"),
        high_future_loss_rate=("high_future_loss_flag", "mean"),
        avg_future_loss_rate_3m=("future_loss_rate_3m", "mean"),
        future_material_value_3m=("future_material_value_3m", "sum"),
        future_gr_exposure_3m=("future_gr_exposure_3m", "sum"),
        future_ncr_count_3m=("future_ncr_count_3m", "sum"),
        future_severe_ncr_count_3m=("future_severe_ncr_count_3m", "sum"),
        internal_supplier_share=("internal_supplier_candidate", "mean"),
    )
    grouped["decile_lift"] = grouped["high_future_loss_rate"] / overall_target_rate if overall_target_rate else np.nan
    grouped["future_loss_rate_lift"] = grouped["avg_future_loss_rate_3m"] / overall_loss_rate if overall_loss_rate else np.nan
    grouped["future_loss_concentration"] = grouped["future_material_value_3m"] / total_loss if total_loss else np.nan
    grouped["score_band"] = grouped["score_decile"].map(lambda value: f"D{int(value):02d}")
    return grouped.sort_values("score_decile", ascending=False)


def assign_risk_band(rank_pct: pd.Series) -> pd.Series:
    return pd.cut(
        rank_pct,
        bins=[0, 0.05, 0.10, 0.20, 0.50, 1.00],
        labels=["Top 5%", "Next 5%", "Next 10%", "Middle 30%", "Bottom 50%"],
        include_lowest=True,
    )


def risk_band_summary(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["rank_pct"] = out["risk_score"].rank(method="first", ascending=False) / len(out)
    out["risk_band"] = assign_risk_band(out["rank_pct"])
    total_loss = out["future_material_value_3m"].sum()
    summary = out.groupby("risk_band", observed=False, as_index=False).agg(
        rows=("vendor_id", "size"),
        unique_vendors=("vendor_id", "nunique"),
        min_score=("risk_score", "min"),
        avg_score=("risk_score", "mean"),
        max_score=("risk_score", "max"),
        high_future_loss_rate=("high_future_loss_flag", "mean"),
        avg_future_loss_rate_3m=("future_loss_rate_3m", "mean"),
        future_material_value_3m=("future_material_value_3m", "sum"),
        future_gr_exposure_3m=("future_gr_exposure_3m", "sum"),
        future_ncr_count_3m=("future_ncr_count_3m", "sum"),
        future_severe_ncr_count_3m=("future_severe_ncr_count_3m", "sum"),
        internal_supplier_share=("internal_supplier_candidate", "mean"),
    )
    summary["future_loss_concentration"] = summary["future_material_value_3m"] / total_loss if total_loss else np.nan
    summary["selected_rate"] = summary["rows"] / len(out)
    return summary


def split_modeling_data(modeling: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = sorted(modeling["month"].unique())
    if len(months) < 4:
        raise ValueError("Need at least four modeling months for a time-based split")
    holdout_count = max(2, int(np.ceil(len(months) * 0.25)))
    test_months = set(months[-holdout_count:])
    train = modeling[~modeling["month"].isin(test_months)].copy()
    test = modeling[modeling["month"].isin(test_months)].copy()
    return train, test


def choose_training_threshold(train: pd.DataFrame) -> float:
    threshold = float(train["future_loss_rate_3m"].quantile(RISK_THRESHOLD_QUANTILE))
    positive_values = train.loc[train["future_loss_rate_3m"].gt(0), "future_loss_rate_3m"]
    if threshold <= 0 and not positive_values.empty:
        threshold = float(positive_values.quantile(0.50))
    return threshold


def coefficient_table(beta: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    coefficients = pd.DataFrame({"feature": ["intercept", *feature_names], "coefficient": beta})
    coefficients["absolute_coefficient"] = coefficients["coefficient"].abs()
    return coefficients.sort_values("absolute_coefficient", ascending=False)


def write_report(
    path: Path,
    stats: dict[str, object],
    train: pd.DataFrame,
    test: pd.DataFrame,
    threshold: float,
    auc: float,
    deciles: pd.DataFrame,
    bands: pd.DataFrame,
    coefficients: pd.DataFrame,
    loss_history: list[float],
) -> None:
    top_decile = deciles.loc[deciles["score_decile"].eq(deciles["score_decile"].max())].iloc[0]
    top_band = bands.loc[bands["risk_band"].astype("string").eq("Top 5%")].iloc[0]
    coef_preview = coefficients.head(12).copy()
    coef_text = "| Feature | Coefficient |\n|---|---:|\n"
    for row in coef_preview.itertuples(index=False):
        coef_text += f"| `{row.feature}` | {row.coefficient:.4f} |\n"

    content = f"""# Baseline Modeling Summary

Generated by `src/modeling_baseline.py`.

## Scope

This is the baseline modeling stage only. It builds an interpretable logistic
regression ranking model and evaluates future operational loss concentration.
It does not build scorecards, use XGBoost, use deep learning, use LLMs, or
aggressively optimize hyperparameters.

## Point-in-Time Target

- Row key: `vendor_id` + `month`.
- Primary continuous outcome: `future_loss_rate_3m = future_material_value_3m / (future_gr_exposure_3m + 1)`.
- Future window: months `t+1`, `t+2`, and `t+3` only. The current month `t` is excluded.
- Rows are retained only when the full future exposure window falls inside the processed exposure range.
- Future GR exposure is based on net receipt quantity and floored at zero after summing the 3-month window so reversal-heavy windows do not create negative loss-rate denominators.
- Modeling month range: {stats['modeling_month_min']} to {stats['modeling_month_max']}.
- Last outcome month used: {stats['last_outcome_month']}.

## Severe NCR Logic

Severe NCR labels are used for evaluation context, not as current-month
features. A future NCR is severe if any condition holds:

- `material_value` exceeds the prior-month-only rolling P80 material value threshold.
- `disposition` contains `{stats['severe_disposition_pattern']}`.
- supplier-responsible disposition contains `{stats['supplier_severe_disposition_pattern']}`.

## Model

- Model: logistic regression implemented with standardized, mostly log-scaled historical features.
- Binary training label: top future loss-rate observations above the training-only P80 threshold.
- Training threshold: {threshold:,.6f}.
- Train rows: {len(train):,}; train months: {train['month'].min()} to {train['month'].max()}.
- Holdout rows: {len(test):,}; holdout months: {test['month'].min()} to {test['month'].max()}.
- Final weighted log-loss trace: {loss_history[-1]:.4f}.

## Holdout Ranking Results

| Metric | Value |
|---|---:|
| Top decile lift | {top_decile['decile_lift']:.2f} |
| Top decile future loss concentration | {top_decile['future_loss_concentration']:.1%} |
| Top decile average future loss rate | {top_decile['avg_future_loss_rate_3m']:.6f} |
| Top 5% future loss concentration | {top_band['future_loss_concentration']:.1%} |
| Top 5% selected rate | {top_band['selected_rate']:.1%} |
| ROC-AUC secondary diagnostic | {auc:.3f} |

## Largest Standardized Coefficients

{coef_text}
## Outputs

- `data/processed/modeling_dataset_baseline.csv`
- `models/baseline_logistic_model.json`
- `reports/decile_analysis.csv`
- `reports/risk_band_summary.csv`
- `reports/modeling_summary.md`

## Notes

- Existing rolling features are inherited from `src/features.py`, where rolling windows are shifted to completed months before the row month.
- Model scores are ranking scores. They are not presented as calibrated probabilities of default or calibrated NCR probability.
- Operational interpretation should distinguish internal/strategic suppliers from external suppliers before governance action.
"""
    path.write_text(content, encoding="utf-8")


def run(processed_dir: Path, reports_dir: Path, models_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    modeling, stats = build_modeling_dataset(processed_dir)
    train, test = split_modeling_data(modeling)
    threshold = choose_training_threshold(train)
    for frame in [train, test, modeling]:
        frame["high_future_loss_flag"] = frame["future_loss_rate_3m"].gt(threshold).astype("int64")

    x_train, medians, means, stds = transform_features(train)
    x_test, _, _, _ = transform_features(test, medians=medians, means=means, stds=stds)
    beta, loss_history = fit_logistic_regression(x_train, train["high_future_loss_flag"])
    train["risk_score"] = predict_scores(x_train, beta)
    test["risk_score"] = predict_scores(x_test, beta)
    test_scored = add_score_deciles(test)

    deciles = decile_analysis(test_scored)
    bands = risk_band_summary(test_scored)
    auc = roc_auc(test_scored["high_future_loss_flag"], test_scored["risk_score"])
    coefficients = coefficient_table(beta, list(x_train.columns))

    all_x, _, _, _ = transform_features(modeling, medians=medians, means=means, stds=stds)
    modeling["risk_score"] = predict_scores(all_x, beta)
    modeling["score_decile"] = add_score_deciles(modeling)["score_decile"]

    modeling.to_csv(processed_dir / "modeling_dataset_baseline.csv", index=False)
    deciles.to_csv(reports_dir / "decile_analysis.csv", index=False)
    bands.to_csv(reports_dir / "risk_band_summary.csv", index=False)

    artifact = {
        "model_type": "baseline_logistic_regression_numpy",
        "future_window_months": FUTURE_WINDOW_MONTHS,
        "target": "high_future_loss_flag",
        "primary_outcome": "future_loss_rate_3m",
        "risk_threshold_quantile": RISK_THRESHOLD_QUANTILE,
        "training_threshold": threshold,
        "feature_names": list(x_train.columns),
        "coefficients": beta.tolist(),
        "imputation_medians": medians.to_dict(),
        "standardization_means": means.to_dict(),
        "standardization_stds": stds.to_dict(),
        "l2_penalty": L2_PENALTY,
        "learning_rate": LEARNING_RATE,
        "iterations": MODEL_ITERATIONS,
        "train_month_min": train["month"].min(),
        "train_month_max": train["month"].max(),
        "holdout_month_min": test["month"].min(),
        "holdout_month_max": test["month"].max(),
        "holdout_auc_secondary": auc,
        "loss_history": loss_history,
    }
    (models_dir / "baseline_logistic_model.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    write_report(reports_dir / "modeling_summary.md", stats, train, test_scored, threshold, auc, deciles, bands, coefficients, loss_history)

    print("Baseline modeling complete")
    print(f"modeling rows: {len(modeling):,}")
    print(f"holdout ROC-AUC secondary: {auc:.3f}")
    print(f"outputs: {reports_dir / 'modeling_summary.md'}, {reports_dir / 'decile_analysis.csv'}, {reports_dir / 'risk_band_summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build baseline supplier risk ranking model.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    run(args.processed_dir, args.reports_dir, args.models_dir)


if __name__ == "__main__":
    main()
