# Task: Feature Engineering

## Goal

Build historical vendor-month risk features from preprocessing outputs.

This task should only create feature tables. Do not train models. Do not define final score bands. Do not build scorecards.

## Inputs

Use:

- data/processed/ncr_event.csv
- data/processed/vendor_month_exposure.csv
- data/processed/vendor_month_base.csv

## Required Output

Create:

- data/processed/vendor_month_features.csv
- reports/feature_engineering_summary.md

## Feature Rules

Use only historical information available at or before each vendor-month.

Do not create future targets.

Do not create model scores.

Do not use future disposition, future cause responsibility, or future completion outcome as current-period features.

## Required Feature Groups

1. NCR frequency features
- ncr_count_1m
- ncr_count_3m
- ncr_count_6m

2. Severity features
- severe_ncr_count_3m
- severe_ncr_count_6m
- max_severity_6m

3. Exposure-normalized features
- ncr_rate_3m
- severe_rate_3m
- loss_rate_3m

4. Loss proxy features
- material_value_sum_3m
- material_value_sum_6m
- material_value_max_6m

5. Recency features
- days_since_last_ncr
- recent_severe_flag

6. Responsibility features
- supplier_fault_share_6m
- internal_fault_share_6m
- unknown_fault_share_6m

7. Trend features
- ncr_rate_change
- severe_rate_change

## Rules

- Use vendor_id + month as the panel key.
- Preserve internal_supplier_candidate.
- Avoid over-engineering.
- Add missing indicators where appropriate.
- Document all feature definitions.
- Validate that no duplicate vendor_id + month rows exist.
- Do not train models.

## Deliverables

- feature engineering script or notebook
- vendor_month_features.csv
- feature_engineering_summary.md