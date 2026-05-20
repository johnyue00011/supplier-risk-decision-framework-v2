# Task: Baseline Modeling

## Goal

Build a baseline supplier risk ranking model using historical vendor-month features.

The objective is NOT naive binary NCR prediction.

The objective is to rank supplier-month observations by disproportionate future operational risk exposure.

## Inputs

Use:

- data/processed/vendor_month_features.csv
- data/processed/ncr_event.csv
- data/processed/vendor_month_exposure.csv

## Modeling Philosophy

This is a ranking and portfolio-segmentation problem.

Do not optimize only for AUC.

Focus on:
- decile lift
- loss concentration
- operational prioritization
- portfolio monitoring

Do not treat model probabilities as calibrated PD unless calibration is explicitly validated.

## Target Definition

Primary target:

future_loss_rate_3m

Definition:

future_material_value_3m / (future_gr_exposure_3m + 1)

Where:
- future windows must use strictly future months only
- current month information cannot enter future target construction

## Severe NCR Logic

Define severe NCR using operational materiality.

Severe NCR if ANY condition holds:
- material_value > rolling P80 threshold
- disposition contains:
  - scrap
  - RTV
  - MRB
- supplier-responsible severe disposition

## Required Outputs

Create:

- modeling dataset
- baseline logistic regression model
- decile analysis
- concentration analysis
- risk segmentation summary
- markdown modeling report

## Required Evaluations

1. Decile lift table
2. Future loss concentration by score band
3. Average future loss rate by decile
4. Top-risk supplier concentration
5. Basic ROC-AUC (secondary only)

## Hard Rules

- no future leakage
- all rolling features must be shifted
- future target must use strictly future months
- preserve point-in-time integrity
- no deep learning
- no LLM
- no XGBoost initially
- use interpretable baseline first

## Deliverables

- src/modeling_baseline.py
- reports/modeling_summary.md
- reports/decile_analysis.csv
- reports/risk_band_summary.csv
- model artifact