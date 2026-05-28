# Supplier Risk Decision Framework

## Business Problem

Most suppliers generate NCR activity over time, making binary NCR occurrence prediction operationally unhelpful.

This project instead focuses on identifying suppliers that disproportionately contribute to future operational loss exposure.

The framework ranks supplier-months using historical NCR behavior, goods receipt exposure, and operational severity indicators to support supplier risk prioritization and portfolio-level monitoring.

The objective is not automatic supplier rejection, but earlier identification of concentrated operational risk.

## Why This Matters

Operational review resources are limited.

A useful risk framework should help procurement, operations, and quality teams identify which suppliers require closer monitoring, additional investigation, or earlier intervention.

The project therefore focuses on:

- risk concentration
- operational loss exposure
- supplier prioritization
- portfolio-level monitoring
- point-in-time risk evaluation

rather than generic defect counting.

## Methodology

1. Construct event-level NCR observations from task-level operational records.
2. Build vendor-month operational exposure using goods receipt activity.
3. Engineer historical rolling features using only prior completed months.
4. Define future 3-month operational loss-rate targets.
5. Train an interpretable baseline ranking model.
6. Evaluate concentration performance using decile-based future loss capture.

The framework explicitly avoids future leakage and uses point-in-time feature construction principles.

## Key Results

- Holdout ROC-AUC: approximately 0.799
- Top decile captured approximately 88.5% of future material loss exposure
- Top 5% captured approximately 70.1% of future material loss exposure

Results suggest that future operational loss exposure is highly concentrated within a relatively small subset of suppliers.

## Risk Interpretation

This framework behaves more similarly to:

- operational risk monitoring
- merchant risk management
- portfolio-level risk prioritization
- concentration risk analysis

than traditional manufacturing reporting.

The project focuses on identifying asymmetric operational risk exposure rather than predicting generic NCR occurrence.
