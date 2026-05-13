# Task: Data Preprocessing

## Goal

Build the clean data foundation for the NCR supplier risk project.

This task should only create clean preprocessing outputs. Do not train models. Do not define final modeling target. Do not build scorecard.

## Inputs

Use the available raw data files:

- NCR All
- NCR Open & Closed
- GR History

## Required Outputs

Create the following processed datasets:

1. ncr_event.csv
- one row per NCR event
- key: Notification Number
- include vendor_id, supplier_name, material, created_date, completion_date, qty, material_value, disposition/severity, cause_responsibility

2. vendor_month_exposure.csv
- one row per vendor_id + month
- use GR movement type logic:
  - 101 = positive receipt
  - 102 = negative reversal
- include received_qty_101, received_qty_102, received_qty_net

3. vendor_month_base.csv
- complete vendor-month base table
- merge vendor exposure and NCR monthly event summaries
- do not create future target yet

## Rules

- Read AGENTS.md first.
- Do not modify raw data files.
- Do not train models.
- Do not optimize AUC.
- Do not use supplier name as final key if vendor_id is available.
- Use Notification Number to merge NCR All and NCR Open & Closed.
- Validate mapping coverage between Notification Number and vendor_id.
- Create data quality summary.

## Deliverables

- preprocessing notebook or script
- processed CSV files
- short markdown summary:
  - row counts
  - unique vendor count
  - date ranges
  - merge coverage
  - missing value issues
  - known limitations