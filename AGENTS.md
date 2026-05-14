# NCR Supplier Risk Decision Framework

## 1. Project Mission

This project builds a supplier risk ranking and segmentation framework using NCR and GR operational data.

This is not a naive binary "will NCR happen" prediction project.

All suppliers generate NCR activity. The goal is to identify vendors with disproportionate future loss risk and support targeted supplier governance decisions.

## 2. Non-Negotiable Framing

The model should support operational decision-making, not automatic supplier rejection.

Risk scores should be used for ranking, segmentation, monitoring, and prioritization.

Some suppliers are internal or strategically non-substitutable and therefore cannot be governed using the same operational actions as standard external suppliers.

For example, suppliers associated with the Changchun internal manufacturing system may generate significant NCR activity but are operationally difficult or impossible to replace through standard procurement processes.

These suppliers should NOT simply be excluded from the analysis.

Instead, the framework should distinguish between:
- external suppliers
- internal/strategic suppliers

These groups may require different governance pathways, monitoring approaches, thresholds, and operational actions.

## 3. Core Data Keys

Use vendor_id as the primary supplier key whenever available.

Do not use supplier name as the final modeling key.

Use Notification Number as the NCR event key.

## 4. Exposure Logic

GR movement type logic:

- 101 = positive goods receipt
- 102 = reversal / negative receipt

Vendor-month exposure should be based on net receipt quantity.

## 5. Modeling Guardrails

Do not optimize only for AUC or accuracy.

Focus on ranking quality, decile lift, selected rate, average loss rate, and loss concentration.

Do not treat model probability as a calibrated PD unless calibration is explicitly validated.

Never use future information as current-period features.