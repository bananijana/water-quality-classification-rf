# water-quality-classification-rf

A Random Forest machine learning pipeline for classifying groundwater samples into hydrochemical water types based on field-measured water quality parameters. Built for hydrogeological applications with small to medium sample sizes.

---

## Overview

Groundwater quality assessment often requires classifying samples into distinct hydrochemical facies to understand aquifer behaviour, contamination pathways, and suitability for drinking or irrigation use. This project automates that classification using a Random Forest model trained on 14 water quality parameters.

Developed as part of hydrogeological field research on shallow alluvial aquifers in the Gangetic plain, West Bengal.

---

## Why Random Forest for Water Quality?

- Handles multicollinearity between parameters (e.g. EC–TDS, Ca–TH)
- Robust to small training sets with class balancing
- Feature importance reveals which parameters drive classification
- No assumption of normality (important for skewed geochemical data)

---

## Target: Water Type Classification

Four hydrochemical classes following the dominant ion approach:

| Water Type | Hydrochemical Signature | Typical Setting |
|---|---|---|
| Ca-HCO₃ | Fresh recharge water | Shallow unconfined aquifer near recharge zone |
| Ca-Cl | Transitional | Mixed recharge–discharge |
| Na-HCO₃ | Ion-exchanged water | Base exchange along flow path |
| Na-Cl | Saline / deep water | Confined or long-residence aquifer |

---

## Model Configuration

| Parameter | Value |
|---|---|
| Estimators | 200 trees |
| Max features | sqrt(n_features) |
| Class weight | Balanced |
| Split | 75/25 stratified |
| Validation | 5-Fold Stratified CV |

---

## Project Structure

```
water-quality-classification-rf/
│
├── dataset.csv          # 80 groundwater samples, 15 columns
├── ml_classify.py       # Full ML pipeline
├── ml_outputs/
│   ├── classification_report.csv
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── roc_curves.png
│   └── cv_scores.png
└── README.md
```

---

## Input Parameters

| Group | Parameters |
|---|---|
| Major ions | Ca, Mg, Na, K, HCO₃, Cl, SO₄, NO₃ |
| Physical | pH, EC, TDS, TH |
| Trace | Fe, Mn |

---

## Requirements & Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Run

```bash
python ml_classify.py
```

---

## Output Description

**Classification Report (CSV)** — Precision, recall, F1-score per class. Identifies which water types are harder to distinguish.

**Confusion Matrix** — Visual summary of correct vs misclassified samples. Na-Cl and Na-HCO₃ overlap is expected due to shared cation.

**Feature Importance** — Identifies which water quality parameters most strongly drive facies separation. EC, TDS, Na and Cl typically rank highest in alluvial aquifer systems.

**ROC Curves** — Per-class AUC values indicate model reliability. AUC > 0.75 considered acceptable for 4-class geochemical classification with small datasets.

**CV Score Bar** — Fold-wise accuracy with mean line. Consistent scores across folds indicate stable model generalisation.

---

## Research Context

This classifier was developed alongside field hydrogeochemical sampling in the Hooghly–Rupnarayan basin. The same pipeline is adaptable to CGWB monitoring datasets or any CSV-format water quality table with major ion data.

---

## Author

**Banani Jana**  
ORCID: https://orcid.org/0009-0007-0146-4535

---

## Citation
If you use this methodology or implementation logic in academic or technical work,
please cite this repository.


DOI: https://doi.org/10.5281/zenodo.19564434

---

## License

MIT License
