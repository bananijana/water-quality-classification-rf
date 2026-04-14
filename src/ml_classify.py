"""
ML Classification Script – Random Forest
------------------------------------------
Predicts water type / lithology class from
hydrogeochemical input features.

Outputs:
  - Confusion matrix
  - Feature importance plot
  - Classification report (CSV)
  - ROC curves (One-vs-Rest)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import warnings
import os

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH  = "dataset.csv"
OUTPUT_DIR = "ml_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES   = ["Ca", "Mg", "Na", "K", "HCO3", "Cl",
              "SO4", "NO3", "pH", "EC", "TDS", "TH", "Fe", "Mn"]
TARGET     = "WaterType"
PALETTE    = {"Ca-HCO3": "#2196F3", "Ca-Cl": "#4CAF50",
              "Na-HCO3": "#FF9800", "Na-Cl": "#E91E63"}

RANDOM_STATE = 42
TEST_SIZE    = 0.25

# ── Load & Encode ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape[0]} samples")
print(f"Class distribution:\n{df[TARGET].value_counts()}\n")

X = df[FEATURES].values
le = LabelEncoder()
y = le.fit_transform(df[TARGET])
classes = le.classes_
print(f"Classes: {list(classes)}")

# ── Train / Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}\n")

# ── Model ─────────────────────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)

# ── Cross-Validation ──────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Classification Report ─────────────────────────────────────────────────────
report = classification_report(y_test, y_pred,
                                target_names=classes, output_dict=True)
report_df = pd.DataFrame(report).T.round(3)
report_df.to_csv(f"{OUTPUT_DIR}/classification_report.csv")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=classes))

# ── 1. Confusion Matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=classes)
disp.plot(ax=ax, cmap="Blues", colorbar=False,
          text_kw={"fontsize": 12, "fontweight": "bold"})
ax.set_title("Confusion Matrix – Random Forest Classifier",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("Predicted Label", fontsize=10)
ax.set_ylabel("True Label", fontsize=10)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=180)
plt.close()
print("Saved: confusion_matrix.png")

# ── 2. Feature Importance ─────────────────────────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#1565C0" if v >= importances.median() else "#90CAF9"
          for v in importances.values]
bars = ax.barh(importances.index, importances.values,
               color=colors, edgecolor="white", height=0.65)

# value labels
for bar, val in zip(bars, importances.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=8, color="#333")

ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
ax.set_title("Random Forest – Feature Importance", fontsize=13,
             fontweight="bold")
ax.axvline(importances.median(), color="red", linestyle="--",
           linewidth=1.2, alpha=0.7, label="Median")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=180)
plt.close()
print("Saved: feature_importance.png")

# ── 3. ROC Curves (One-vs-Rest) ───────────────────────────────────────────────
y_test_bin = label_binarize(y_test, classes=range(len(classes)))
class_colors = list(PALETTE.values())

fig, ax = plt.subplots(figsize=(8, 6))
for i, (cls, col) in enumerate(zip(classes, class_colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=col, linewidth=2,
            label=f"{cls}  (AUC = {roc_auc:.2f})")

ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves – One-vs-Rest (Random Forest)", fontsize=12,
             fontweight="bold")
ax.legend(fontsize=9, loc="lower right", framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/roc_curves.png", dpi=180)
plt.close()
print("Saved: roc_curves.png")

# ── 4. CV Score Bar ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
bar_cols = ["#1565C0" if s >= cv_scores.mean() else "#90CAF9"
            for s in cv_scores]
ax.bar(folds, cv_scores, color=bar_cols, edgecolor="white",
       width=0.5)
ax.axhline(cv_scores.mean(), color="red", linestyle="--",
           linewidth=1.5, label=f"Mean = {cv_scores.mean():.4f}")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("5-Fold Cross-Validation Accuracy", fontsize=12,
             fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for i, (bar, val) in enumerate(zip(ax.patches, cv_scores)):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
            f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/cv_scores.png", dpi=180)
plt.close()
print("Saved: cv_scores.png")

print(f"\n✓ All ML outputs saved to: {OUTPUT_DIR}")
print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
