#!/usr/bin/env python3
"""
ML Classification Project - Supervised Classification Model
Titanic Survival Prediction using Logistic Regression & Random Forest

Author: Arena.ai Agent
Date: 2026-06-07
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, learning_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────
# 0. Setup output directories
# ──────────────────────────────────────────────────────
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('notebook', exist_ok=True)

np.random.seed(42)

print("=" * 70)
print("ML CLASSIFICATION PROJECT - TITANIC SURVIVAL PREDICTION")
print("=" * 70)

# ──────────────────────────────────────────────────────
# 1. Load & Explore Data
# ──────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("1. DATA LOADING & EXPLORATION")
print("─" * 70)

# Load Titanic dataset from seaborn
df = sns.load_dataset('titanic')
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nData info:")
print(df.info())
print(f"\nMissing values per column:")
print(df.isnull().sum())
print(f"\nTarget distribution (survived):")
print(df['survived'].value_counts())
print(f"  (% survived: {df['survived'].mean()*100:.2f}%)")

# ──────────────────────────────────────────────────────
# 2. Data Preprocessing
# ──────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("2. DATA PREPROCESSING")
print("─" * 70)

# Make a copy
data = df.copy()

# Drop columns that leak information or are non-predictive
# 'alive' is a string version of 'survived' -> target leakage
# 'who', 'adult_male', 'embark_town' are duplicates/alternate forms
# 'deck' has too many missing values
# 'class' is the same as 'pclass' (categorical)
cols_to_drop = ['alive', 'who', 'adult_male', 'deck', 'embark_town', 'class']
data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])
print(f"Columns after dropping redundant ones: {list(data.columns)}")

# Handle missing values
# Age: median by Sex and Pclass
data['age'] = data.groupby(['sex', 'pclass'])['age'].transform(
    lambda x: x.fillna(x.median())
)
# Embarked: fill with mode
data['embarked'] = data['embarked'].fillna(data['embarked'].mode()[0])

print(f"\nMissing values after imputation:")
print(data.isnull().sum())

# Encode categorical variables
label_encoders = {}
categorical_cols = ['sex', 'embarked']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
    print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print(f"\nProcessed data (first 5 rows):")
print(data.head())

# Feature selection
feature_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
X = data[feature_cols].copy()
y = data['survived'].copy()

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Features: {feature_cols}")

# ──────────────────────────────────────────────────────
# 3. Train/Test Split
# ──────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("3. TRAIN/TEST SPLIT (80/20 stratified)")
print("─" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set:   {X_train.shape[0]} samples "
      f"(survived: {y_train.mean()*100:.1f}%)")
print(f"Test set:       {X_test.shape[0]} samples "
      f"(survived: {y_test.mean()*100:.1f}%)")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling applied (StandardScaler).")

# ──────────────────────────────────────────────────────
# 4. Cross-Validation Setup
# ──────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("4. CROSS-VALIDATION (Stratified 5-Fold)")
print("─" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ──────────────────────────────────────────────────────
# 5. Model 1: Logistic Regression
# ──────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("5. MODEL 1: LOGISTIC REGRESSION")
print("─" * 70)

lr = LogisticRegression(max_iter=1000, random_state=42)

# Cross-validation
lr_cv_scores = cross_val_score(lr, X_train_scaled, y_train, cv=cv, scoring='accuracy')
print(f"CV Accuracy scores: {lr_cv_scores}")
print(f"CV Accuracy mean:   {lr_cv_scores.mean():.4f} (+/- {lr_cv_scores.std()*2:.4f})")

lr_cv_roc = cross_val_score(lr, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
print(f"CV ROC-AUC scores:  {lr_cv_roc}")
print(f"CV ROC-AUC mean:    {lr_cv_roc.mean():.4f} (+/- {lr_cv_roc.std()*2:.4f})")

# Train on full training set
lr.fit(X_train_scaled, y_train)
lr_train_pred = lr.predict(X_train_scaled)
lr_test_pred = lr.predict(X_test_scaled)
lr_test_proba = lr.predict_proba(X_test_scaled)[:, 1]

# ──────────────────────────────────────────────────────
# 6. Model 2: Random Forest
# ──────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("6. MODEL 2: RANDOM FOREST")
print("─" * 70)

rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

# Cross-validation
rf_cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=cv, scoring='accuracy')
print(f"CV Accuracy scores: {rf_cv_scores}")
print(f"CV Accuracy mean:   {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std()*2:.4f})")

rf_cv_roc = cross_val_score(rf, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
print(f"CV ROC-AUC scores:  {rf_cv_roc}")
print(f"CV ROC-AUC mean:    {rf_cv_roc.mean():.4f} (+/- {rf_cv_roc.std()*2:.4f})")

# Train on full training set
rf.fit(X_train_scaled, y_train)
rf_train_pred = rf.predict(X_train_scaled)
rf_test_pred = rf.predict(X_test_scaled)
rf_test_proba = rf.predict_proba(X_test_scaled)[:, 1]

# ──────────────────────────────────────────────────────
# 7. Model Evaluation & Metrics
# ──────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("7. MODEL EVALUATION - TEST SET METRICS")
print("═" * 70)

def print_metrics(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_proba)

    print(f"\n  {'─' * 40}")
    print(f"  {name:^38}")
    print(f"  {'─' * 40}")
    print(f"  {'Accuracy :':20s} {acc:.4f}")
    print(f"  {'Precision:':20s} {prec:.4f}")
    print(f"  {'Recall   :':20s} {rec:.4f}")
    print(f"  {'F1-Score :':20s} {f1:.4f}")
    print(f"  {'ROC-AUC  :':20s} {roc:.4f}")
    print(f"  {'─' * 40}")
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc}

print("\n>>> TRAINING SET PERFORMANCE (may overfit for RF):")
lr_train_metrics = print_metrics("Logistic Regression (Train)", y_train, lr_train_pred, lr.predict_proba(X_train_scaled)[:, 1])
rf_train_metrics = print_metrics("Random Forest (Train)", y_train, rf_train_pred, rf.predict_proba(X_train_scaled)[:, 1])

print("\n>>> TEST SET PERFORMANCE (generalization):")
lr_metrics = print_metrics("Logistic Regression (Test)", y_test, lr_test_pred, lr_test_proba)
rf_metrics = print_metrics("Random Forest (Test)", y_test, rf_test_pred, rf_test_proba)

# ──────────────────────────────────────────────────────
# 8. Comparison Table
# ──────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("8. MODEL COMPARISON SUMMARY")
print("═" * 70)

comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'CV Accuracy (mean)', 'CV ROC-AUC (mean)'],
    'Logistic Regression': [
        f"{lr_metrics['accuracy']:.4f}",
        f"{lr_metrics['precision']:.4f}",
        f"{lr_metrics['recall']:.4f}",
        f"{lr_metrics['f1']:.4f}",
        f"{lr_metrics['roc_auc']:.4f}",
        f"{lr_cv_scores.mean():.4f}",
        f"{lr_cv_roc.mean():.4f}"
    ],
    'Random Forest': [
        f"{rf_metrics['accuracy']:.4f}",
        f"{rf_metrics['precision']:.4f}",
        f"{rf_metrics['recall']:.4f}",
        f"{rf_metrics['f1']:.4f}",
        f"{rf_metrics['roc_auc']:.4f}",
        f"{rf_cv_scores.mean():.4f}",
        f"{rf_cv_roc.mean():.4f}"
    ]
})
print(f"\n{comparison.to_string(index=False)}")

# Determine best model
lr_score = lr_metrics['f1'] * 0.3 + lr_metrics['roc_auc'] * 0.7
rf_score = rf_metrics['f1'] * 0.3 + rf_metrics['roc_auc'] * 0.7
best_model_name = "Logistic Regression" if lr_score >= rf_score else "Random Forest"
print(f"\n  ✓ Best model (weighted F1+ROC-AUC): {best_model_name}")

# ──────────────────────────────────────────────────────
# 9. Generate Plots
# ──────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("9. GENERATING VISUALIZATIONS")
print("─" * 70)

# Set up the plotting style
sns.set_style("whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

# --- Figure 1: Confusion Matrices ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, model_name, y_pred in zip(axes, ['Logistic Regression', 'Random Forest'], [lr_test_pred, rf_test_pred]):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Died', 'Survived'])
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    ax.set_title(f'{model_name} - Confusion Matrix', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('notebook/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Confusion matrices saved.")

# --- Figure 2: ROC Curves ---
fig, ax = plt.subplots(figsize=(8, 6))

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_test_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_test_proba)

ax.plot(fpr_lr, tpr_lr, lw=2.5,
        label=f'Logistic Regression (AUC = {lr_metrics["roc_auc"]:.3f})')
ax.plot(fpr_rf, tpr_rf, lw=2.5,
        label=f'Random Forest (AUC = {rf_metrics["roc_auc"]:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
ax.fill_between(fpr_lr, tpr_lr, alpha=0.08, color='#2E86AB')
ax.fill_between(fpr_rf, tpr_rf, alpha=0.08, color='#A23B72')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
plt.tight_layout()
plt.savefig('notebook/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ ROC curves saved.")

# --- Figure 3: Feature Importance (Random Forest) ---
fig, ax = plt.subplots(figsize=(8, 5))
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(feature_cols)))

bars = ax.barh(range(len(indices)), importances[indices], color=colors[::-1], edgecolor='navy', linewidth=0.6)
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([feature_cols[i] for i in indices], fontsize=11)
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('Random Forest - Feature Importance', fontsize=14, fontweight='bold')
ax.invert_yaxis()

# Add value labels
for bar, val in zip(bars, importances[indices]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('notebook/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Feature importance plot saved.")

# --- Figure 4: Logistic Regression Coefficients ---
fig, ax = plt.subplots(figsize=(8, 5))
coefs = lr.coef_[0]
coef_indices = np.argsort(np.abs(coefs))[::-1]
colors_coef = ['#2E86AB' if c > 0 else '#A23B72' for c in coefs[coef_indices]]

bars = ax.barh(range(len(coef_indices)), coefs[coef_indices], color=colors_coef, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(coef_indices)))
ax.set_yticklabels([feature_cols[i] for i in coef_indices], fontsize=11)
ax.set_xlabel('Coefficient Value', fontsize=12)
ax.set_title('Logistic Regression - Feature Coefficients', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8)
ax.invert_yaxis()

for bar, val in zip(bars, coefs[coef_indices]):
    ax.text(val + 0.02 if val > 0 else val - 0.12,
            bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('notebook/lr_coefficients.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ LR coefficients plot saved.")

# --- Figure 5: Metrics Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(10, 6))
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

metric_key_map = {'Accuracy': 'accuracy', 'Precision': 'precision', 'Recall': 'recall',
                  'F1-Score': 'f1', 'ROC-AUC': 'roc_auc'}
lr_vals = [lr_metrics[metric_key_map[m]] for m in metrics_names]
rf_vals = [rf_metrics[metric_key_map[m]] for m in metrics_names]

x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax.bar(x - width/2, lr_vals, width, label='Logistic Regression',
               color='#2E86AB', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, rf_vals, width, label='Random Forest',
               color='#A23B72', edgecolor='black', linewidth=0.5)

ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Test Set Performance - Model Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=11)
ax.set_ylim([0, 1.15])
ax.legend(fontsize=11)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('notebook/metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Metrics comparison chart saved.")

# --- Figure 6: Learning Curves ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (model, name) in zip(axes, [
    (LogisticRegression(max_iter=1000, random_state=42), 'Logistic Regression'),
    (RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42), 'Random Forest')
]):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train_scaled, y_train,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=cv, scoring='accuracy', n_jobs=-1, random_state=42
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    ax.plot(train_sizes, train_mean, 'o-', color='#2E86AB', label='Training score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#2E86AB')
    ax.plot(train_sizes, test_mean, 'o-', color='#A23B72', label='Cross-validation score')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='#A23B72')
    ax.set_xlabel('Training examples')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{name} - Learning Curve', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim([0.5, 1.02])

plt.tight_layout()
plt.savefig('notebook/learning_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Learning curves saved.")

# --- Figure 7: Data Exploration (EDA) ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Survival by class
sns.barplot(x='pclass', y='survived', data=df, ax=axes[0,0], palette='Blues_d', edgecolor='black')
axes[0,0].set_title('Survival Rate by Passenger Class', fontweight='bold')
axes[0,0].set_xlabel('Passenger Class')
axes[0,0].set_ylabel('Survival Rate')

# Survival by sex
sns.barplot(x='sex', y='survived', data=df, ax=axes[0,1], palette='Set2', edgecolor='black')
axes[0,1].set_title('Survival Rate by Sex', fontweight='bold')
axes[0,1].set_xlabel('Sex')
axes[0,1].set_ylabel('Survival Rate')

# Age distribution
axes[0,2].hist([df[df['survived']==0]['age'].dropna(),
                df[df['survived']==1]['age'].dropna()],
               bins=20, label=['Died', 'Survived'],
               color=['#E74C3C', '#2ECC71'], alpha=0.6, edgecolor='black')
axes[0,2].set_title('Age Distribution by Survival', fontweight='bold')
axes[0,2].set_xlabel('Age')
axes[0,2].set_ylabel('Count')
axes[0,2].legend()

# Fare distribution
axes[1,0].hist([df[df['survived']==0]['fare'].dropna(),
                df[df['survived']==1]['fare'].dropna()],
               bins=20, label=['Died', 'Survived'],
               color=['#E74C3C', '#2ECC71'], alpha=0.6, edgecolor='black')
axes[1,0].set_title('Fare Distribution by Survival', fontweight='bold')
axes[1,0].set_xlabel('Fare')
axes[1,0].set_ylabel('Count')
axes[1,0].legend()

# Survival by embarkation
sns.barplot(x='embarked', y='survived', data=df, ax=axes[1,1], palette='muted', edgecolor='black')
axes[1,1].set_title('Survival Rate by Embarkation Port', fontweight='bold')
axes[1,1].set_xlabel('Embarkation Port')
axes[1,1].set_ylabel('Survival Rate')

# SibSp vs survival
sns.barplot(x='sibsp', y='survived', data=df, ax=axes[1,2], palette='viridis', edgecolor='black')
axes[1,2].set_title('Survival Rate by # Siblings/Spouses', fontweight='bold')
axes[1,2].set_xlabel('Number of Siblings/Spouses')
axes[1,2].set_ylabel('Survival Rate')

plt.suptitle('Exploratory Data Analysis - Titanic Dataset', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('notebook/eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ EDA plots saved.")

print("\n" + "=" * 70)
print("PROJECT COMPLETE! All analyses and plots generated.")
print("=" * 70)
print(f"\nGenerated files:")
print(f"  - notebook/confusion_matrices.png")
print(f"  - notebook/roc_curves.png")
print(f"  - notebook/feature_importance.png")
print(f"  - notebook/lr_coefficients.png")
print(f"  - notebook/metrics_comparison.png")
print(f"  - notebook/learning_curves.png")
print(f"  - notebook/eda_plots.png")
