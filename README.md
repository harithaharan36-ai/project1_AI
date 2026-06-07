# ML Classification Project — Breast Cancer Detection

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 📋 Project Overview

A supervised binary classification project to predict whether a breast tumour is **malignant** or **benign** using the **Wisconsin Breast Cancer Dataset**.

**Models compared:**
- Logistic Regression
- Random Forest Classifier

**Evaluation metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## 📁 Repository Structure

```
ml-classification-project/
├── classification_notebook.ipynb   # Full Jupyter Notebook with code, plots & metrics
├── reports/
│   └── model_selection_report.md   # Short report summarising model selection & results
├── README.md                       # This file — environment setup & instructions
└── requirements.txt                # Exact package versions for reproducibility
```

---

## 🚀 Environment Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/ml-classification-project.git
cd ml-classification-project
```

### 2. Create a virtual environment (recommended)

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate       # Linux / macOS
# or
venv\Scripts\activate          # Windows
```

**Using conda:**
```bash
conda create -n ml-classification python=3.10
conda activate ml-classification
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ▶️ Running the Notebook

### Option A: Jupyter Notebook (local)
```bash
# Ensure your virtual environment is activated
jupyter notebook classification_notebook.ipynb
```

### Option B: Jupyter Lab (local)
```bash
jupyter lab classification_notebook.ipynb
```

### Option C: VS Code
Open the project folder and click **Run All** on the notebook file.

### Option D: Google Colab (no local setup)
1. Upload `classification_notebook.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Run `!pip install -r requirements.txt` in the first cell
3. Execute all cells

---

## 📊 Expected Output

After running the full notebook you will see:

| Metric              | Logistic Regression | Random Forest |
|---------------------|---------------------|---------------|
| **Accuracy**        | ~0.9737             | ~0.9649       |
| **Precision**       | ~0.9722             | ~0.9589       |
| **Recall**          | ~0.9859             | ~0.9859       |
| **F1-Score**        | ~0.9790             | ~0.9722       |
| **ROC-AUC**         | ~0.9980             | ~0.9984       |

Visual outputs include:
- 📊 Class distribution bar & pie charts
- 🔥 Feature correlation heatmap
- 📦 Cross-validation performance box plots
- 🔢 Confusion matrices
- 📈 ROC Curves
- 🏆 Feature importance chart (Random Forest)
- 📊 Final metrics comparison bar chart

---

## 📦 Package Versions (from `requirements.txt`)

```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
jupyter==1.0.0
notebook==7.0.2
```

> **Note:** The dataset (`load_breast_cancer`) is bundled with `scikit-learn` — no external downloads needed.

---

## 🔄 Reproducibility

- **Random seed:** `random_state=42` is used consistently across all operations.
- **No external data dependencies** — the dataset downloads automatically via sklearn.
- **Exact package versions** are pinned in `requirements.txt`.
- The notebook can be run end-to-end without any manual intervention.

---

## 📄 Report

The full model selection report is available at:
📁 [`reports/model_selection_report.md`](reports/model_selection_report.md)

---

## 📬 Submission Links

| Item | Link |
|------|------|
| GitHub Repository | [https://github.com/<your-username>/ml-classification-project](https://github.com/<your-username>/ml-classification-project) |
| Task Submission | [https://internspark.in/submission.html](https://internspark.in/submission.html) |

---

## 👨‍💻 Author

Arena.ai Agent — ML Classification Project

---

## 📝 License

This project is for educational / portfolio purposes under the MIT License.
