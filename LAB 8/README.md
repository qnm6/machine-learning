# Lab 8: K-Nearest Neighbors (KNN)
**Binary Classification Using the KNN Project Dataset**
 
[![Problem](https://img.shields.io/badge/Problem-KNN%20Classification-blue)](#)
[![Samples](https://img.shields.io/badge/Samples-1000-green)](#)
[![Features](https://img.shields.io/badge/Features-10-orange)](#)
 
---
 
## Problem Statement
 
The goal of this lab is to apply the **K-Nearest Neighbors** algorithm to classify data points into one of two target classes. The dataset is artificial, so the focus is entirely on the machine learning workflow — including feature scaling, model training, and optimizing K using the elbow method.
 
- **Target Variable:** `TARGET CLASS` (0 or 1)
- **Problem Type:** Binary Classification
- **Dataset Size:** 1000 entries
---
 
## Dataset Features
 
The dataset contains 10 anonymized numerical features:
 
| # | Feature | Type |
|:-:|---------|:----:|
| 1 | `XVPM` | Numeric |
| 2 | `GWYH` | Numeric |
| 3 | `TRAT` | Numeric |
| 4 | `TLLZ` | Numeric |
| 5 | `IGGA` | Numeric |
| 6 | `HYKR` | Numeric |
| 7 | `EDFS` | Numeric |
| 8 | `GUUB` | Numeric |
| 9 | `MGJM` | Numeric |
| 10 | `JHZC` | Numeric |
| 11 | `TARGET CLASS` | **Target Variable** |
 
---
 
## Steps Performed
 
1. **Load the Data** — Read `KNN_Project_Data` CSV into a DataFrame.
2. **Exploratory Data Analysis (EDA)** — Generated a pairplot with hue set to `TARGET CLASS` to visualize class separation.
3. **Standardize the Features** — Applied `StandardScaler` to normalize all features before modeling (required for KNN).
4. **Prepare for Modeling** — 70/30 train/test split with `random_state=101`.
5. **Train Initial Model** — Fitted a `KNeighborsClassifier` with K=1 as a baseline.
6. **Evaluate Initial Model** — Assessed using confusion matrix and classification report.
7. **Elbow Method** — Iterated K from 1 to 39, plotting error rate vs. K to find the optimal value.
8. **Retrain with Optimal K** — Retrained with **K=31** based on the elbow plot; re-evaluated performance.
---
 
## Key Findings
 
| Finding | Detail |
|---------|--------|
| **Feature Scaling** | Standardization was essential — KNN is distance-based and sensitive to feature scale. |
| **Optimal K** | K=31 was selected via the elbow method, giving an error rate of ~0.16 with no overfitting. |
| **Model Improvement** | Retraining with K=31 improved performance significantly over the K=1 baseline. |
| **Evaluation Metrics** | Confusion matrix and classification report (precision, recall, F1-score) used for both models. |
 
---
 
## Libraries Used
 
- **Pandas:** For data loading and manipulation.
- **NumPy:** For numerical operations and error rate calculation.
- **Matplotlib:** For plotting the elbow curve.
- **Seaborn:** For the EDA pairplot.
- **Scikit-learn:** For scaling, model training, train/test splitting, and evaluation metrics.
