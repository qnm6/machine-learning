# Lab 10: Support Vector Machines
**Iris Flower Species Classification**
 
[![Problem](https://img.shields.io/badge/Problem-Support%20Vector%20Machine%20Classification-blue)](#)
[![Samples](https://img.shields.io/badge/Samples-150-green)](#)
[![Features](https://img.shields.io/badge/Features-4-orange)](#)
 
---
 
## Problem Statement
 
The goal of this lab is to implement a **Support Vector Machine (SVM)** classifier to predict the species of an iris flower based on physical measurements. The Iris dataset is a classic multi-class classification benchmark consisting of 150 samples across three species.
 
- **Target Variable:** `species` (setosa, versicolor, virginica)
- **Problem Type:** Multi-class Classification
- **Dataset Size:** 150 entries
---
 
## Dataset Features
 
| # | Feature | Description |
|:-:|---------|-------------|
| 1 | `sepal_length` | Length of the sepal in centimeters |
| 2 | `sepal_width` | Width of the sepal in centimeters |
| 3 | `petal_length` | Length of the petal in centimeters |
| 4 | `petal_width` | Width of the petal in centimeters |
| 5 | `species` | **Target Variable** (setosa, versicolor, virginica) |
 
---
 
## Steps Performed
 
1. **Load the Data** — Used `seaborn` to load the built-in Iris dataset into a DataFrame and inspected the initial rows.
2. **Exploratory Data Analysis (EDA)** — Created a `pairplot` to identify class separability and a `kdeplot` of sepal dimensions specifically for the Setosa species.
3. **Prepare for Modeling** — Conducted a 70/30 train/test split using `train_test_split` with a fixed `random_state=101`.
4. **Train SVM Model** — Initialized a baseline `SVC()` classifier and fitted it to the training data.
5. **Evaluate Model** — Generated predictions and analyzed performance using a `classification_report` and a `confusion_matrix`.
6. **GridSearch Optimization** — Defined a parameter grid for `C` and `gamma` and used `GridSearchCV` to find the most effective hyperparameters.
7. **Final Evaluation** — Re-evaluated the tuned model on the test set to compare performance against the baseline model.
---
 
## Key Findings
 
| Finding | Detail |
|---------|--------|
| **Class Separability** | Exploratory analysis confirms that the Setosa species is linearly separable and highly distinct from the other two species. |
| **Model Performance** | The SVM classifier achieved near-perfect accuracy (approx. 98–100%) due to the high quality and distinct features of the dataset. |
| **Parameter Tuning** | `GridSearchCV` was effective in identifying the optimal `C` and `gamma` values, though the baseline model was already highly accurate. |
| **Boundary Visualization** | Heatmap visualizations of the confusion matrix illustrate the model's high precision in distinguishing between Versicolor and Virginica. |
| **Workflow Robustness** | Even for small datasets, following a GridSearch workflow ensures the model is not over-fitting and generalizes well to new data. |
 
---
 
## Libraries Used
 
- **Pandas:** For data handling and basic manipulation.
- **NumPy:** For numerical computation.
- **Matplotlib:** For static plot generation and figure formatting.
- **Seaborn:** For advanced statistical visualizations and loading the dataset.
- **Scikit-learn:** For the SVM algorithm, hyperparameter tuning (`GridSearchCV`), and evaluation metrics.
