<div align="center">

# Lab 6: Linear Regression
**Predicting Yearly Spending Using the Ecommerce Customers Dataset**

[![Problem](https://img.shields.io/badge/Problem-Linear%20Regression-blue)](#)
[![Samples](https://img.shields.io/badge/Samples-500-green)](#)
[![Features](https://img.shields.io/badge/Features-4-orange)](#)

</div>

---

## Problem Statement

The goal of this lab is to apply **Linear Regression** to predict how much a customer spends yearly based on their e-commerce usage behavior.

After completing a walkthrough using the USA Housing dataset, the same workflow is independently applied to the Ecommerce Customers dataset.

- **Target Variable:** `Yearly Amount Spent` (continuous dollar value)
- **Problem Type:** Regression
- **Dataset Size:** 500 entries

---

## Dataset Features

| # | Feature | Description | Type |
|:-:|---------|-------------|:----:|
| 1 | `Avg. Session Length` | Average duration of in-store style advice sessions (minutes) | Numeric |
| 2 | `Time on App` | Average time spent on the mobile app (minutes) | Numeric |
| 3 | `Time on Website` | Average time spent on the website (minutes) | Numeric |
| 4 | `Length of Membership` | Number of years the customer has been a member | Numeric |
| 5 | `Yearly Amount Spent` | **Target Variable** — total amount spent per year ($) | Numeric |

> Non-predictive string columns (`Email`, `Address`, `Avatar`) were excluded from modeling.

---

## Steps Performed

1. **Load the Data** — Read the Ecommerce Customers CSV into a DataFrame.
2. **Explore the Data** — Used `head()`, `info()`, and `describe()` for an initial overview.
3. **Data Cleaning** — Checked for null values using `isnull().sum()` (None found).
4. **Feature Engineering** — Selected 4 numerical features; dropped non-predictive string columns.
5. **Prepare for Modeling** — 70/30 train/test split using `train_test_split` with `random_state=101`.
6. **Train the Model** — Fitted a `LinearRegression` model on the training set.
7. **Evaluate the Model** — Measured MAE, MSE, and RMSE; plotted a residual histogram to assess fit.

---

## Key Findings

| Finding | Detail |
|---------|--------|
| **Data Quality** | Dataset is clean with **0** missing values, requiring minimal preprocessing. |
| **Key Predictor** | `Length of Membership` is expected to be the strongest predictor of yearly spending. |
| **Model Fit** | Residual histogram showed an approximately normal distribution, indicating a good fit. |
| **Evaluation** | MAE, MSE, and RMSE were used to quantify prediction error on the test set. |

---

## Libraries Used

- **Pandas:** For data loading and manipulation.
- **NumPy:** For numerical operations and RMSE computation.
- **Matplotlib:** For plotting predictions and residuals.
- **Seaborn:** For statistical visualizations (residual histogram).
- **Scikit-learn:** For model training, train/test splitting, and evaluation metrics.
