# Lab 7: Logistic Regression
**Predicting Ad Clicks Using the Advertising Dataset**

[![Problem](https://img.shields.io/badge/Problem-Logistic%20Regression-blue)](#)
[![Samples](https://img.shields.io/badge/Samples-1000-green)](#)
[![Features](https://img.shields.io/badge/Features-5-orange)](#)

---

## Problem Statement

The goal of this lab is to apply **Logistic Regression** to predict whether an internet user will click on an advertisement based on their browsing behavior and demographics.

- **Target Variable:** `Clicked on Ad` (0 = Did Not Click, 1 = Clicked)
- **Problem Type:** Binary Classification
- **Dataset Size:** 1000 entries

---

## Dataset Features

| # | Feature | Description | Type |
|:-:|---------|-------------|:----:|
| 1 | `Daily Time Spent on Site` | Time consumer spent on site (minutes) | Numeric |
| 2 | `Age` | Age of the customer (years) | Numeric |
| 3 | `Area Income` | Average income of the consumer's geographical area | Numeric |
| 4 | `Daily Internet Usage` | Average minutes per day the consumer is online | Numeric |
| 5 | `Male` | Whether the consumer is male (1 = Yes, 0 = No) | Categorical |
| 6 | `Clicked on Ad` | **Target Variable** — whether the ad was clicked | Categorical |

---

## Steps Performed

1. **Load the Data** — Read `advertising.csv` into a DataFrame.
2. **Explore the Data** — Used `head()`, `info()`, and `describe()` for an initial overview.
3. **Exploratory Data Analysis (EDA)** — Visualized the data using:
   - Histogram of Age distribution.
   - Jointplots for Area Income vs. Age, Daily Time on Site vs. Age (KDE), and Daily Time on Site vs. Daily Internet Usage.
   - Pairplot with hue set to `Clicked on Ad` to identify class separation.
4. **Prepare for Modeling** — Selected 5 features; 67/33 train/test split with `random_state=42`.
5. **Train the Model** — Fitted a `LogisticRegression` model on the training set.
6. **Evaluate the Model** — Generated a classification report with precision, recall, and F1-score.

---

## Key Findings

| Finding | Detail |
|---------|--------|
| **Model Type** | Logistic Regression — appropriate for binary classification. |
| **EDA Insight** | Pairplot revealed clear class separation, suggesting the features are strong predictors. |
| **Evaluation** | Model performance assessed using a full classification report (precision, recall, F1-score). |
| **Data Quality** | Clean dataset requiring no missing value handling or encoding (except the binary `Male` column). |

---

## Libraries Used

- **Pandas:** For data loading and manipulation.
- **NumPy:** For numerical operations.
- **Matplotlib:** For basic plotting.
- **Seaborn:** For statistical visualizations (histograms, jointplots, pairplot).
- **Scikit-learn:** For model training, train/test splitting, and classification report.
