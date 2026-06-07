# Lab 9: Random Forest Classification
**Loan Repayment Prediction Using LendingClub Data**
 
[![Problem](https://img.shields.io/badge/Problem-Random%20Forest%20Classification-blue)](#)
[![Samples](https://img.shields.io/badge/Samples-9578-green)](#)
[![Features](https://img.shields.io/badge/Features-13-orange)](#)
 
---
 
## Problem Statement
 
The goal of this lab is to apply **Decision Tree** and **Random Forest** classifiers to predict whether a borrower will fully repay their loan, using publicly available data from [LendingClub.com](https://www.lendingclub.com/). The dataset covers loans issued between 2007–2010, before LendingClub went public.
 
- **Target Variable:** `not.fully.paid` (1 = not fully paid, 0 = fully paid)
- **Problem Type:** Binary Classification
- **Dataset Size:** 9,578 entries
---
 
## Dataset Features
 
| # | Feature | Description |
|:-:|---------|-------------|
| 1 | `credit.policy` | 1 if the customer meets LendingClub's credit underwriting criteria, 0 otherwise |
| 2 | `purpose` | Purpose of the loan (e.g., credit card, debt consolidation, small business) |
| 3 | `int.rate` | Interest rate of the loan as a proportion |
| 4 | `installment` | Monthly installment owed by the borrower if the loan is funded |
| 5 | `log.annual.inc` | Natural log of the borrower's self-reported annual income |
| 6 | `dti` | Debt-to-income ratio of the borrower |
| 7 | `fico` | FICO credit score of the borrower |
| 8 | `days.with.cr.line` | Number of days the borrower has had a credit line |
| 9 | `revol.bal` | Borrower's revolving balance |
| 10 | `revol.util` | Borrower's revolving line utilization rate |
| 11 | `inq.last.6mths` | Number of creditor inquiries in the last 6 months |
| 12 | `delinq.2yrs` | Times the borrower was 30+ days past due in the past 2 years |
| 13 | `pub.rec` | Number of derogatory public records |
| 14 | `not.fully.paid` | **Target Variable** |
 
---
 
## Steps Performed
 
1. **Load the Data** — Read `loan_data.csv` into a DataFrame and inspected it using `info()`, `head()`, and `describe()`.
2. **Exploratory Data Analysis (EDA)** — Visualized FICO score distributions by `credit.policy` and `not.fully.paid`; created countplots by loan purpose; used jointplot and lmplot to explore the FICO vs. interest rate trend.
3. **Handle Categorical Features** — Converted the `purpose` column into dummy variables using `pd.get_dummies()` with `drop_first=True` to avoid multicollinearity.
4. **Prepare for Modeling** — 70/30 train/test split with `random_state=101`.
5. **Train Decision Tree** — Fitted a baseline `DecisionTreeClassifier` with default parameters.
6. **Evaluate Decision Tree** — Assessed using confusion matrix and classification report.
7. **Train Random Forest** — Fitted a `RandomForestClassifier` with `n_estimators=200` and `random_state=101`.
8. **Evaluate Random Forest** — Re-evaluated using confusion matrix and classification report; compared performance against the Decision Tree.
---
 
## Key Findings
 
| Finding | Detail |
|---------|--------|
| **Overall Accuracy** | Random Forest outperformed Decision Tree in overall accuracy (85% vs. 72%). |
| **Class Imbalance Trade-off** | Decision Tree had better recall for the minority class (`not.fully.paid=1`) at 23% vs. only 2% for Random Forest. |
| **Business Relevance** | Catching bad loans (high recall on class 1) may be more valuable in practice — the Decision Tree does better here despite lower overall accuracy. |
| **Categorical Encoding** | Dummy variable encoding of `purpose` was required before modeling. |
| **Recommended Improvements** | Techniques like SMOTE, class weighting, or threshold adjustment could improve minority class detection for both models. |
 
---
 
## Libraries Used
 
- **Pandas:** For data loading, manipulation, and dummy variable encoding.
- **NumPy:** For numerical operations.
- **Matplotlib:** For histogram and distribution plots.
- **Seaborn:** For countplots, jointplots, and lmplots during EDA.
- **Scikit-learn:** For train/test splitting, Decision Tree, Random Forest, and evaluation metrics.
