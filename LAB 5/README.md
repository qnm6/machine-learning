<div align="center">

# Lab 5: Feature Engineering (Classification)
**Order Status Prediction using a Talabat-style Orders Dataset**

</div>

## Student tasks

### Task 1: New Engineered Feature
**Feature Created:** `price_ratio` (`Total_Price / (Delivery_Distance_km + 1)`)
**Justification:** The `price_ratio` feature captures the economic value density of an order. A low-priced order requiring a very long delivery distance might be more prone to driver cancellation or delays ("In Transit" status) because the incentive is lower relative to the effort. Conversely, high-value orders over short distances might be prioritized, leading to more "Delivered" outcomes.

### Task 2: Alternative `is_peak_hour` Rule
**New Rule:** Created `is_peak_hour_v2` to include both lunch (12:00 - 14:00) and dinner (18:00 - 21:00) windows.
**Discussion:** By including the lunch rush, the model can capture a second major wave of high demand. This often improves performance because traffic and driver shortages during lunch affect delivery times and cancellation rates similarly to the dinner rush.

### Task 3: Adjusting `top_k` for `Item_Name_reduced`
**Values Tested:** `top_k` = 10, 30, and 50.
**Comparison & Results:** * **Accuracy:** Modifying the `top_k` items resulted in the exact same accuracy (0.8519) across all three variations.
* **Discussion:** This indicates that the `Item_Name` feature has very low predictive power regarding `Order_Status`. The Random Forest classifier relies heavily on stronger predictors (such as distance or traffic) and effectively ignores the item name, meaning that increasing the granularity of the food categories does not provide any new, useful patterns for the model to learn from.

### Task 4: Feature Selection using `SelectFromModel`
**Discussion:** Feature selection was highly beneficial. The `OneHotEncoder` creates many dummy columns for categories like `City` and `Payment_Method` that have near-zero predictive power. Filtering out this noise with `SelectFromModel` forces the Random Forest to rely only on strong signals (like distance and traffic). This results in a simpler, faster model that is less prone to overfitting, without sacrificing accuracy.
