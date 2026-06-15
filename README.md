# 📊 End-to-End Customer Churn Prediction & Retention Insights

## 🎯 Business Problem
Customer retention is a critical challenge for telecom and subscription-based companies. Retaining an existing customer is significantly cheaper than acquiring a new one. This project aims to build a robust Machine Learning model to predict which customers are at a high risk of churning, enabling the marketing team to take proactive retention actions.

## 📈 Dataset & Class Imbalance Challenge
The dataset contains customer demographics, account information, and service details. 
* **The Catch:** The dataset suffers from a severe **Class Imbalance** (where loyal customers heavily outnumber churning customers). 
* Default models failed to catch the minority class (Churn), leading to low **Recall**.

## 🧪 The Data Science Journey (Experiments)
To solve this, the project followed a structured, experimental approach:
1. **Model Selection:** Evaluated `Logistic Regression`, `Decision Tree`, `Random Forest`, and `XGBoost` using 5-Fold Cross-Validation. **Random Forest** achieved the best potential.
2. **Experiment 1 (Baseline):** Trained default Random Forest. It suffered from overfitting and low Recall on the Churn class due to imbalance.
3. **Experiment 2 (Data-Level Approach):** Applied **SMOTE** to balance training data. It improved overall sensitivity but lacked structural stability on unseen test data.
4. **Experiment 3 (Algorithmic Approach - Champion Model):** Optimized Random Forest directly using `class_weight='balanced'` and executed a thorough **GridSearchCV** tuned for **F1-Score**.

## 🏆 Final Model Performance (The Results)
The final tuned Random Forest model achieved a highly robust performance on unseen real-world test data:
* **Recall (Class 1): 74%** — Effectively capturing nearly 3/4 of all churning customers!
* **Optimized Trade-off:** Maintained a strong balance between Precision and Recall to minimize false alarms while maximizing customer saves.

## 🚨 Top Business Insights
Using **Feature Importance**, we discovered the top drivers behind customer churn:
1. **Contract Type:** Short-term (Month-to-month) contracts are highly vulnerable to churn.
2. **Tenure:** Newer customers have a significantly higher risk of leaving.
3. **Internet Service Type:** Fiber optic users showed a strange correlation with higher churn (requires infrastructure/pricing inspection).

## 💻 Technical Stack
* **Language:** Python 3
* **Libraries:** Pandas, NumPy, Scikit-Learn, Imbalanced-Learn, Matplotlib, Seaborn
* **Production Ready:** The champion model is serialized and saved as `customer_churn_model.pkl` via `joblib`, fully optimized for deployment.