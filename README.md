# 📊 End-to-End Customer Churn Prediction & Retention Optimization

## 🎯 Business Problem & Strategic Goal

Customer churn directly impacts recurring revenue and Customer Lifetime Value (LTV), while acquiring a new customer is often more expensive than retaining an existing one.

The objective of this project is to build an end-to-end **customer churn prediction system** that identifies customers who are at high risk of leaving, allowing the business to prioritize them for proactive retention campaigns.

Rather than optimizing solely for accuracy, this project follows a **business-driven machine learning approach**, where model selection and classification thresholds are evaluated based on the trade-off between:

* Detecting as many potential churners as possible.
* Avoiding excessive false alarms and unnecessary campaign costs.
* Maintaining a practical balance between Precision and Recall.
* Translating model predictions into actionable business decisions.

### 🎯 Business Objective

The primary objective is to maximize the identification of potential churners while maintaining an acceptable level of precision.

This makes **Recall for the churn class** an especially important metric because failing to identify a customer who is likely to churn represents a potential missed retention opportunity.

---

## 🏗️ Project Workflow

The project follows an end-to-end machine learning workflow:

```text
Raw Customer Data
       │
       ▼
Exploratory Data Analysis (EDA)
       │
       ▼
Data Cleaning & Feature Engineering
       │
       ▼
Train / Test Split
       │
       ▼
Leakage-Free Preprocessing Pipeline
       │
       ├── Numerical Features
       │      ├── Missing Value Imputation
       │      └── Standard Scaling
       │
       └── Categorical Features
              ├── Missing Value Handling
              └── Encoding
       │
       ▼
Model Training & Cross-Validation
       │
       ├── Logistic Regression + SMOTE
       ├── Random Forest + SMOTE
       ├── Soft Voting Classifier
       └── XGBoost + Class Weighting
       │
       ▼
Model Comparison
       │
       ▼
XGBoost Selected as Champion Model
       │
       ▼
Decision Threshold Analysis
       │
       ▼
Threshold = 0.45
       │
       ▼
Final Evaluation on Unseen Test Set
       │
       ▼
Business Interpretation & ROI Scenario
```

---

## 🔒 Leakage-Free Pipeline Architecture

A major focus of this project was preventing **Data Leakage** during preprocessing, resampling, and model evaluation.

All transformations that learn information from the data are performed inside the appropriate `Pipeline` / `imblearn.pipeline.Pipeline` structures and are fitted only on the training portion of each Cross-Validation fold.

This ensures that information from validation or test data is not used during preprocessing or model training.

### Pipeline Concept

```text
Training Data
     │
     ▼
Feature Preprocessing
     │
     ├── Numerical
     │     ├── Imputation
     │     └── Scaling
     │
     └── Categorical
           ├── Imputation
           └── Encoding
     │
     ▼
Class Imbalance Handling
     │
     ▼
Classifier
     │
     ▼
Predicted Probabilities
     │
     ▼
Decision Threshold
```

### Why This Matters

A common mistake in machine learning projects is to preprocess or resample the entire dataset before Cross-Validation.

For example:

```text
❌ Incorrect:

Entire Dataset
      ↓
SMOTE / Scaling / Encoding
      ↓
Cross-Validation
```

This can allow information from validation folds to influence the training process.

Instead, this project follows:

```text
✅ Correct:

Training Fold
      ↓
Preprocessing
      ↓
Resampling / Class Balancing
      ↓
Model Training
      ↓
Validation Fold
```

This provides a more reliable estimate of model performance.

---

# 🧪 Model Exploration & Comparative Results

Several classification strategies were evaluated using **5-Fold Cross-Validation**.

The main evaluation metrics were:

* **Recall** — ability to identify actual churners.
* **Precision** — proportion of predicted churners who actually churn.
* **F1-Score** — balance between Precision and Recall.
* **ROC-AUC** — ability to rank churners above non-churners.
* **Accuracy** — overall classification correctness.

| Model Strategy                | Strategy Type           | Train Recall | Test Recall | Test Precision | Test F1-Score | Test ROC-AUC | Test Accuracy |
| ----------------------------- | ----------------------- | -----------: | ----------: | -------------: | ------------: | -----------: | ------------: |
| **Logistic Regression**       | Linear + SMOTE          |       81.12% |  **79.47%** |         52.66% |        0.6334 |       0.8437 |        75.34% |
| **Random Forest**             | Tree Ensemble + SMOTE   |       74.20% |      70.76% |         58.08% |        0.6378 |       0.8421 |        78.45% |
| **Soft Voting Classifier**    | Tri-Model Ensemble      |       78.39% |      74.68% |         55.71% |        0.6380 |       0.8490 |        77.28% |
| **XGBoost (Default 0.50)**    | Cost-Sensitive Weighted |       75.10% |      72.69% |     **57.61%** |    **0.6426** |   **0.8507** |    **78.33%** |
| **XGBoost (Tuned @ 0.45) 🏆** | Threshold-Optimized     |       80.45% |  **78.01%** |         53.85% |        0.6372 |   **0.8507** |        76.19% |

### 🏆 Why XGBoost?

XGBoost was selected as the **Champion Model** because it provided a strong overall balance between:

* Churn detection capability.
* Precision.
* F1 performance.
* ROC-AUC.
* Flexibility for probability-based decision making.
* Compatibility with a business-oriented threshold optimization strategy.

The default XGBoost model achieved the strongest ROC-AUC among the evaluated strategies:

> **ROC-AUC = 0.8507**

The final decision threshold was then tuned separately to align the model's predictions more closely with the business objective of identifying more potential churners.

---

# 🎚️ Decision Threshold Optimization

Most binary classifiers use a default probability threshold of:

```text
0.50
```

However, the default threshold is not necessarily optimal for a business problem.

For customer churn, the cost of missing a potential churner can be significant because the business loses the opportunity to intervene.

Therefore, the model's predicted probabilities were evaluated across multiple decision thresholds.

### Threshold Comparison

|     Threshold |     Recall |  Precision |   F1-Score |   Accuracy |
| ------------: | ---------: | ---------: | ---------: | ---------: |
|        0.3500 |     0.8605 |     0.4905 |     0.6248 |     0.7231 |
|        0.4000 |     0.8279 |     0.5138 |     0.6341 |     0.7439 |
| **0.4500 🏆** | **0.7801** | **0.5385** | **0.6372** | **0.7619** |
|        0.5000 |     0.7256 |     0.5702 |     0.6386 |     0.7799 |
|        0.5314 |     0.6950 |     0.5960 | **0.6417** |     0.7920 |

### Why 0.45?

The threshold of **0.45** was selected to prioritize higher churn detection compared with the default XGBoost threshold of 0.50 while maintaining an acceptable Precision level.

Compared with the default threshold:

```text
Threshold 0.50
Recall = 72.69%

        ↓

Threshold 0.45
Recall = 78.01%
```

This represents an increase of approximately:

```text
+5.32 percentage points in churn Recall
```

The trade-off is a reduction in Precision from:

```text
57.61% → 53.85%
```

and a reduction in Accuracy from:

```text
78.33% → 76.19%
```

For this business scenario, the trade-off is acceptable because the primary objective is to identify more potential churners for retention intervention rather than maximize overall Accuracy.

> **Important:** The 0.45 threshold is not the highest-F1 threshold in the tested table. It was selected because the project prioritizes higher churn Recall and a practical Precision/Recall trade-off.

---

# 🏆 Final Champion Model

The final production candidate is:

```text
XGBoost
+
Leakage-Free Preprocessing Pipeline
+
Cost-Sensitive Learning
+
Decision Threshold = 0.45
```

The model produces a churn probability for every customer:

```text
P(Churn = 1 | Customer Features)
```

The final business decision is then:

```text
If Probability >= 0.45
        ↓
High Risk / Target for Retention

If Probability < 0.45
        ↓
Low Risk / Do Not Prioritize
```

This separates two different concepts:

```text
Model Probability
       ↓
"How likely is this customer to churn?"

Decision Threshold
       ↓
"At what probability should the business take action?"
```

---

# 🧪 Final Evaluation on Unseen Test Data

After completing model development and threshold analysis, the final Champion Model was evaluated on a completely **unseen hold-out test set containing 1,405 customers**.

### Final Test Results

```text
🔥 FINAL TEST SET EVALUATION
Threshold = 0.45
============================================================

              precision    recall  f1-score   support

           0     0.9082    0.7331    0.8114      1053
           1     0.4937    0.7784    0.6042       352

    accuracy                         0.7445      1405

   macro avg     0.7010    0.7558    0.7078      1405
weighted avg     0.8044    0.7445    0.7594      1405

🔹 Test ROC-AUC Score: 0.8433

============================================================
```

### Key Final Metrics

| Metric              |     Result |
| ------------------- | ---------: |
| **Churn Precision** |     49.37% |
| **Churn Recall**    | **77.84%** |
| **Churn F1-Score**  |     0.6042 |
| **Accuracy**        |     74.45% |
| **ROC-AUC**         |     0.8433 |

The final model successfully identifies approximately **77.84% of actual churners** in the unseen test set.

---

# 📉 Confusion Matrix Analysis

The final test-set confusion matrix contains:

|                         | Actual Non-Churn | Actual Churn |
| ----------------------- | ---------------: | -----------: |
| **Predicted Non-Churn** |              772 |           78 |
| **Predicted Churn**     |              281 |          274 |

### Interpretation

* **True Positives (274):**
  Customers correctly identified as churners.

* **False Negatives (78):**
  Actual churners that the model failed to identify.

* **True Negatives (772):**
  Customers correctly identified as non-churners.

* **False Positives (281):**
  Customers predicted as high-risk who did not actually churn.

From a retention perspective, the **274 True Positives** represent customers who could potentially be prioritized for proactive retention campaigns.

---

# 💡 Business Impact & Illustrative ROI Scenario

To demonstrate how model predictions could translate into business value, consider the following **illustrative assumptions**:

* **Estimated Customer Lifetime Value (LTV): $200**
* **Retention Campaign Cost per Targeted Customer: $10**

These values are assumptions used to demonstrate the financial reasoning and are **not measured outcomes from the dataset**.

## 1. Potential Customer Value at Risk

The model correctly identified:

```text
274 True Positives
```

Assuming an LTV of $200 per customer:

```text
274 × $200 = $54,800
```

Therefore, the model identified customers associated with approximately:

> **$54,800 of potential customer lifetime value at risk.**

This does **not** mean that $54,800 was actually saved. Actual revenue preservation would depend on the effectiveness of the retention campaign.

---

## 2. Estimated Campaign Cost

The model generated:

```text
281 False Positives
```

Assuming a $10 campaign cost per targeted customer:

```text
281 × $10 = $2,810
```

This represents an estimated cost associated with interventions directed toward customers who were incorrectly classified as high risk.

---

## 3. Illustrative Business Interpretation

Under these assumptions:

```text
Potential LTV identified at risk = $54,800
Estimated campaign cost         = $2,810
```

The model therefore demonstrates a potentially attractive business case **if retention interventions successfully prevent a meaningful portion of predicted churn**.

The actual ROI would depend on additional business information such as:

* Retention campaign success rate.
* Actual customer LTV.
* Campaign cost.
* Discount or incentive cost.
* Incremental retention rate.
* Revenue contribution after retention.
* Customer segment.
* Long-term retention behavior.

A production system should therefore optimize the threshold using an explicit business cost/benefit function once these values are available.

---

# 📊 Business Decision Framework

The final system can be interpreted as a decision-support tool:

```text
                Customer
                    │
                    ▼
            XGBoost Pipeline
                    │
                    ▼
          Churn Probability
                    │
                    ▼
            Threshold = 0.45
                    │
          ┌─────────┴─────────┐
          │                   │
     Probability          Probability
        < 0.45               ≥ 0.45
          │                   │
          ▼                   ▼
     Low Risk             High Risk
                              │
                              ▼
                     Retention Campaign
```

This approach allows the business to move from:

> **Prediction → Decision → Action**

rather than treating the ML model as an isolated prediction system.

---

# 🚀 Inference / Production Usage

The trained pipeline and optimized threshold can be saved together as a single artifact.

Example:

```python
import joblib
import pandas as pd

# Load trained pipeline and threshold
artifacts = joblib.load("churn_xgboost_pipeline.joblib")

pipeline = artifacts["pipeline"]
threshold = artifacts["optimal_threshold"]

# Load new customer data
new_data = pd.read_csv("data/new_customers.csv")

# Generate churn probabilities
probs = pipeline.predict_proba(new_data)[:, 1]

# Apply business decision threshold
predictions = (probs >= threshold).astype(int)

# Append predictions
new_data["Churn_Probability"] = probs

new_data["Risk_Status"] = [
    "High Risk" if prediction == 1 else "Low Risk"
    for prediction in predictions
]

print(new_data.head())
```

### Example Output

```text
Customer_ID    Churn_Probability    Risk_Status
------------------------------------------------
1001           0.71                 High Risk
1002           0.23                 Low Risk
1003           0.58                 High Risk
1004           0.18                 Low Risk
```

The resulting predictions can then be integrated into a retention workflow where high-risk customers are prioritized for further business action.

---

# 🧠 Key Machine Learning Concepts Demonstrated

This project demonstrates practical understanding of several important Data Science concepts:

### Data Science

* Exploratory Data Analysis (EDA)
* Data Cleaning
* Missing Value Handling
* Feature Engineering
* Numerical Feature Scaling
* Categorical Encoding
* Class Imbalance Handling
* Model Evaluation
* Cross-Validation
* Probability-Based Classification
* Decision Threshold Optimization
* Business-Oriented Model Selection

### Machine Learning

* Logistic Regression
* Random Forest
* XGBoost
* Ensemble Learning
* Soft Voting
* Cost-Sensitive Learning
* SMOTE
* Precision / Recall Trade-off
* F1-Score
* ROC-AUC
* Confusion Matrix

### ML Engineering

* Leakage-Free Pipelines
* Reproducible preprocessing
* Serialized model artifacts
* Separate training and inference workflows
* Threshold persistence alongside the trained model

---

# 🛠️ Technical Stack

### Programming Language

* Python 3.10+

### Machine Learning

* Scikit-Learn
* XGBoost
* Imbalanced-Learn

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Model Deployment / Persistence

* Joblib
* Serialized XGBoost Pipeline

---

# 📁 Suggested Project Structure

```text
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Model_Comparison.ipynb
│   └── 04_Final_Model.ipynb
│
├── models/
│   └── churn_xgboost_pipeline.joblib
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── requirements.txt
│
└── README.md
```

---

# 🎯 Final Takeaways

The final solution is not simply a churn classifier optimized for Accuracy.

It is a **business-oriented machine learning pipeline** designed to:

1. Prevent Data Leakage through proper pipeline architecture.
2. Compare multiple classification strategies using Cross-Validation.
3. Select **XGBoost** as the final modeling approach.
4. Optimize the classification threshold according to the business objective.
5. Use a **0.45 decision threshold** to prioritize higher churn Recall.
6. Evaluate the final system on completely unseen test data.
7. Translate model predictions into actionable retention decisions.
8. Demonstrate the potential financial impact through an illustrative ROI framework.

### Final Model

```text
🏆 Champion Model
XGBoost
+
Leakage-Free Pipeline
+
Cost-Sensitive Learning
+
Decision Threshold = 0.45
+
Unseen Test ROC-AUC = 0.8433
+
Unseen Test Churn Recall = 77.84%
```

> **The core objective is not simply to predict who will churn, but to identify customers where proactive retention action can create measurable business value.**
