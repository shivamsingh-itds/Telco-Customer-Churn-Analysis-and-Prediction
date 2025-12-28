# Telco Customer Churn Analysis and Prediction

## 📌 Project Overview
Customer churn is a critical challenge in the telecommunications industry, as retaining existing customers is often more cost-effective than acquiring new ones.  
This project focuses on analyzing customer behavior and building a machine learning model to predict customer churn using a real-world Telco dataset.

The project follows an end-to-end data science workflow, from data ingestion and preprocessing to model training, evaluation, and deployment-ready pipeline design.

---

## 🎯 Objectives
- Perform exploratory data analysis (EDA) to understand churn patterns
- Identify key factors contributing to customer churn
- Apply feature engineering techniques to improve model performance
- Build and evaluate multiple classification models
- Select the best-performing model based on business-relevant metrics

---

## 📂 Dataset
- **Source:** Kaggle – Telco Customer Churn Dataset
- **Records:** ~7,000 customers
- **Target Variable:** `Churn` (Yes / No)
- **Features Include:**
  - Demographics (gender, senior citizen, dependents)
  - Account information (tenure, contract type, payment method)
  - Services subscribed (internet, phone, security, streaming)
  - Billing information (monthly charges, total charges)

---

## 🔍 Exploratory Data Analysis (EDA)
Key insights from EDA:
- Customers on **month-to-month contracts** show significantly higher churn
- **Low tenure customers** are more likely to churn
- Customers without **TechSupport** or **OnlineSecurity** have higher churn rates
- Higher **monthly charges combined with short tenure** increase churn risk

EDA focused on churn-rate analysis and business-driven insights rather than only correlation metrics.

---

## 🛠 Feature Engineering
- Converted categorical variables into numerical format
- Handled service-related values such as *No internet service* and *No phone service*
- Created derived features like:
  - `AvgMonthlySpend`
- Scaled numerical features for model stability

---

## 🤖 Model Building
Models evaluated:
- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- AdaBoost
- Support Vector Machine
- **Gradient Boosting (Final Model)**

### 🔑 Model Selection Rationale
Gradient Boosting was selected as the final model due to:
- Best balance between accuracy and churn detection
- Strong generalization performance
- Stable results compared to single-tree models

---

## 📊 Final Model Performance
- **Accuracy:** ~80%
- Confusion matrix and classification metrics analyzed with focus on minimizing false negatives (missed churners)

---

## 🚀 How to Run the Project
```
python -m src.model_pipeline
``` 
--- 
## 📌 Key Learnings

- Importance of business-driven evaluation metrics in churn problems

- Handling real-world categorical inconsistencies

- Building CI/CD-safe ML pipelines using pathlib

- Modular and production-ready ML project structure
---

## 🔮 Future Improvements

- ROC-AUC and threshold tuning

- Class imbalance handling with class weights or SMOTE

- Model explainability using SHAP

- Deployment as a web application

---

## 👤 Author

**Shivam Singh**
Aspiring Data Scientist | Machine Learning Enthusiast

🔗 GitHub: [https://github.com/shivamsingh-itds]
🔗 LinkedIn: [www.linkedin.com/in/shivamsinghds]

---

⭐ If you find this project helpful, feel free to star the repository!