# Telco Churn Prediction

## Overview

Customer churn prediction is a critical challenge for telecom companies due to the highly competitive nature of the industry. This project focuses on building a predictive model to identify customers at risk of churn, enabling companies to take proactive steps to improve retention and reduce revenue loss.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)
- [Key Insights](#key-insights)
- [Model Performance](#model-performance)
- [Results and Recommendations](#results-and-recommendations)

---

## Problem Statement

Telecom companies face high customer churn rates, with some losing millions of dollars monthly due to dissatisfied customers switching to competitors. By predicting churn ahead of time, companies can focus retention efforts on high-risk customers and improve overall profitability.

---

## Objectives

This project aims to answer the following questions:
- What percentage of customers are likely to churn?
- Are there patterns in churn based on customer demographics or services?
- Which features have the most significant impact on predicting churn?
- Which machine learning model provides the best performance for churn prediction?

---

## Dataset Description

The dataset includes 7043 customer records with the following key features:

### Demographic Features:
- **Gender**: Male or Female
- **SeniorCitizen**: Whether the customer is a senior citizen (1 or 0)
- **Partner**: Whether the customer has a partner (Yes or No)
- **Dependents**: Whether the customer has dependents (Yes or No)

### Service Features:
- **PhoneService**: Whether the customer has phone service (Yes or No)
- **InternetService**: DSL, Fiber optic, or No
- **OnlineSecurity, TechSupport, DeviceProtection**: Whether the customer subscribed to these services (Yes, No, or No internet service)

### Account Features:
- **Contract**: Month-to-month, One year, or Two year
- **PaymentMethod**: Electronic check, Mailed check, Bank transfer, or Credit card
- **MonthlyCharges**: Monthly amount charged to the customer
- **TotalCharges**: Total amount charged to the customer
- **Churn**: Target variable (Yes or No)

---

## Methodology

1. **Exploratory Data Analysis (EDA)**:
   - Analyzed demographic and service patterns.
   - Visualized key trends and distributions using interactive plots.
   - Observed high churn rates among customers with month-to-month contracts and no online security.

2. **Data Preprocessing**:
   - Handled missing values and outliers.
   - Encoded categorical variables using target-guided ordinal encoding.
   - Balanced the dataset using BorderlineSMOTE to address class imbalance.
   - Scaled numerical features with `StandardScaler`.

3. **Model Building**:
   - Evaluated multiple classification models: Logistic Regression, Random Forest, K-Nearest Neighbors, Decision Tree, AdaBoost, XGBoost, and Stacking Classifier.
   - Performed hyperparameter tuning using `RandomizedSearchCV`.

4. **Feature Importance**:
   - Identified key features contributing to churn prediction (e.g., contract type, tenure, and online security).

---

## Technologies Used

- **Languages**: Python
- **Libraries**:
  - Data Handling: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`, `plotly`
  - Machine Learning: `scikit-learn`, `xgboost`, `mlens`
  - Data Balancing: `imblearn`

---

## Key Insights

- **Churn Drivers**:
  - Customers with month-to-month contracts, electronic check payments, and higher monthly charges are at higher risk.
  - Lack of online security and tech support significantly contributes to churn.
- **Customer Loyalty**:
  - Customers with longer tenure and two-year contracts are less likely to churn.
  - Senior citizens show a higher churn rate.

---

## Model Performance

### Best Model: Stacking Classifier

| Metric            | Score  |
|-------------------|--------|
| Accuracy          | 86%    |
| AUC-ROC           | 0.86   |

### Visualization:
- Confusion Matrix and ROC curve confirm the model’s reliability.

---

## Results and Recommendations

### Results:
- The Stacking Classifier outperformed other models in terms of stability and accuracy.
- Feature importance analysis highlighted contract type and tenure as critical predictors.

### Recommendations:
- **Retention Strategies**:
  - Provide incentives to customers with month-to-month contracts to switch to long-term plans.
  - Improve service offerings, particularly online security and tech support.
  - Offer loyalty programs for senior citizens and at-risk customers.
- **Proactive Measures**:
  - Target high-risk customers with personalized retention strategies.
  - Monitor customers with electronic check payments for early signs of dissatisfaction.

---

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/PRABHAJMANI/Telco-Churn-Analysis-and-Prediction.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook Telco_Churn_Prediction.ipynb
   ```

---