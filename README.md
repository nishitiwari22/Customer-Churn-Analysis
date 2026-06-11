# Customer Churn Prediction using Machine Learning 🚀

A Machine Learning-based web application that predicts customer churn using customer demographic and financial data. This project combines **Data Analytics, Exploratory Data Analysis (EDA), Machine Learning, and Streamlit Deployment** to help businesses identify customers who are likely to discontinue services.

---

# 📌 Project Overview

Customer churn prediction is one of the most important business applications of Machine Learning. Companies lose significant revenue when customers stop using their services. This project aims to predict customer churn using multiple machine learning algorithms and provide actionable business insights.

The project includes:

* Data Cleaning & Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Machine Learning Model Training
* Model Evaluation & Comparison
* Streamlit Web Application Deployment

---

# 🎯 Objectives

* Analyze customer behavior and identify churn patterns
* Perform data preprocessing and feature scaling
* Compare multiple ML algorithms
* Evaluate models using classification metrics
* Deploy the best-performing model using Streamlit
* Generate business insights for customer retention

---

# 🛠️ Technologies Used

| Category             | Technologies        |
| -------------------- | ------------------- |
| Programming Language | Python              |
| Data Analysis        | Pandas, NumPy       |
| Visualization        | Matplotlib, Seaborn |
| Machine Learning     | Scikit-learn        |
| Deployment           | Streamlit           |
| Version Control      | Git & GitHub        |

---

# 📂 Dataset Features

The dataset contains 10,000+ customer records with attributes such as:

| Feature         | Description               |
| --------------- | ------------------------- |
| CreditScore     | Customer credit score     |
| Geography       | Customer country/location |
| Gender          | Male/Female               |
| Age             | Customer age              |
| Tenure          | Years with bank           |
| Balance         | Account balance           |
| NumOfProducts   | Number of bank products   |
| HasCrCard       | Credit card ownership     |
| IsActiveMember  | Active membership status  |
| EstimatedSalary | Customer salary           |
| Exited          | Churn status              |

---

# 📊 Exploratory Data Analysis (EDA)

Performed detailed EDA to identify trends and customer behavior patterns.

### Key Insights:

* Customers with lower tenure showed higher churn probability
* Older customers had relatively higher churn rates
* Active members were less likely to churn
* Customers with fewer products demonstrated higher churn tendencies
* Balance and salary influenced churn behavior in certain segments

### Visualizations Included:

* Churn Distribution
* Correlation Heatmap
* Age vs Churn Analysis
* Tenure vs Churn
* Confusion Matrix
* ROC Curve

---

# 🤖 Machine Learning Models Used

The following algorithms were implemented and compared:

| Model                | Purpose                        |
| -------------------- | ------------------------------ |
| Logistic Regression  | Baseline binary classification |
| Decision Tree        | Rule-based classification      |
| Random Forest        | Ensemble learning              |
| XGBoost *(Optional)* | Advanced boosting algorithm    |

---

# 📈 Model Evaluation

Models were evaluated using:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC-AUC Score

### Best Model Performance

| Metric   | Score |
| -------- | ----- |
| Accuracy | 80%   |
| ROC-AUC  | 0.85  |

---

# 🌐 Streamlit Deployment

An interactive Streamlit web application was developed for real-time churn prediction.

### Features:

* User-friendly UI
* Real-time predictions
* Instant churn probability output
* Business-focused prediction insights

---

# 🧠 Business Recommendations

Based on model findings:

* Target low-tenure customers with loyalty programs
* Increase engagement for inactive members
* Monitor high-balance customers with low activity
* Personalize retention campaigns for high-risk users

---

# 🏗️ Project Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Streamlit Deployment
```

---

# 📸 Screenshots

## GitHub Repository Structure

(Add screenshot here)

## Streamlit Application

(Add deployed app screenshot here)

---

# 🚀 How to Run the Project

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 3️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

# 📁 Project Structure

```text
customer-churn-prediction/
│
├── dataset/
├── notebooks/
├── models/
├── app.py
├── churn_model.pkl
├── requirements.txt
├── README.md
└── screenshots/
```

---

# 🔮 Future Enhancements

* Implement Deep Learning models
* Real-time API integration
* Cloud deployment using AWS/GCP
* SHAP Explainable AI integration
* Automated retraining pipelines
* Dashboard analytics

---

# 📚 References

* Scikit-learn Documentation
* Pandas Documentation
* Streamlit Documentation
* Kaggle Customer Churn Dataset
* Research papers on customer churn prediction

---

# 👩‍💻 Author

**Nishi Tiwari**
MCA Student | AI/ML & Data Analytics Enthusiast

* Passionate about Machine Learning, Data Analytics, and building practical AI applications.
* Experienced with Python, Scikit-learn, Streamlit, and Data Visualization.

---

# ⭐ Project Highlights

✅ Machine Learning-based prediction system
✅ Real-world business use case
✅ Comparative model evaluation
✅ Interactive Streamlit deployment
✅ Business insights & recommendations
✅ End-to-end ML workflow implementation
