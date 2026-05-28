# Customer Churn Prediction

## Overview

This project is a Machine Learning web application that predicts Customer Churn based on gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges


The project uses:

* Python
* Flask
* Scikit-learn
* Pandas
* NumPy
* HTML/CSS

The trained ML model is deployed using Flask so users can enter data through a web form and get predictions instantly.

---

## Problem Statement

Educational institutions often need to analyze factors affecting student performance.
This project predicts student scores using Machine Learning models trained on historical student data.

---

## Features

* Data preprocessing pipeline
* Model training and evaluation
* Flask web integration
* Real-time prediction
* Exception handling
* Logging system
* Modular project structure

---

## Project Structure

```bash
Second_ml/
│
├── artifacts/ 
│   ├── model.pkl 
│   ├── preprocessor.pkl 
│   ├── data.csv 
│   ├── test.csv 
│   └── train.csv 
│
├── notebook/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── EDA.ipynb 
│
├── src/
│   ├── components/  
│   ├──     ├── data_ingestion.py 
│   ├──     ├── data_transformation.py 
│   ├──     ├── model_evaluate.py 
│   ├──     └── model_training.py
│   ├── pipeline/ 
│   ├──     ├── predict_pipeline.py
│   ├──     └── train_pipeline.py
│   ├── exception.py 
│   ├── logger.py 
│   └── utils.py 
│
├── templates/ 
│   ├── predict.html
│   ├── index.html 
│   └── home.html 
│
├── app.py 
├── setup.py
├── requirements.txt 
└── README.md 

```

---

## Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Data Transformation
5. Model Training
6. Model Evaluation
7. Model Saving
8. Flask Deployment

---

## Installation

### Clone Repository

```bash
git clone <https://github.com/Tekraj123/second-end-to-end-ml-projects->
cd customer Churn
```

### Create Virtual Environment

```bash
conda create -p venv python==3.13.9 -y
```

### Activate Environment


#### Windows
```bash
conda activate venv
```
```bash
venv\Scripts\activate
```


### Install Requirements

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
python app.py
```

Application runs on:

```bash
http://127.0.0.1:5000
```

---

## Input Features

* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies
* Contract Type
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges


---

## Model Performance

| Model                        | Accuracy | Precision | Recall | F1 Score |
| ---------------------------- | -------- | --------- | ------ | -------- |
| Random Forest Classifier     |          |           |        |          |
| Decision Tree Classifier     |          |           |        |          |
| Gradient Boosting Classifier |          |           |        |          |
| Logistic Regression          |          |           |        |          |
| XGBoost Classifier           |          |           |        |          |
| CatBoost Classifier          |          |           |        |          |
| AdaBoost Classifier          |          |           |        |          |



---

## Future Improvements

* Docker deployment
* Cloud deployment
* User authentication
* Better UI design
* Model monitoring

---

## Author

Tekraj Upadhyaya

GitHub: <https://github.com/Tekraj123/second-end-to-end-ml-projects->

