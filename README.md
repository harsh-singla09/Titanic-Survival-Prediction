# Titanic Survival Prediction System

## Project Overview

This project predicts whether a passenger would survive the Titanic disaster using Machine Learning.

The model is trained on the Titanic dataset and uses features such as:

* Sex
* Age
* Passenger Class (Pclass)
* IsAlone

The final model was trained using XGBoost and achieved strong performance on unseen test data.

---

## Project Structure

Titanic_Project/

├── data/

│   └── Titanic_Dataset.csv

├── models/

│   └── my_model.joblib

├── src/

│   ├── train.py

│   ├── predict.py

│   └── utils.py

├── outputs/

│   ├── feature_importance.png

│   └── roc_curve.png

├── requirements.txt

└── README.md

---

## Features Used

| Feature | Description                               |
| ------- | ----------------------------------------- |
| Sex     | Male or Female                            |
| Age     | Passenger Age                             |
| Pclass  | Passenger Class (1, 2, 3)                 |
| IsAlone | Whether the passenger was traveling alone |

---

## Model Performance

### XGBoost Results

* Training Accuracy: 84.13%
* Testing Accuracy: 82.68%
* Cross Validation Accuracy: 82.94%
* ROC-AUC Score: 0.8672

### Feature Importance

1. Sex → 0.6976
2. Pclass → 0.1791
3. FamilySize → 0.0786
4. Age → 0.0447
5. IsAlone → 0.0000

---

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* XGBoost
* Joblib
* Matplotlib

---

## Installation

Install required packages:

pip install -r requirements.txt

---

## Training the Model

Run:

python src/train.py

This will:

* Load and preprocess data
* Train the model
* Evaluate performance
* Save the trained model

---

## Making Predictions

Run:

python src/predict.py

Example Input:

Gender: Female

Age: 20

Pclass: 1

IsAlone: Not Alone

Example Output:

Passenger likely SURVIVED

Survival Probability: 90.85%

High Confidence

---

## Learning Outcomes

During this project, the following Machine Learning concepts were implemented:

* Data Cleaning
* Feature Engineering
* Train/Test Split
* Cross Validation
* Hyperparameter Tuning
* GridSearchCV
* RandomizedSearchCV
* Feature Selection
* ROC-AUC Analysis
* Threshold Tuning
* Model Explainability
* Model Saving & Loading
* Ensemble Learning
* XGBoost

---

## Author

Harsh Singla

Machine Learning & AI Enthusiast

45 Days AI/ML Learning Challenge
