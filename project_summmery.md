# Titanic Survival Prediction

## Project Overview
This project predicts whether a passenger would survive the Titanic disaster using Machine Learning.

## Dataset
Titanic Dataset (891 passengers)

## Features Used
- Sex
- Age
- Pclass
- IsAlone

## Data Preprocessing
- Filled missing Age values using mean.
- Encoded Sex (male=0, female=1).
- Created IsAlone feature.

## Models Trained
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

## Best Model
XGBoost

## Performance
- Test Accuracy: 82.68%
- ROC-AUC Score: 0.8672

## Deployment
Input-based prediction system using joblib model loading.

## Conclusion
XGBoost achieved the best balance of accuracy and generalization.