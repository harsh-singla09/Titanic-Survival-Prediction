import pandas as pd
import xgboost as xgb
import joblib

from utils import preprocess_data
from utils import create_features

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv(
    r"C:\Users\Munish kumar\Desktop\45-days\DAY-1\Titanic_Dataset.csv"
)

# =========================================
# DATA CLEANING
# =========================================

# Handle missing values
df = preprocess_data(df)
df = create_features(df)


# =========================================
# FEATURES & TARGET
# =========================================

X = df[['Sex', 'Age', 'Pclass' ,'IsAlone']]

y = df['Survived']

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# BASE MODEL
# =========================================

model = xgb.XGBClassifier(
    eval_metric='logloss'
)

# =========================================
# HYPERPARAMETER TUNING
# =========================================

param_grid = {

    'max_depth': [2, 3, 5, 10],

    'learning_rate': [0.01, 0.1, 0.2],

    'n_estimators': [50, 100, 200]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5
)

# Train GridSearch
grid_search.fit(X_train, y_train)

# =========================================
# BEST MODEL
# =========================================

best_model = grid_search.best_estimator_

# =========================================
# BEST PARAMETERS
# =========================================

print("\n===== BEST PARAMETERS =====")

print(f"Best Parameters: {grid_search.best_params_}")

print(f"Best CV Score: {grid_search.best_score_}")

# =========================================
# CROSS VALIDATION
# =========================================

cv_scores = cross_val_score(
    best_model,
    X,
    y,
    cv=5
)

print("\n===== CROSS VALIDATION =====")

print(f"Cross Validation Scores: {cv_scores}")

print(f"Mean CV Accuracy: {cv_scores.mean():.2%}")

# =========================================
# PREDICTIONS
# =========================================

y_probs = best_model.predict_proba(X_test)[:, 1]

threshold = 0.7

y_pred = (y_probs >= threshold).astype(int)

# =========================================
# CONFUSION MATRIX
# =========================================

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\n===== CONFUSION MATRIX =====")

print(cm)

print(f"TP: {tp}")

print(f"TN: {tn}")

print(f"FP: {fp}")

print(f"FN: {fn}")

# =========================================
# TRAINING & TESTING ACCURACY
# =========================================

train_preds = best_model.predict(X_train)

test_preds = best_model.predict(X_test)

train_acc = accuracy_score(
    y_train,
    train_preds
)

test_acc = accuracy_score(
    y_test,
    test_preds
)

print("\n===== MODEL ACCURACY =====")

print(f"Training Accuracy: {train_acc:.2%}")

print(f"Testing Accuracy: {test_acc:.2%}")

# =========================================
# CLASSIFICATION REPORT
# =========================================

target_names = [
    'Perished',
    'Survived'
]

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_names
    )
)

# =========================================
# FEATURE IMPORTANCE
# =========================================

importance = best_model.feature_importances_

feature_importance = pd.DataFrame({

    'Feature': X.columns,

    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\n===== FEATURE IMPORTANCE =====")

print(feature_importance)

# =========================================
# SAVE TRAINED MODEL
# =========================================

joblib.dump(
    best_model,
    'my_model.joblib'
)

print("\nModel Saved Successfully!")

# =========================================
# LOAD SAVED MODEL
# =========================================

loaded_model = joblib.load(
    'my_model.joblib'
)

print("Model Loaded Successfully!")