import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


model = joblib.load(r"C:\Users\Munish kumar\Desktop\Titanic-project\models\my_model.joblib")

df = pd.read_csv(
    r"C:\Users\Munish kumar\Desktop\45-days\DAY-1\Titanic_Dataset.csv"
)

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1}).astype(int)

df['IsAlone'] = (
    df['SibSp'] + df['Parch'] + 1
).apply(lambda x: 1 if x == 1 else 0)

X = df[['Sex', 'Age', 'Pclass', 'IsAlone']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

y_pred = model.predict(X_test)

#------------------------------------
#Confusion Matrix heatmap
#------------------------------------

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred
)

plt.title("Confusion Matrix")
plt.show()




#------------------------------------
#Feature Importance Chart
#------------------------------------

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(6,4))

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title("Feature Importance")

plt.xlabel("Features")
plt.ylabel("Importance")

plt.show()




plt.savefig("confusion_matrix.png")
plt.savefig("feature_importance.png")