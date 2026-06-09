import joblib
import pandas as pd

from utils import encode_gender
from utils import encode_isalone


# Load Saved Model
model = joblib.load(r"C:\Users\Munish kumar\Desktop\Titanic-project\models\my_model.joblib")


# Prediction Function
def predict_survival(sex, age, pclass,  isalone):

    # -----------------------------
    # Input Validation
    # -----------------------------

    # age validation
    if age < 0 or age > 120:
        print("Invalid Age!")
        return

    # pclass validation
    if pclass not in [1, 2, 3]:
        print("Pclass must be 1, 2, or 3")
        return

    

    # sex encoding
    sex = encode_gender(sex)
    isalone = encode_isalone(isalone)
    
    # -----------------------------
    # Create Input DataFrame
    # -----------------------------

    data = pd.DataFrame({
        'Sex': [sex],
        'Age': [age],
        'Pclass': [pclass],
        'IsAlone': [isalone]
    })

    # -----------------------------
    # Prediction
    # -----------------------------

    # probability of survival
    probability = model.predict_proba(data)[0][1]

# custom threshold
    threshold = 0.3

# manual prediction
    if probability >= threshold:
        prediction = 1
    else:
        prediction = 0



    # -----------------------------
    # Output Result
    # -----------------------------

    print("\n===== Prediction Result =====")
    print("\nFeature Summery:")
    
    print("\n===== Model Insights =====")

    if sex == 1:
        print("- Female passengers generally had higher survival rates.")

    if pclass == 1:
        print("- First-class passengers generally had better survival chances.")

    if isalone == 1:
        print("- Traveling alone may affect survival probability.")

    if prediction == 1:
        print("Passenger likely SURVIVED")
    else:
        print("Passenger likely PERISHED")

    print(f"Survival Probability: {probability:.2%}")

    if probability >= 0.80:
        print("High Confidence")

    elif probability >= 0.60:
        print("Medium Confidence")

    else:
        print("Low Confidence")

    

    print("\n===== Feature Importance =====")

    for feature, score in zip(
        ['Sex', 'Age', 'Pclass', 'IsAlone'],
        model.feature_importances_
    ):
        print(f"{feature}: {score:.4f}")



 # -----------------------------
    # Taking Input 
    # -----------------------------

predict_survival(
    sex= input("enter you gender MALE/FEMALE: "),
    age=int(input("Enter your age: ")),
    pclass=int(input("Enter your PClass: ")),
    isalone=input(" you are alone/not alone: ")
)
