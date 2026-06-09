from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load(
    r"C:\Users\Munish kumar\Desktop\Titanic-project\models\my_model.joblib"
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        try:
            sex = request.form["sex"]
            age = float(request.form["age"])
            pclass = int(request.form["pclass"])
            isalone = request.form["isalone"]

            # Encode values
            sex = 1 if sex.lower() == "female" else 0

            isalone = (
                1 if isalone.lower() == "alone"
                else 0
            )

            # Create dataframe
            data = pd.DataFrame({
                "Sex": [sex],
                "Age": [age],
                "Pclass": [pclass],
                "IsAlone": [isalone]
            })

            # Prediction probability
            probability = model.predict_proba(data)[0][1]

            # Prediction
            prediction = (
                "SURVIVED"
                if probability >= 0.5
                else "PERISHED"
            )

            return render_template(
                "index.html",
                prediction=prediction,
                probability=f"{probability:.2%}"
            )

        except Exception as e:
            return render_template(
                "index.html",
                error=str(e)
            )
        
    if probability >= 0.80:
        confidence = "High"

    elif probability >= 0.60:
        confidence = "Medium"

    else:
        confidence = "Low"

    return render_template(
    "index.html",
    prediction=prediction,
    probability=f"{probability:.2%}",
    confidence=confidence
)


if __name__ == "__main__":
    app.run(debug=True)