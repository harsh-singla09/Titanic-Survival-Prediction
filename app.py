from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load trained model
model_path = os.path.join("models", "my_model.joblib")
model = joblib.load(model_path)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        sex = request.form["sex"]
        age = float(request.form["age"])
        pclass = int(request.form["pclass"])
        isalone = request.form["isalone"]

        # Encode values
        sex = 1 if sex.lower() == "female" else 0
        isalone = 1 if isalone.lower() == "alone" else 0

        # Create dataframe
        data = pd.DataFrame({
            "Sex": [sex],
            "Age": [age],
            "Pclass": [pclass],
            "IsAlone": [isalone]
        })

        # Predict probability
        probability = model.predict_proba(data)[0][1]

        # Prediction result
        prediction = (
            "SURVIVED"
            if probability >= 0.5
            else "PERISHED"
        )

        # Confidence level
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

    except Exception as e:
        return render_template(
            "index.html",
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)