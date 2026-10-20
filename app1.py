
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
saved_data = joblib.load("student_model.joblib")
model = saved_data["model"]
features = saved_data["features"]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        studytime = int(request.form["studytime"])
        failures = int(request.form["failures"])
        absences = int(request.form["absences"])
        G1 = int(request.form["G1"])
        G2 = int(request.form["G2"])

        student_data = pd.DataFrame(
            [[studytime, failures, absences, G1, G2]],
            columns=features
        )

        prediction = model.predict(student_data)[0]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
