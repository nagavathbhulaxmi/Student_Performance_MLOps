from flask import Flask, request, render_template_string
import pickle
from pathlib import Path

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "student_model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Student Performance Prediction</title>
</head>
<body>
    <h1>Student Performance Prediction</h1>
    <form method="POST">
        <label>Study Hours:</label>
        <input type="number" name="study_hours" min="0" max="24" step="any" required>
        <br><br>
        <label>Attendance (%):</label>
        <input type="number" name="attendance" min="0" max="100" step="any" required>
        <br><br>
        <label>Previous Marks:</label>
        <input type="number" name="previous_marks" min="0" max="100" step="any" required>
        <br><br>
        <button type="submit">Predict Marks</button>
    </form>
    {% if prediction is not none %}
        <h2>Predicted Final Marks: {{ prediction }} / 100</h2>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        previous_marks = float(request.form["previous_marks"])

        if not (0 <= study_hours <= 24 and
                0 <= attendance <= 100 and
                0 <= previous_marks <= 100):
            return "Please enter valid values.", 400

        result = model.predict([[
            study_hours, attendance, previous_marks
        ]])[0]

        prediction = round(max(0, min(100, result)), 2)

    return render_template_string(HTML, prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)