from flask import Flask, render_template, request
import joblib
import numpy as np
import csv
import os
import pandas as pd

# ================================
# ⚡ Load exact-reproduction model
# ================================
model = joblib.load("D:/VMK/flask_app/model_exact.save")
scaler = joblib.load("D:/VMK/flask_app/transform_exact.save")

# Feature order used during training
feature_order = ['u_q', 'coolant', 'u_d', 'motor_speed', 'i_d', 'i_q', 'ambient']

app = Flask(__name__)

# CSV file to store prediction history
csv_file = "prediction_history.csv"

# Initialize CSV file if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(feature_order + ["predicted_temperature", "temp_state"])

# ================================
# Home page
# ================================
@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

# ================================
# Glossary page
# ================================
@app.route("/glossary")
def glossary():
    return render_template("glossary.html")

# ================================
# Manual prediction page
# ================================
@app.route("/manual_predict", methods=["GET", "POST"])
def manual_predict():
    prediction = None
    temp_state = None
    if request.method == "POST":
        try:
            # Collect input values from form in the correct feature order
            input_values = [
                float(request.form['u_q']),
                float(request.form['coolant']),
                float(request.form['u_d']),
                float(request.form['motor_speed']),
                float(request.form['i_d']),
                float(request.form['i_q']),
                float(request.form['ambient'])
            ]

            # Prepare input and scale
            input_array = np.array([input_values])
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]
            prediction = round(prediction, 2)

            # Determine temperature state
            if prediction < 60:
                temp_state = "Safe"
            elif 60 <= prediction <= 80:
                temp_state = "Warm"
            else:
                temp_state = "Overheated"

            # Save prediction to CSV
            with open(csv_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(input_values + [prediction, temp_state])

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("manual_predict.html", prediction=prediction, temp_state=temp_state)

# ================================
# Sensor-based prediction page
# ================================
@app.route("/sensor_predict", methods=["GET", "POST"])
def sensor_predict():
    prediction = None
    temp_state = None
    if request.method == "POST":
        try:
            # Simulate sensor readings
            u_q = np.random.uniform(-0.5, 0.5)
            coolant = np.random.uniform(20, 100)
            u_d = np.random.uniform(-0.5, 0.5)
            motor_speed = np.random.uniform(0, 2000)
            i_d = np.random.uniform(-5, 5)
            i_q = np.random.uniform(-5, 5)
            ambient = np.random.uniform(10, 40)

            # Prepare input in correct feature order
            input_values = [u_q, coolant, u_d, motor_speed, i_d, i_q, ambient]
            input_array = np.array([input_values])
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]
            prediction = round(prediction, 2)

            # Determine temperature state
            if prediction < 60:
                temp_state = "Safe"
            elif 60 <= prediction <= 80:
                temp_state = "Warm"
            else:
                temp_state = "Overheated"

            # Save simulated prediction to CSV
            with open(csv_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(input_values + [prediction, temp_state])

            prediction = f"Predicted Motor Temperature: {prediction} °C ({temp_state})"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("sensor_predict.html", prediction_text=prediction)

# ================================
# Prediction history page
# ================================
@app.route("/history")
def history():
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        records = df.to_dict(orient='records')
    else:
        records = []

    return render_template("history.html", records=records)

# ================================
# Run Flask app
# ================================
if __name__ == "__main__":
    app.run(debug=True)
