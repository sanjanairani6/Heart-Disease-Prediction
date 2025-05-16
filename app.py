# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model/heart_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        scaled = scaler.transform([features])
        prediction = model.predict(scaled)[0]

        # You can adjust logic here
        if prediction == 0:
            result = "Low risk of heart disease"
        elif prediction == 1:
            result = "High risk of heart disease"
        else:
            result = "Moderate risk of heart disease"

        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
