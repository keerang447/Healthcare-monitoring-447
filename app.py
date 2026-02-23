from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("health_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    heart_rate = int(request.form['heart_rate'])
    sugar_level = int(request.form['sugar_level'])

    data = np.array([[heart_rate, sugar_level]])
    prediction = model.predict(data)

    if prediction[0] == 1:
        return "High Risk - Consult Doctor"
    else:
        return "Normal Condition"

if __name__ == '__main__':
    app.run(debug=True)