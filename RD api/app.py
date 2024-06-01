import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load your model
with open("api.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Extract input features from form
    try:
        feat = [
            int(request.form['Gender']),
            int(request.form['Age']),
            int(request.form['Diagnosis Age']),
            int(request.form['Blood Group']),
            int(request.form['Birth Order']),
            int(request.form['Marital Status']),
            int(request.form['Lifestyle']),
            int(request.form['Weight']),
            int(request.form['Obesity']),
            int(request.form['Obesity In Family']),
            int(request.form['Fast Food']),
            int(request.form['Smoking']),
            int(request.form['High BP']),
            int(request.form['Diabetes']),
        ]


        features = np.array(feat).reshape(1, -1)
        prediction = model.predict(features)[0]  # Get the prediction
        
        return render_template("result.html", prediction_text=f"Might be possible that you have {prediction}")
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
