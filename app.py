from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load trained model and label encoder
model = joblib.load("disease_prediction_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load dataset to get symptom columns
df = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")
symptom_columns = df.columns[:-1]  # Extract symptom column names

@app.route("/")
def home():
    return "Disease Prediction API is Running!"

# API endpoint for disease prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Get JSON input
        user_symptoms = data.get("symptoms", [])  # Extract symptoms list

        # Create binary symptom input vector
        input_vector = np.zeros(len(symptom_columns))
        for symptom in user_symptoms:
            if symptom in symptom_columns:
                input_vector[symptom_columns.get_loc(symptom)] = 1  # Mark as 1 if present

        # Make prediction
        prediction = model.predict([input_vector])
        predicted_disease = label_encoder.inverse_transform(prediction)[0]

        return jsonify({"predicted_disease": predicted_disease})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
