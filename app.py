# load the libraries
from logging import debug
from math import pi
import pickle
from urllib import request
from flask import Flask, resquests, jsonify
import numpy as np
import pandas as pd

# load the flask app
app=Flask(__name__)

# load the trained model
with open('/Users/macbookpro/Desktop/Heart Disease API Using Flask/heart_disease_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# load the scaler file
with open('/Users/macbookpro/Desktop/Heart Disease API Using Flask/heart_disease_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route("/")
def home():
    return "App is running" # here is where the HTML page come into place

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # get the JSON data from the API request
        data = request.get_json()
        input_data = pd.DataFrame([data])

        # check if input is provided
        if not data:
            return jsonify({"error": "Input data not provided"}), 400
        # validate input columns
        # age	sex	cp	trestbps	chol	fbs	restecg	thalach	exang	oldpeak	slope	ca	thal
        required_columns = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
        if not all(col in input_data.columns for col in required_columns):
            return jsonify({"error": f"Required columns missing. Required columns: {required_columns}"}), 400
        
        # scale the data
        scaled_data = model.transform(input_data)
        # 47:00

        # make prediction
        prediction = model.predict(scaled_data)

        response = {
            "prediction": "Diabetes" if prediction[0] == 1 else "No Diabetes"
        }

        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__=="__main__":
    app.run(debug=True)