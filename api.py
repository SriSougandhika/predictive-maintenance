import pandas as pd
import pickle
from flask import Flask, request, jsonify
import numpy as np
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load models and scaler
with open('logistic_regression_model.pkl', 'rb') as f:
    log_reg = pickle.load(f)
with open('random_forest_model.pkl', 'rb') as f:
    random_forest = pickle.load(f)
with open('svc_model.pkl', 'rb') as f:
    svc = pickle.load(f)
with open('gradient_boosting_model.pkl', 'rb') as f:
    grad_boost = pickle.load(f)
with open('knn_model.pkl', 'rb') as f:
    knn = pickle.load(f)
with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# Preprocess function (assuming the one you provided)
def preprocess_data(df):
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN values if needed
    df.dropna(inplace=True)

    # Feature engineering - Creating a new feature: Torque per rpm
    df['Torque per rpm'] = df['Torque [Nm]'] / df['Rotational speed [rpm]']

    # Convert 'Product ID' into quality variants by extracting the first letter
    df['Product_Quality'] = df['Product ID'].apply(lambda x: x[0])

    # Drop 'Product ID' and 'UID'
    df = df.drop(columns=['Product ID', 'UID'])

    # Label encode the categorical features
    label_encoder = LabelEncoder()
    df['Type'] = label_encoder.fit_transform(df['Type'])
    df['Product_Quality'] = label_encoder.fit_transform(df['Product_Quality'])

    # Features and target
    x = df.drop(columns=['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])  # Drop target columns for input data

    # Scale the numerical features
    x_scaled = scaler.transform(x)  # Use the already fitted scaler
    return x_scaled


@app.route('/predict', methods=['POST'])
def predict():
    # Load the CSV file from the form
    file = request.files['file']
    data = pd.read_csv(file)

    # Preprocess the data
    X_scaled = preprocess_data(data)
    print(X_scaled)
    print(log_reg.predict(X_scaled))
    # Predict using each model
    predictions = {
        'Logistic Regression': log_reg.predict(X_scaled).tolist(),
        'Random Forest': random_forest.predict(X_scaled).tolist(),
        'Support Vector Classifier': svc.predict(X_scaled).tolist(),
        'Gradient Boosting': grad_boost.predict(X_scaled).tolist(),
        'K-Nearest Neighbors': knn.predict(X_scaled).tolist(),
        'K-Means Clustering': kmeans.predict(X_scaled).tolist(),
    }

    return jsonify(predictions)


if __name__ == '__main__':
    app.run(debug=True)
