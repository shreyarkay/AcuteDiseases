"""
Routes and views for the Flask application.
"""

from datetime import datetime
from flask import render_template, request, jsonify
from DOCTOR_APP import app
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


@app.route('/')
@app.route('/home')
def home():
    return render_template("index.html")  # Ensure you have an index.html template


@app.route('/heart-disease-prediction', methods=['POST'])
def heart_disease_prediction():
    user_input = request.json  # Expecting JSON input
    # Load the Heart Disease Dataset
    url = "https://raw.githubusercontent.com/johnsmith88/heart-disease-dataset/master/heart.csv"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    data = pd.read_csv(url, names=columns, encoding='latin-1')

    # Data Preprocessing
    data['age'].fillna(data['age'].mean(), inplace=True)
    X = data.drop('target', axis=1)
    y = data['target']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle Class Imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Standardize the features
    scaler = StandardScaler()
    X_train_resampled = scaler.fit_transform(X_train_resampled)
    X_test = scaler.transform(X_test)

    # Create and fit the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)

    # User Input for Prediction
    input_data = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    result = 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'
    return jsonify({'prediction': result})


@app.route('/diabetes-prediction', methods=['POST'])
def diabetes_prediction():
    user_input = request.json  # Expecting JSON input
    # Load the Pima Indians Diabetes Dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(url, names=columns)

    # Data Preprocessing
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[cols_with_zero] = data[cols_with_zero].replace(0, np.nan)
    data[cols_with_zero] = data[cols_with_zero].fillna(data[cols_with_zero].median())

    X_full = data.drop(columns=['Outcome'])
    y = data['Outcome']

    # Normalize the features
    scaler = MinMaxScaler((-1, 1))
    X_full = scaler.fit_transform(X_full)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.15, random_state=42)

    # Create and fit the XGBoost model
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # User Input for Prediction
    input_data = np.array([user_input])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    result = "Diabetes" if prediction[0] == 1 else "No Diabetes"
    return jsonify({'prediction': result})


@app.route('/parkinsons-prediction', methods=['POST'])
def parkinsons_prediction():
    user_input = request.json  # Expecting JSON input
    # Load the Parkinson's Dataset
    dataset_path = 'parkinsons.data'  # Update this path to your local dataset
    data = pd.read_csv(dataset_path)

    # Data Preprocessing
    data = data.drop(columns=['name'])
    X_full = data.drop(columns=['status'])
    y = data['status']

    # Normalize the features
    scaler = MinMaxScaler((-1, 1))
    X_full = scaler.fit_transform(X_full)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.15, random_state=42)

    # Create and fit the XGBoost model
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # Feature Importance Ranking
    feature_importance = model.feature_importances_
    feature_names = data.columns[:-1]
    important_features = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    important_features = important_features.sort_values(by='Importance', ascending=False)
    top_features = important_features['Feature'].head(4).tolist()

    # Retain Top 4 Features
    selected_features = top_features + ['status']
    data_selected = data[selected_features]
    X = data_selected.drop(columns=['status'])
    y = data_selected['status']

    # Normalize selected features
    X = scaler.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Retrain the model with selected features
    model.fit(X_train, y_train)

    # User Input for Prediction
    input_data = np.array([user_input])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    result = "Parkinson's" if prediction[0] == 1 else "No Parkinson's"
    return jsonify({'prediction': result})
