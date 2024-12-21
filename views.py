from datetime import datetime
from flask import request, render_template, jsonify

#from flask import render_template
from doctor_app import app
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
    return render_template('index.html')


def heart_disease_prediction(user_input):
    # Load the Heart Disease Dataset
    url = "https://www.kaggle.com/api/v1/datasets/download/johnsmith88/heart-disease-dataset"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    data = pd.read_csv(url, names=columns, encoding='latin-1')
    if data.isnull().sum().any():
            data = data.fillna(data.mean())


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
    return 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'


def diabetes_prediction(user_input):
    # Load the Pima Indians Diabetes Dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(url, names=columns)
    if data.isnull().sum().any():
            data = data.fillna(data.mean())


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
    return "Diabetes" if prediction[0] == 1 else "No Diabetes"


def parkinsons_prediction(user_input):
    # Load the Parkinson's Dataset
    dataset_path = 'doctor_app/dataset/parkinsons.data'  # Update this path
    data = pd.read_csv(dataset_path)
    if data.isnull().sum().any():
            data = data.fillna(data.mean())


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
    return "Parkinson's" if prediction[0] == 1 else "No Parkinson's"
