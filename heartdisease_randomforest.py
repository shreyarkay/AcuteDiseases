# Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load the Heart Disease Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/heart-disease.csv"
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
data = pd.read_csv(url, names=columns)

# Data Preprocessing
# Check for missing values
print(data.isnull().sum())

# Separate features and target variable
X = data.drop('target', axis=1)  # Features
y = data['target']                # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle Class Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Standardize the features
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Create the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model to the training data
model.fit(X_train_resampled, y_train_resampled)

# Use the model to make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', conf_matrix)

class_report = classification_report(y_test, y_pred)
print('Classification Report:\n', class_report)

# User Input for Prediction
def predict_heart_disease(input_data):
    input_data = pd.DataFrame([input_data])  # Convert input to DataFrame
    input_scaled = scaler.transform(input_data)  # Scale the input
    prediction = model.predict(input_scaled)
    return 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'
age=int(input('Enter Age:'))
sex=int(input('Enter Sex 1 as male and enter 0 as female:'))
cp=int(input('Enter Chest Pain Type:'))
trestbps=int(input('Enter Resting Blood Pressure:'))
chol=int(input('Enter Serum Cholesterol:'))
fbs=int(input('Enter Fasting Blood Sugar:'))
restecg=int(input('Enter Resting Electrocardiographic Results:'))
thalach=int(input('Enter Maximum Heart Rate Achieved:'))
exang=int(input('Enter Exercise Induced Angina:'))
oldpeak=float(input('Enter ST Depression Induced by Exercise:'))
slope=int(input('Enter Slope of the Peak Exercise ST Segment:'))
ca=int(input('Enter Number of Major Vessels Colored by Fluoroscopy:'))
thal=int(input('Enter Thallium Stress Test Result:'))
# Example user input
user_input = {
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}

result = predict_heart_disease(user_input)
print(f'Prediction: {result}')
