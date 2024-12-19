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

# Load the Pima Indians Diabetes Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=columns)

# Data Preprocessing
# Replace zeros with NaN for specific columns
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_with_zero] = data[cols_with_zero].replace(0, np.nan)

# Impute missing values with median
data[cols_with_zero] = data[cols_with_zero].fillna(data[cols_with_zero].median())

# Separate features and target variable
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']                # Target variable

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

# User Input for Prediction
def predict_diabetes(input_data):
    input_data = pd.DataFrame([input_data])  # Convert input to DataFrame
    input_scaled = scaler.transform(input_data)  # Scale the input
    prediction = model.predict(input_scaled)
    return 'Diabetes' if prediction[0] == 1 else 'No Diabetes'

Pregnancies=int(input("Enter the number of Pregnancies:"))
Glucose=int(input("Enter the number of Glucose:"))
BloodPressure=int(input("Enter the number of BloodPressure:"))
SkinThickness=int(input("Enter the number of SkinThickness:"))
Insulin=int(input("Enter the number of Insulin:"))
BMI=float(input("Enter the number of BMI:"))
DiabetesPedigreeFunction=float(input("Enter the number of DiabetesPedigreeFunction:"))
Age=int(input("Enter the number of Age:"))
user_input = {
    'Pregnancies': Pregnancies,
    'Glucose': Glucose,
    'BloodPressure': BloodPressure,
    'SkinThickness': SkinThickness,
    'Insulin': Insulin,
    'BMI': BMI,
    'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
    'Age': Age
}

result = predict_diabetes(user_input)
print(f'Prediction: {result}')
