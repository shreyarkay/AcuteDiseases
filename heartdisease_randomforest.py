import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
hdata=pd.read_csv('/content/heart.csv')
hdata.head()
hdata.tail()
hdata.shape
hdata.info()
hdata.isnull().sum()
hdata.describe()
hdata['target'].value_counts()
X=hdata.drop(columns='target',axis=1)
Y=hdata['target']
X
Y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)
Y_pred=model.predict(X_test)
acc=accuracy_score(Y_test,Y_pred)
acc
inputData=(75,0,2,145,233,1,0,150,0,2.3,0,0,1)
input_array_data=np.asarray(inputData)
input_data_reshaped=input_array_data.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
prediction
if (prediction[0]==1):
  print('The Person has a Heart Disease')
else:
  print('The Person does not have Heart Disease')
