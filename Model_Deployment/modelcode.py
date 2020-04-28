'''
Importing Libraries
'''

# Basic File handling
import pandas as pd
import numpy as np

# Import Data from URL
import io
import requests

# Pre-Processing
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

# Splitting data for Model Training
from sklearn.model_selection import train_test_split 

# Training Model
from sklearn.ensemble import RandomForestClassifier
# Random Forest Classifier was chosen because it was most Accurate

# Model Evaluation : Accuracy Score
from sklearn.metrics import accuracy_score

# Export Trained Model as *.pkl
import joblib


'''
Getting the Data Set in the Program
'''
url = 'https://raw.githubusercontent.com/iSiddharth20/Predictive-Analysis-for-Machine-Faliure/master/dataset.csv'
s = requests.get(url).content
data = pd.read_csv(io.StringIO(s.decode('utf-8')))


'''
Removing all the Null Values and Resetting Index of the Data Frame
'''
data.dropna(inplace=True)
data.reset_index(inplace=True)
data.drop(columns=['index'],axis=1,inplace=True)


'''
Removing Outliers using InterQuartile Range
Q1 : First Quartile
Q2 : Second Quartile
IQR : Inter Quartile Range

Only data points within the Inter Quartile Range will be stored
'''

# Finding Key Values
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75) 
IQR = Q3 - Q1

# Selecting Valid Data Points
data = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]

# Resetting Index of Data Frame
data.reset_index(inplace=True)
data.drop(columns=['index'],axis=1,inplace=True)


'''
Training the Model
'''

# Encoding Values to Unique Integers to aid Mathematical Calculations
data['broken'] = label_encoder.fit_transform(data['broken']) 
data['broken'].unique()

# X : Features and Dependent Variables
# y : Target Variable
X = data.drop('broken',axis = 1)
Y = data['broken']

# Splitting the Data for Training and Testing in 70:30 Ratio
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 0)

# Using Random Forest
model = RandomForestClassifier(max_depth=10, random_state=0)
model.fit(X_train,Y_train)

# Testing the Model
Y_pred = model.predict(X_test)

# Confidence / Accuracy
confidence = round(accuracy_score(Y_test,Y_pred)*100,3)

