# Basic File handling
import pandas as pd
import numpy as np

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

# Getting the Data Set in the Program
data=pd.read_csv('dataset.csv')

# Removing all the Null Values
data.dropna(inplace=True)

# Encoding Values to Unique Integers to aid Mathematical Calculations

data['broken']= label_encoder.fit_transform(data['broken']) 
data['team']= label_encoder.fit_transform(data['team']) 
data['provider']= label_encoder.fit_transform(data['provider']) 

data['broken'].unique()
data['team'].unique() 
data['provider'].unique() 

X = data.drop('broken',axis = 1)    #X : Features and Dependent Variables
y = data['broken']    #y : Target Variable

# Splitting the Data for Training and Testing in 80:20 Ratio

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

# Training the Model
model = RandomForestClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

# Accuracy Score
confidence = accuracy_score(y_test,y_pred)*100