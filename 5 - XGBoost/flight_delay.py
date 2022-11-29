# ------------------ Importing Libraries ------------------

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import json
import pickle

# ------------------ Loading Data -----------------------

df = pd.read_csv("flights.csv", dtype={'ORIGIN_AIRPORT': 'str', 'DESTINATION_AIRPORT': 'str'})
df = df.loc[:10000, :] #lowered data entries to 10000 so that it speed up processing time but feel free to change this value however it will change results. 

# ------------------ Identifying Features and Target Variable ------------------

print(df.shape)
print(df.columns)
print(df.describe())
target = "ARRIVAL_DELAY"
features = ['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'FLIGHT_NUMBER',
       'TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
       'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT',
       'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE',
       'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME',
       'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON',
       'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',
       'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']

for i in ['DIVERTED', 'CANCELLED', 'CANCELLATION_REASON', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY','LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']:
       features.remove(i)

print(features)

# ------------------ Data Manipulation ------------------

print(df[target].isna().sum())
df = df[df[target].notna()]
print(df[target].isna().sum())
print(df[features].isna().sum()) #removing the rows where target was NaN removed all other missing values. Yay! Very lickely data leckage or just not enough information. 

print(df[features].head())
for i in features:
    if df[i].dtype != object:
        int_col = df[i] 
        int_col[int_col < 0] = 0 #if value is negative then assigns 0 as we want delay and not early arrival.

print(df[features].head())

# ------------------ Feature Encoding ------------------

label_encoders = {}

for i in features:
    if df[i].dtype == object:
        labelencoder = LabelEncoder()
        label_encoders[i] = labelencoder
        df[i] = labelencoder.fit_transform(df[i])

# ------------------ Splitting Data into Train and Test datasets ------------------

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.33, random_state=0)

# ------------------ Feature Selection ------------------

print("feature selection...")

rf = RandomForestRegressor(random_state=0) 
rf.fit(X_train,y_train) 
rfe = RFECV(rf,cv=5,scoring="neg_mean_squared_error") 
rfe.fit(X_train,y_train)  
selected_features = rfe.get_feature_names_out(features)
print(selected_features) #['DEPARTURE_DELAY', 'SCHEDULED_TIME', 'ELAPSED_TIME']

# ------------------ Hyperparameter Tuning and Model Fitting ------------------

print("hyperparameter tuning...")

param_grid = {
'max_depth': [i for i in np.arange(3,11,1)],
'gamma': [i for i in np.arange(1,10,1)],
'alpha': [i for i in np.arange(0,10,1)],
"eta": [i for i in np.arange(0, 0.5, 0.05)],
'min_child_weight': [i for i in np.arange(0,10,1)],
"lamba":[i for i in np.arange(0,10,1)]
}

model = xgb.XGBRegressor()
model_grid = RandomizedSearchCV(model, param_grid, random_state=0)
model_grid = model_grid.fit(X_train[selected_features], y_train)
print(model_grid.best_params_) 

param_grid = { 
'max_depth': [1],
'gamma': [i for i in np.arange(8,10,0.5)],
'alpha': [i for i in np.arange(4,6,0.5)],
'min_child_weight': [i for i in np.arange(2,4,0.5)],
"lamba":[i for i in np.arange(1,3,0.5)],
"eta": [i for i in np.arange(0,0.2, 0.05)]
}

model = GridSearchCV(model, param_grid)
model.fit(X_train[selected_features], y_train) 

print(model.best_params_) #'alpha': 4.5, 'eta': 0.15000000000000002, 'gamma': 8.0, 'lamba': 1.0, 'max_depth': 1, 'min_child_weight': 2.0
print("model fitted.") 

# ------------------ Predicting Values with Test Dataset ------------------

predictions = model.predict(X_test[selected_features])
predictions = [value for value in predictions] 

# ------------------ Model Evaluation ------------------

print("Root Mean Square Error: {} minutes".format(mean_squared_error(y_test, predictions)**0.5))
