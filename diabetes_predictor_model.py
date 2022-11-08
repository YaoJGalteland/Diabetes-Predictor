## 1-Import libraries
from numpy import nan
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

## 2-Import dataset
df = pd.read_csv('diabetes.csv')
# display all columns
pd.set_option("display.max_columns", None)

## 3-Exploratory data analysis
# Too many unrealistic zeros, change zero to nan
for col in df:
    if col == 'Pregnancies' or col == 'Outcome':
        continue
    df.loc[df[df[col] == 0].index, col] = nan

## 4- use KNNImputer to impute missing values in step 6

## 5-Set X and y variables
# split into input and output elements
data = df.values
ix = [i for i in range(data.shape[1]) if i != 8]
X, y = data[:, ix], data[:, 8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10, shuffle=True)

## 6-Set algorithm
# define modeling pipeline
# use KNNImputer to impute missing values
# set algorithm by random forest algorithm
pipeline = Pipeline(steps=[('i', KNNImputer(n_neighbors=21)), ('m', RandomForestClassifier())])
# fit the model
pipeline.fit(X_train, y_train)

## 7-Evaluate
model_predict = pipeline.predict(X_test)
print(pipeline.score(X_test,y_test))
print(confusion_matrix(y_test, model_predict))
print(classification_report(y_test, model_predict))

## 8-Predict
input = [0,138,0,0,0,36.3,0.933,25]
for i in range(1, 7):
    if input[i] == 0:
        input[i] = np.nan
print("The prediction is {}.".format(pipeline.predict([input])[0]))

## 9-Save the model to disk
filename = 'finalized_model.pkl'
pickle.dump(pipeline, open(filename, 'wb'))