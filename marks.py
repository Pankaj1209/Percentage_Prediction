import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("student_info.csv")
#df = pd.read_csv("student_info.csv")
df2 = df.fillna(df.mean())
X = df2.drop("student_marks",axis = "columns")
y = df2.drop("study_hours",axis = "columns")
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 51)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
lr.predict([[4]])[0][0].round(2)
import pickle
pickle.dump(lr,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
