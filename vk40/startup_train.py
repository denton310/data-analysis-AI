import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
df = pd.read_csv('startup.csv')

X = df.iloc[:, :-1]   #muut paitsi profit
y = df.iloc[:, [-1]]  #profit

#Parempi tapa hoitaa dummyt
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['State'])], remainder='passthrough')
X=ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


model =  LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print (f'r2: {r2}')
print (f'mae: {mae}')
print (f'rmse: {rmse}')

#tallennetaan malli levylle

with open('startup-model.pickle', 'wb') as f:
    pickle.dump(model, f)
    
#tallennetaan levylle
with open('startup-ct.pickle', 'wb') as f:
    pickle.dump(ct, f)