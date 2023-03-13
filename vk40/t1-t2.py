import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer



df = pd.read_csv('startup.csv')

X = df.iloc[:, :-1]   #muut paitsi profit
y = df.iloc[:, [-1]]  #profit

#muutetaan kategoriset muuttuja (state) dummy muuttujiksi

# dummies_state = pd.get_dummies(X['State'], drop_first=(True))
# X = X.join(dummies_state)
# X.drop('State', inplace=True, axis=1)

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

#df_new_company = pd.read_csv('new_company.csv')
df_new_company = pd.read_csv('new_company_ct.csv')
df_new_company = ct.transform(df_new_company)

y_new = model.predict(df_new_company)
print (f'Uuden yrityksen voitto: {y_new}')