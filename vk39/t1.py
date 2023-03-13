import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = pd.Series([1, 2, 3, 4,6, 7, 8])
y =  2 * x + 3

df = pd.DataFrame({'x':x,'y' : y})

df.plot(kind='scatter', x = 'x', y = 'y')
plt.show()

X = df.loc[:,['x']]
y = df.loc[:,['y']]

regr = LinearRegression()
regr.fit(X,y)

y_pred = regr.predict([[4,3]])

coef = regr.coef_
inter = regr.intercept_

print ('Suoran yhtälö on: ')
print (f'y = {coef[0][0]} * x + {inter[9]}')
     
plt.scatter(df.x, df.y)
plt.scatter(4,3, y_pred, c='r')
plt.plot(df.x, df.y)
plt.show()

