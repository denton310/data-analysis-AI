import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle
import matplotlib.pyplot as plt

df = pd.read_csv('housing.csv')

X = df.iloc[:, [2,3,4,7,9]]   #opetus sarakkeet
X = X.fillna(0)               #muutetaan datassa "nan" arvot nollaksi

y = df.iloc[:, [8]]           #talon arvo

X_org = X

#dummyt
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['ocean_proximity'])], remainder='passthrough')
X=ct.fit_transform(X)

# opetusdata ja testidata
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# skaalataan X ja y standard scalerilla
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)



scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# opetetaan neuroverkko
model = Sequential()
model.add(Dense(units=12, input_dim=X.shape[1],activation='relu')) #input+ensimmäinen hidden
# model.add(Dropout(0.2))
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(units=1, activation='linear')) #output



model.compile(loss='mse', optimizer='adam',metrics=['mse'])
history=model.fit(X_train, y_train, epochs=10, batch_size=16,
validation_data=(X_test, y_test))



#oppimisen visualisointi
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# ennustetaan testidatalla
y_pred = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mea = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mea)

print (f'r2: {round(r2,4)}')
print (f'mae: {round(mae,4)}')
print (f'rmse: {round(rmse,4)}')

# tallennetaan malli
model.save('housing-model.h5')

# tallennetaan skaaleri
with open('housing-scaler_X.pickle', 'wb') as f:
    pickle.dump(scaler_X, f)

# tallennetaan skaaleri
with open('housing-scaler_y.pickle', 'wb') as f:
    pickle.dump(scaler_y, f)



# dummy käsittelijä
with open('housing-ct.pickle', 'wb') as f:
    pickle.dump(ct, f)
    