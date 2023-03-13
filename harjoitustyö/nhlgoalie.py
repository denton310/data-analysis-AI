import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

#The basic stats for each goalie that had some ice time in each game. 
#If the team's "backup" goalie is not used they are not recorded in this table.

df = pd.read_csv('game_goalie_stats.csv')

X = df.iloc[:,[2,3,7,8,15]]
y = df.iloc[:,[-3]]                 #save percentage

# find min and max values for each column, ignoring nan, -inf, and inf
mins = [np.nanmin(X[:, i][X[:, i] != -np.inf]) for i in range(X.shape[1])]
maxs = [np.nanmax(X[:, i][X[:, i] != np.inf]) for i in range(X.shape[1])]

# go through matrix one column at a time and replace  + and -infinity 
# with the max or min for that column
    # for i in range(X.shape[1]):
    # X[:, i][X[:, i] == -np.inf] = mins[i]
    # X[:, i][X[:, i] == np.inf] = maxs[i]

# X.any(X.isnan(mat))

X_data = X

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop=('first')), ['decision'])], remainder='passthrough')
X = ct.fit_transform(X.astype(str))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test) 

dataval = [X_data['timeOnIce'].mean(), #average time on ice
           X_data['game_id'].max(), # Maximum of the column values
           X_data['game_id'].min()] # Minimum of the column values


# print(f'Keskimääräinen jääaika: {dataval[timeOnIce]}')

