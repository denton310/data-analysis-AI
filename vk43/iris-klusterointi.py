import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('iris.csv')

X = df.drop('species', axis =1)

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Klustereiden määrä')
plt.ylabel('Neliösumma')
plt.show()

#päädytään klustereiden määrään 3
model = KMeans(n_clusters = 3, random_state = 0)
model.fit(X)

#Ennustetaan klusterit
y_pred = model.predict(X)

X['C'] = y_pred
X.loc[X['C'] == 0, 'pred_s'] = 'versicolor'
X.loc[X['C'] == 1, 'pred_s'] = 'setosa'
X.loc[X['C'] == 2, 'pred_s'] = 'virginica'
X['real_s'] = df['species']

print()
print(pd.crosstab(X['pred_s'], X['real_s']))