import pandas as  pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot  as plt

df = pd.read_csv('titanic-class-age-gender-survived.csv')

#X = df.iloc[:, :-1]
X = df.iloc[:, [0,1,2]]
y = df.iloc[:,[ -1]]

X_org = X
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['PClass','Gender'])], remainder='passthrough',)
X=ct.fit_transform(X) #ensimmäisellä kerralla fit_transform

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=(0))

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test) #model.predict_proba antaa pohjat josta yes/no tulee

cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred, 2)
ps = precision_score(y_test, y_pred)
rc = recall_score(y_test, y_pred)

print (f'{cm}')
print (f'Mallin ulkoinen tarkkuus: {ac*100:.02f}%')  #:.04f komennolla saa rajattua desimaaleja
print (f'precision_score: {ps:.04f}')
print (f'recall_score: {rc:.04f}')

sns.heatmap(cm, annot=True, fmt='g')
plt.show()

tn, fp, fn, tp = cm.ravel()

#kokeillaan mallia uudella datalla

df_new = pd.read_csv('titanic-new.csv')
df_new_org = df_new
df_new = df_new.iloc[:, [0,1,2]]
df_new = ct.transform((df_new))

y_new = model.predict(df_new)
y_new_proba = model.predict_proba(df_new)

for i in range (len(y_new)):
    print (f'{df_new_org.iloc[i]}\nSelviytyminen: {y_new[i]} ({y_new_proba[i][1]:.02f})')
    
    
    
    
    
    
    