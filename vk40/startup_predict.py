import pandas as pd
import pickle #load encoder

# ladataan malli levyltä
with open('startup-model.pickle', 'rb') as f:

    model = pickle.load(f)

# load encoder

with open('startup-ct.pickle', 'rb') as f:

    ct = pickle.load(f)

# ennusta uudella datalla
Xnew = pd.read_csv('new_company_ct.csv')
Xnew_org = Xnew
Xnew=ct.transform(Xnew)
ynew= model.predict(Xnew) 

for i in range (len(ynew)):

    print (f'{Xnew_org.iloc[i]}\nVoitto: {ynew[i][0]}\n')
