from datetime import datetime, timedelta
import pandas as pd

df_emp = pd.read_csv("employees.csv", header=0, dtype={'phone1':str, 'phone2':str})
df_dep = pd.read_csv("departments.csv", header=0)

df = pd.merge(df_emp, df_dep, how='inner', on='dep')

df.drop('image', inplace = True, axis = 1)

emp_count = df.shape[0]

rows, cols = df.shape

m_count = sum(df['gender']==0)
f_count = sum(df.gender==1)

m_pros = round(m_count / emp_count * 100,1)
f_pros = round(f_count / emp_count * 100,1)

min_salary = df['salary'].min()
max_salary = df['salary'].max()
avg_salary = df['salary'].mean()

avg_salary_tuotekeh = df[df['dname']=='Tuotekehitys']['salary'].mean()

count_no_phone2 = sum(df['phone2'].isna())

df['age'] = pd.to.datetime(df['bdate']).map(lambda x: (datetime.now() -  x) // timedelta(days=365.2425))