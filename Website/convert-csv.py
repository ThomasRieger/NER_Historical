import pandas as pd

df = pd.read_excel('./DATA/Data_original.xlsx', header=0)
df.to_csv('./DATA/data_v1.csv', index=False)