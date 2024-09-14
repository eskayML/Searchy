import pandas as pd
import json, ast

df = pd.read_parquet('captioned_ecom.parquet')

print(ast.literal_eval(df["description"].values[0]))
for index,item in df.iterrows():
    for key in ast.literal_eval(item["description"]):
        print(list(key.items())[0])
    print('\n'*2)