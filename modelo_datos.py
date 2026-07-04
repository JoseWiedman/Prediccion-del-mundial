import pandas as pd

import os

url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

os.makedirs("data", exist_ok=True)

df= pd.read_csv(url)


df.to_csv("data/results.csv", index=False)


print(df.head())

print("\nArchivo Guardado Correctamente en: data/results.csv")