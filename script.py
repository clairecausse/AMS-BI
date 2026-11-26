import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    "base_prospect.csv",
    encoding="latin1",
    sep=None,
    engine="python"
)

print(df.head())

# Dim
print("Nombre de lignes :", df.shape[0])
print("Nombre de colonnes :", df.shape[1])

print(df.info())