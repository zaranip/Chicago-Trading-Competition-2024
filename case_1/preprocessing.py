import numpy as np
import pandas as pd

df = pd.read_csv("Case1_Historical.csv")
symbols = ["EPT","DLO","MKU","IGM","BRV"]

df['JCR'] = (3*df['EPT'] + 3*df['IGM'] + 4*df['BRV'])/10
df['JAK'] = (2*df['EPT'] + 5*df["DLO"] + 3*df['MKU'])/10

df.to_csv("Case1_Historical_Amended.csv", index=False)
