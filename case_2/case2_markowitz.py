from universal import tools, algos
from universal.algos import CRP
import pandas as pd

data = pd.read_csv('Case 2 Data 2024.csv', index_col=0)

algo = algos.BestMarkowitz()
# run
result = algo.run(data)
print("BestMarkowitz")
print(result.summary())