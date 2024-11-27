

import sys
import pandas as pd


csv_file = "BlastoLabels_singleFileWithID.csv"
datasets = pd.read_csv(csv_file)

nunq = datasets.iloc[:,0].nunique()
print(nunq)