
import sys
import pandas as pd


csv_file = "/home/phd2/Scrivania/CorsoRepo/cellPIV/_02_temporalData/FinalBlastoLabels.csv"
datasets = pd.read_csv(csv_file)

nunq = datasets.iloc[:,0].nunique()
print(nunq)

dim = datasets.shape
print(dim)

