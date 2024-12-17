import pandas as pd

df = pd.read_csv("/home/phd2/Scrivania/CorsoRepo/cellPIV/Normalized_train_7Days_sum_mean_mag.csv")
df2 = pd.read_csv("/home/phd2/Scrivania/CorsoRepo/cellPIV/Normalized_test_7Days_sum_mean_mag.csv")

print(df.value_counts("BLASTO NY"))
print(df2.value_counts("BLASTO NY"))
