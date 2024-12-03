
import pandas as pd

# Leggi i due fogli in DataFrame di pandas
df = pd.read_excel("/home/phd2/Scrivania/CorsoRepo/cellPIV/DB morpheus UniPV.xlsx")

print(df["slide_well"].value_counts())




