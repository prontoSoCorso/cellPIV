import pandas as pd

xls = pd.ExcelFile("BlastoLabels.xlsx", engine='openpyxl')

df = pd.read_excel("BlastoLabels.xlsx")

sheet1 = pd.read_excel(xls, sheet_name="blasto NO SI")
sheet2 = pd.read_excel(xls, sheet_name="blasto NON in foglio 1 ")

num_video_first_sheet = sheet1.value_counts("BLASTO NY")
num_video_second_sheet = sheet2.value_counts("presente in foglio 1")

print(num_video_first_sheet)
print(num_video_second_sheet)

print(2015+1874+917)