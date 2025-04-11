"""
import cv2

print(cv2.cuda.getCudaEnabledDeviceCount())  # Should be >0
"""

"""
import pycuda.driver as cuda
cuda.init()
print("Number of CUDA devices:", cuda.Device.count())
"""
"""
import cv2
print(cv2.getBuildInformation())  # Look for "CUDA" in the output
"""

import pandas as pd
import numpy as np

# Caricare il file Excel
xls = pd.ExcelFile("/home/phd2/Scrivania/CorsoRepo/cellPIV/datasets/DB morpheus UniPV.xlsx", engine='openpyxl')
sheet = pd.read_excel(xls, sheet_name="lista")

print(sheet['PN'].value_counts())


