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

